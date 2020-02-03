# https://github.com/kinziro/keras-grad-cam/blob/master/grad-cam.py
# の実装をtensorflow.keras用に書き直した
from tensorflow.keras.layers import Lambda
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def register_gradient():
    # GuidedBackPropが登録されていなければ登録
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        # 自作勾配を登録するデコレーター
        # 今回は_GuidedBackProp関数を"GuidedBackProp"として登録
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            '''逆伝搬してきた勾配のうち、順伝搬/逆伝搬の値がマイナスのセルのみ0にして逆伝搬する'''
            dtype = op.inputs[0].dtype
            # grad : 逆伝搬してきた勾配
            # tf.cast(grad > 0., dtype) : gradが0以上のセルは1, 0以下のセルは0の行列
            # tf.cast(op.inputs[0] > 0., dtype) : 入力のが0以上のセルは1, 0以下のセルは0の行列
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def grad_cam(input_model, img, category_index, layer_name, nb_classes=2):
    '''
    Parameters
    ----------
    input_model : model
        評価するKerasモデル
    img : tuple等
        入力画像(枚数, 縦, 横, チャンネル)
    category_index : int
        入力画像の分類クラス
    layer_name : str
        最後のconv層の後のactivation層のレイヤー名.
        最後のconv層でactivationを指定していればconv層のレイヤー名.
        batch_normalizationを使う際などのようなconv層でactivationを指定していない場合は、
        そのあとのactivation層のレイヤー名.
    nb_classes : int, optional
        分類クラス数
    Returns
    ----------
    cam : tuple
        Grad-Camの画像
    heatmap : tuple
        ヒートマップ画像
    '''

    # ----- 1. 入力画像の予測クラスを計算 -----

    # 入力のcategory_indexが予想クラス

    # ----- 2. 予測クラスのLossを計算 -----

    # 入力データxのcategory_indexで指定したインデックス以外を0にする処理の定義
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)

    # 引数のinput_modelの出力層の後にtarget_layerレイヤーを追加
    # modelのpredictをすると予測クラス以外の値は0になる
    x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(x)
    model = keras.models.Model(input_model.layers[0].input, x)

    # 予測クラス以外の値は0なのでsumをとって予測クラスの値のみ抽出
    loss = K.sum(model.layers[-1].output)
    # 引数のlayer_nameのレイヤー(最後のconv層)のoutputを取得する
    conv_output = [l for l in model.layers if l.name is layer_name][0].output

    # ----- 3. 予測クラスのLossから最後のconv層への逆伝搬(勾配)を計算 -----

    # 予想クラスの値から最後のconv層までの勾配を計算する関数を定義
    # 定義した関数の
    # 入力 : [判定したい画像.shape=(1, 224, 224, 3)]、
    # 出力 : [最後のconv層の出力値.shape=(1, 14, 14, 512), 予想クラスの値から最後のconv層までの勾配.shape=(1, 14, 14, 512)]
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    # 定義した勾配計算用の関数で計算し、データの次元を整形
    # 整形後
    # output.shape=(14, 14, 512), grad_val.shape=(14, 14, 512)
    output, grads_val = gradient_function([img])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    # ----- 4. 最後のconv層のチャンネル毎に勾配を平均を計算して、各チャンネルの重要度(重み)とする -----

    # weights.shape=(512, )
    # cam.shape=(14, 14)
    # ※疑問点1：camの初期化はzerosでなくて良いのか?
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    # cam = np.zeros(output.shape[0 : 2], dtype = np.float32)    # 私の自作モデルではこちらを使用

    # ----- 5. 最後のconv層の順伝搬の出力にチャンネル毎の重みをかけて、足し合わせて、ReLUを通す -----

    # 最後のconv層の順伝搬の出力にチャンネル毎の重みをかけて、足し合わせ
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # 入力画像のサイズにリサイズ(14, 14) → (224, 224)
    cam = cv2.resize(cam, (64, 64))
    # 負の値を0に置換。処理としてはReLUと同じ。
    cam = np.maximum(cam, 0)
    # 値を0~1に正規化。
    # ※疑問2 : (cam - np.min(cam))/(np.max(cam) - np.min(cam))でなくて良いのか?
    heatmap = cam / np.max(cam)
    # heatmap = (cam - np.min(cam))/(np.max(cam) - np.min(cam))    # 私の自作モデルではこちらを使用

    # ----- 6. 入力画像とheatmapをかける -----

    # 入力画像imageの値を0~255に正規化. image.shape=(1, 224, 224, 3) → (224, 224, 3)
    # Return to BGR [0..255] from the preprocessed image
    img = img[0, :]
    img -= np.min(img)
    # ※疑問3 : np.uint8(image / np.max(image))でなくても良いのか?
    img = np.minimum(img, 255)

    # heatmapの値を0~255にしてカラーマップ化(3チャンネル化)
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    # 入力画像とheatmapの足し合わせ
    cam = np.float32(cam) + np.float32(img)
    # 値を0~255に正規化
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap
