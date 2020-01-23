import seaborn as sns
import matplotlib.pyplot as plt


def plot_history(stack, filename=None):
    sns.set_context("poster")
    plt.figure(figsize=(10, 12))

    epochs = len(stack.history['acc'])
    e = range(epochs)

    plt.subplot(2, 1, 1)
    sns.lineplot(x=e, y=stack.history['acc'], label="acc", color='darkcyan')
    sns.lineplot(x=e, y=stack.history['val_acc'], label='val_acc', color='coral')
    plt.title("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    sns.lineplot(x=e, y=stack.history['loss'], label="loss", color='darkcyan')
    sns.lineplot(x=e, y=stack.history['val_loss'], label='val_loss', color='coral')
    plt.title("loss")
    plt.xlabel("epoch")
    plt.legend(loc='best')

    if filename is not None:
      plt.savefig(filename)

    plt.show()
    plt.figure()
