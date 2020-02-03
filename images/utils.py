# txtファイルからリストを読み込む
def get_list_from_txt(filepath: str):
    f = open(filepath)
    lines = f.readlines()
    f.close()

    list_ = []
    for line in lines:
        list_.append(line.replace("\n", ""))

    return list_
