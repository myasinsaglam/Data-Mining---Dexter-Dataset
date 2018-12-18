import numpy as np


filename  = "/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/dexter_valid.txt"
filename_label = "/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/dexter_valid_labels.txt"

def create_dataset(filename,filename_label,output_name):

    with open(filename,"r") as file:
        lines = file.readlines()

    # print(len(lines))
    with open(filename_label,"r") as file_label:
        labels = file_label.readlines()
        labels = [int(label[:-1]) for label in labels]

    # All labels
    all_labels  = []

    for item in labels:
        # print(item)
        if item == -1 :
            # print("1,0")
            all_labels.append([1,0])
        else:
            # print("0,1")
            all_labels.append([0,1])
        # input("Enter")

    all_labels = np.array(all_labels)

    # print("Label shape : ",all_labels.shape)


    # print(len(labels),labels)

    # input("Control")

    total_keyword = 20000
    # Keyword index range 0-19999 , total : 20000 keywords
    # base = np.zeros(total_keyword)
    all_datas = []
    # print(len(base))
    # input("Base ")
    # minval = 900000
    max_word_index = 0
    for line in lines:
        base = np.zeros(total_keyword)
        sum_freq = 0
        # print(line)
        freq = line[:-2].split(" ")
        for item in freq:
            print("index : ",int(item.split(":")[0]), " Value : ", int(item.split(":")[1]))
            base[int(item.split(":")[0])] = int(item.split(":")[1])
            sum_freq+=int(item.split(":")[1])
            print("Freq sum ",sum_freq)
        base = base/sum_freq
        # for i in base:
        #     print(i)
        different_word_num = len(freq)
        # print(freq)
        # print(freq[-1])
        # # print(different_word_num)
        # if different_word_num < minval:
        #     minval = different_word_num
        all_datas.append(base)
        # print(all_datas)

        # input("wait")

    # print(len(all_datas))
    all_datas = np.array(all_datas)
    # print("Data shape : ",all_datas.shape)
    # print("Min val : " , minval)

    np.save(output_name+"_data.npy",all_datas)
    np.save(output_name+"_label.npy",all_labels)
    # input()