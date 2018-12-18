import numpy as np


filename_train_data = "/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/dexter_train.txt"
filename_train_labels = "/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/dexter_train_labels.txt"

filename_test_data = "/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/dexter_valid.txt"
filename_test_labels = "/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/dexter_valid_labels.txt"


# One hot encoded labelling format : -1 is [1,0] , +1 is [0,1]

def create_dataset(filename, filename_label, output_name, norm=True):

    with open(filename, "r") as file:
        lines = file.readlines()

    with open(filename_label, "r") as file_label:
        labels = file_label.readlines()
        labels = [int(item[:-1]) for item in labels]

    # All labels
    all_labels = []

    for item in labels:
        if item == -1:
            all_labels.append([1, 0])
        else:
            all_labels.append([0, 1])

    all_labels = np.array(all_labels)

    total_keyword = 20000
    all_datas = []
    for line in lines:
        base = np.zeros(total_keyword)
        sum_freq = 0
        freq = line[:-2].split(" ")
        for item in freq:
            # print("index : ",int(item.split(":")[0]), " Value : ", int(item.split(":")[1]))
            base[int(item.split(":")[0])] = int(item.split(":")[1])
            sum_freq += int(item.split(":")[1])
            # print("Freq sum ",sum_freq)
        if norm:
            base = base/sum_freq
        all_datas.append(base)

    all_datas = np.array(all_datas)

    np.save(output_name+"_data.npy", all_datas)
    np.save(output_name+"_label.npy", all_labels)

create_dataset(filename_train_data, filename_train_labels, "non_norm_train", norm=False)
create_dataset(filename_test_data, filename_test_labels, "non_norm_test", norm=False)

create_dataset(filename_train_data, filename_train_labels, "train", norm=True)
create_dataset(filename_test_data, filename_test_labels, "test", norm=True)

# label = np.load("/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/deneme_val_label.npy")
# data = np.load("/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/deneme_val_data.npy")

# for i in range(len(label)):
#     print(i, " . ", label[i])
#     print(data[i])
