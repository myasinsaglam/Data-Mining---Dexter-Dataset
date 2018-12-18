import random
from keras.layers import Dense
import numpy as np
from keras import Sequential
from keras.models import load_model
import copy
from collections import defaultdict
import matplotlib.pyplot as plt

random.seed(1234)


# v1 = x axis, v2 = y axis
def plotter(v1, v2, label_v1, label_v2, figname):
    plt.plot(v1, v2)
    plt.xlabel(label_v1)
    plt.ylabel(label_v2)
    plt.title(figname)
    plt.grid(True)
    plt.savefig(figname+".png")
    plt.show()


# A function that calculates Euclidean Distance between two vectors
def calculate_euclidean_distance(vector1, vector2):
    euclidean_distance = vector1 - vector2
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


# A function that returns similarity values for K-Means algorithm
def calculate_similarities(sample, centroids):
    similarities = []
    for centroid in centroids:
        similarity = calculate_euclidean_distance(sample, centroid)
        similarities.append(similarity)
    return np.array(similarities)


def k_means(k_val, dataset):
    centroids = []
    feature_num = len(dataset[0])
    data_num = len(dataset)
    prev_assignments = np.ones(data_num) * (-1)
    iteration = 0

    print("Dataset length : ", data_num)
    # Initial centroids chosen from dataset
    for i in range(k_val):
        index = random.randint(0, data_num - 1)
        centroids.append(dataset[index])

    while True:
        iteration += 1
        print("Iter : ", iteration)

        assignments = []

        # Similarity calculation to centroids for each instance
        for i in range(data_num):
            similarities = calculate_similarities(dataset[i], centroids)
            # centroid assignment vector creating for instances
            assignments.append(np.argmin(similarities))

        # Algorithm stops when assignments not changed
        if list(prev_assignments) == list(assignments):
            return centroids, list(assignments)

        new_centroids = np.zeros((k_val, feature_num), np.float)
        cls_dist = np.zeros(k_val)
        # sum operation of centroids
        for i in range(data_num):
            new_centroids[assignments[i]] += dataset[i]
            cls_dist[assignments[i]] += 1
        # Mean operation of centroids
        print("Centroid distribution : ", cls_dist)
        for i in range(k_val):
            new_centroids[i] = new_centroids[i] / (cls_dist[i]+0.0001)  # To prevent zero division

        centroids = copy.deepcopy(new_centroids)
        prev_assignments = copy.deepcopy(assignments)


# Running k_means++ algorithm with calculating accuracy implementation mentioned in lecture
def run_kmeans(k_value, data_train, label_train, data_test, label_test):
    # Centroid values and index assignments taking as an output of k-means algorithm
    centroids, assigments = k_means(k_value, data_train)

    # Calculating accuracy of centroids
    real_binary_labels = []
    for item in label_train:
        real_binary_labels.append(np.argmax(item))

    # Calculating centroid actual label distribution
    cent_dist = {}
    for i in range(k_value):
        cent_dist[i] = np.zeros(label_train.shape[1])

    print(cent_dist)
    for i in range(len(real_binary_labels)):
        cent_dist[assigments[i]][real_binary_labels[i]] += 1

    print(cent_dist)

    centroid_lookup = {}
    for i in range(k_value):
        centroid_lookup[i] = np.argmax(cent_dist[i])

    print(centroid_lookup)

    predicted = []
    real = []
    for i in range(len(data_test)):
        nearest_centroid_index = calculate_neighbours(1, data_test[i], centroids)[0]
        predicted.append(centroid_lookup[nearest_centroid_index])
        real.append(np.argmax(label_test[i]))

    print_confusion_matrix(real, predicted)

    return real, predicted


# A function that takes k_val,test sample and dataset as an argument and return k-nearest data indexes from dataset
def calculate_neighbours(k_val, sample, dataset):
    similarities = []
    # Calculating distances
    for item in dataset:
        similarity = calculate_euclidean_distance(sample, item)
        similarities.append(similarity)

    # for item in similarities[:10]:
    #     print(item)
    # Finding indexes of k-nearest samples
    indexes = np.arange(0, len(similarities))
    datas = copy.deepcopy(similarities)
    for i in range(k_val):
        for j in range(len(datas) - i):
            j += i
            if datas[j] < datas[i]:
                datas[i], datas[j] = datas[j], datas[i]
                indexes[i], indexes[j] = indexes[j], indexes[i]

    # print(datas[:k_val])
    # print(indexes[:k_val])

    return indexes[:k_val]


# A function that classifies given instance by using knn algorithm
def knn_classify_instance(k_val, dataset, labels, instance):
    result_indexes = calculate_neighbours(k_val, instance, dataset)
    k_predicted = []
    for item in result_indexes:
        k_predicted.append(np.argmax(labels[item]))
    # print(k_predicted)
    freq = defaultdict(int)
    for item in k_predicted:
        freq[item] += 1
    freq_dict = dict(sorted(freq.items(), key=lambda kv: kv[1], reverse=True))
    # print("freqs :", freq_dict)
    return list(freq_dict.keys())[0]


# A function that classifies given test set and calculates confusion matrix of results according to K-NN algorithm.
def knn_classifier(k_val, train_data, train_label, test_data, test_label):
    predictions = []
    reals = []
    for item in test_label:
        reals.append(np.argmax(item))

    for i in range(len(test_data)):
        item = test_data[i]
        result = knn_classify_instance(k_val, train_data, train_label, item)
        predictions.append(result)
    accuracy = print_confusion_matrix(reals, predictions)
    return accuracy


def train_mlp_classifier(x_train, y_train, x_test, y_test):
    """

    :param x_train: train datas
    :param y_train: train labels
    :param x_test: test datas
    :param y_test: test labels
    """
    model = Sequential()
    # model.add(Input(shape=(20000),dtype=float))
    model.add(Dense(32, input_dim=x_train.shape[1]))
    model.add(Dense(64))
    model.add(Dense(128))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    print('Train...')
    model.fit(x_train,
              y_train,
              batch_size=16,
              epochs=8,
              validation_data=[x_test, y_test])

    model.save("dexter_mlp_model.h5")


def test_mlp_classifier(dir_model, test_data, test_label):
    predicts = []
    reals = []
    # misclassified = 0
    # print(test_data[0].shape)
    model = load_model(dir_model)
    for i in range(len(test_data)):
        item = np.expand_dims(test_data[i], 0)
        # result = np.zeros(test_label.shape[1])
        # result[np.argmax(model.predict(item)[0])] = 1
        predicts.append(np.argmax(model.predict(item)[0]))
        reals.append(np.argmax(test_label[i]))
        # if predicts[i] != reals[i]:
        #     misclassified += 1
    # acc = 100 - float(misclassified / len(test_data) * 100)
    # print("Total : ", len(test_data), " Misclassified : ", misclassified, " Accuracy : % ", format(acc, '.2f'))
    # print(reals)
    # print(predicts)
    return reals, predicts


# Return confusion matrix values for binary Labelled results
def print_confusion_matrix(reals, predicts):
    tp = 0  # True Positive, Actual == Pred == 1
    tn = 0  # True Negative, Actual == Pred == 0
    fp = 0  # False Positive, Actual == 0 , Pred == 1
    fn = 0  # False Negative, Actual == 1 , Pred == 0

    for j in range(len(reals)):
        if reals[j] == 1 and reals[j] == predicts[j]:
            tp += 1
        if reals[j] == 0 and reals[j] == predicts[j]:
            tn += 1
        if reals[j] == 1 and predicts[j] == 0:
            fn += 1
        if reals[j] == 0 and predicts[j] == 1:
            fp += 1

    print("\n****\nTP : ", tp, "\t FP : ", fp, "\nFN : ", fn, "\t TN : ", tn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1_score = (2*precision*recall)/(precision+recall)

    print("****\n\nAccuracy : ", format(accuracy*100, '.2f'), " %\nPrecision : ", format(precision*100, '.2f'),
          " %\nRecall : ", format(recall*100, '.2f'), " %\nF1-Score : ", format(f1_score*100, '.2f'), " %")
    return accuracy


if __name__ == '__main__':
    dir_data = "/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/"
    data_train = np.load(dir_data + "non_norm_train_data.npy")
    label_train = np.load(dir_data + "non_norm_train_label.npy")
    data_test = np.load(dir_data + "non_norm_test_data.npy")
    label_test = np.load(dir_data + "non_norm_test_label.npy")

    mlp_train_data = np.load(dir_data + "train_data.npy")
    mlp_train_label = np.load(dir_data + "train_label.npy")
    mlp_test_data = np.load(dir_data + "valid_data.npy")
    mlp_test_label = np.load(dir_data + "valid_label.npy")

    options = [0,0,1]

    if options[0] == 1:
        # K-NN CLASSIFICATION ANALYSIS
        k_values = np.arange(1, 15, 2)
        results = []
        print("KNN CLASSIFIER RESULTS : ")
        for k_value in k_values:
            print("\n\nK-VALUE : ", k_value)
            results.append(knn_classifier(k_value, data_train, label_train, data_test, label_test))

        plotter(k_values, results, "K Value", "Accuracy", "K-NN-Accuracy")

    if options[1] == 1:
        # K-MEANS CLUSTERING ANALYSIS
        k_values = np.arange(2, 11, 1)
        results_kmeans = []
        print("K-MEANS++ CLUSTERING RESULTS : ")
        for k_value in k_values:
            print("\n\nK-VALUE : ", k_value)
            actual, predicted = run_kmeans(k_value, data_train, label_train, data_test, label_test)
            accuracy = print_confusion_matrix(actual, predicted)
            results_kmeans.append(accuracy)

        plotter(k_values, results_kmeans, "K Value", "Accuracy", "K-Means-Accuracy")


    if options[2] == 1:
        # Classifier MLP
        # If model will be re-trained, the commented line below must be opened
        # train_mlp_classifier(x_train=mlp_train_data, y_train=mlp_train_label, x_test=mlp_test_data, y_test=mlp_test_label)
        dir_classifier_model = "/Users/Macbook/Desktop/ComputerEngineering/Guz2019/DataMining/Project/dexter_mlp_model.h5"
        actuals, predictions = test_mlp_classifier(dir_model=dir_classifier_model,
                                                   test_data=mlp_test_data, test_label=mlp_test_label)
        print("MLP CLASSIFIER RESULTS : ")
        print_confusion_matrix(actuals, predictions)
