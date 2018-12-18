
def train(train_datas,train_labels, epoch_num, learning_rate, test_datas,test_labels):
    plot_x = []
    plot_y = []
    sig = lambda t: 1 / (1 + np.exp(-t))

    # My given network 3 layer with nodes per layers are layer1 = 5, layer2 = 5, layer3 = 7
    l1_w = np.random.uniform(0, 1, (20000, 2))  # input - layer 1 weights
    l2_w = np.random.uniform(0, 1, (2, 4))  # layer1 - layer 2 weights
    l3_w = np.random.uniform(0, 1, (4, 8))  # layer2 - layer3 weights
    l4_w = np.random.uniform(0, 1, (8, 1))  # layer3 - output_layer weights
    all_weights = [l1_w, l2_w, l3_w, l4_w]

    for epoch in range(0, epoch_num):
        for i in range(0, len(train_datas)):
            inputs = train_datas[i]
            # print(inputs)
            inputs = inputs[np.newaxis]
            if np.argmax(train_labels[i]) == 1:
                output = 1
            else:
                output = -1
            # print(train_labels[i],output)

            # output = train_labels[i]
            # print(output)
            # input("wait")

            # Calculating layer outputs
            l1_o = sig(np.dot(inputs, l1_w))
            l2_o = sig(np.dot(l1_o, l2_w))
            l3_o = sig(np.dot(l2_o, l3_w))
            l4_o = np.tanh(np.dot(l3_o, l4_w))

            # res = np.argmax(l4_o[0])
            # print(output," <---> ",l4_o[0])
            # print("Real : ", np.argmax(output), "Pred : ",res)

            # print("Output : ",l4_o, "Class : ",res)

            error = output - l4_o[0][0]
            # print("Err : ", error)
            # input("wait")
            # print("Error : ",error," Sum : ",np.sum(error)," Real vec : ",output, "Predict : ",l4_o, "1-l4_o",1-l4_o)

            # Calculating Deltas
            l4_d = (-(error)) * l4_o[0][0] * (1 - l4_o[0][0])
            # print("l4-d",l4_d)
            l3_d = np.dot(l4_d, l4_w.T) * (l3_o) * (1 - l3_o)
            l2_d = np.dot(l3_d, l3_w.T) * (l2_o) * (1 - l2_o)
            l1_d = np.dot(l2_d, l2_w.T) * (l1_o) * (1 - l1_o)

            # Updating Weights
            l4_w -= np.dot(l3_o.T, l4_d) * learning_rate
            l3_w -= np.dot(l2_o.T, l3_d) * learning_rate
            l2_w -= np.dot(l1_o.T, l2_d) * learning_rate
            l1_w -= np.dot(inputs.T, l1_d) * learning_rate

            # weights will be passed into test function
            all_weights[0] = l1_w
            all_weights[1] = l2_w
            all_weights[2] = l3_w
            all_weights[3] = l4_w

        acc = test(weights=all_weights, test_datas=test_datas,test_labels=test_labels)
        print("EPOCH ", epoch, "Accuracy : ", acc)
        plot_x.append(epoch + 1)
        plot_y.append(acc)
        if epoch%100 == 0:
            learning_rate /= 5
        # print(test(weights=weights,test_objects=test_objects))
    # print("Val",test(weights=weights,test_objects=test_objects))
    # confusion_matrix(test_objects)
    plt.plot(plot_x, plot_y)
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.title("Figure 1 st PART OF HW")
    plt.show()
    return all_weights

def test(test_datas,test_labels, weights):
    misclassified = 0
    sig = lambda t: 1 / (1 + np.exp(-t))
    predicts = []
    reals = []
    for i in range(0, len(test_datas)):
        inputs = test_datas[i]
        inputs = inputs[np.newaxis]
        if np.argmax(test_labels[i]) == 1:
            output = 1
        else:
            output = -1
        # output = test_labels[i]

        # Calculating layer outputs
        l1_o = sig(np.dot(inputs, weights[0]))
        l2_o = sig(np.dot(l1_o, weights[1]))
        l3_o = sig(np.dot(l2_o, weights[2]))
        l4_o = np.tanh(np.dot(l3_o, weights[3]))

        print( l4_o)
        # predicted = np.argmax(l4_o[0])
        # real = np.argmax(output)
        print("l4_0 : ",l4_o[0][0])
        if l4_o[0][0] >= 0:
            predicted = 1
        else:
            predicted = -1

        real = output
        print("P : ",predicted," R : ",real)
        predicts.append(predicted)
        reals.append(real)
        # print("Output layer : ", l4_o, "Predicted : ",predicted," Name : ",test_objects[i]['file_dir'].split("/")[-2], "Real : ",real)
        # x
        if predicted != real:
            # print(test_objects[i]['file_dir'].split("/")[-1])
            misclassified += 1
    accuracy = (1.0 - (misclassified / len(test_datas)))
    print("Accuracy is ",accuracy)
    # print(accuracy_score(reals,predicts)) #Odevde Kullanilmasi gerektigi yazildigi icin kullanilmistir.
    return accuracy * 100, accuracy_score(reals, predicts) * 100
