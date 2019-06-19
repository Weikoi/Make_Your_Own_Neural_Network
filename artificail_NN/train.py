from artificail_NN.NeuralNetwork import neuralNetwork as nn
import numpy as np

inputnodes = 784
hiddennodes = 200
outputnodes = 10
lr = 0.2

net = nn(inputnodes, hiddennodes, outputnodes, lr)

with open("../data/mnist_train.csv", 'r') as f:
    data_list = f.readlines()

    epochs = 5

    for e in range(epochs):
        for record in data_list:
            all_values = record.split(",")

            scaled_input = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
            targets = np.zeros(outputnodes) + 0.01
            targets[int(all_values[0])] = 0.99
            net.train(scaled_input, targets)
        print("已经训练第%d世代\n" % (e+1))

with open("../data/mnist_test.csv", 'r') as file:
    test_data_list = file.readlines()
    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = net.query(inputs)
        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        # append correct or incorrect to list
        if label == correct_label:
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass

        pass

scorecard_array = np.asarray(scorecard)
print("预测准确率为", scorecard_array.sum() / scorecard_array.size)
