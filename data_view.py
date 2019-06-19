
import matplotlib.pyplot as plt
import numpy as np

with open("./data/mnist_train_100.csv", 'r') as f:
    data_list = f.readlines()
    all_values = data_list[4].split(",")

    image_array = np.asfarray(all_values[1:]).reshape(28, 28)
    plt.imshow(image_array, cmap="Greys", interpolation="None")
    plt.show()

