import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curve():

    # Load data.
    data_p = os.path.join("..", "data", "model_accuracy.csv")
    print("read {}".format(data_p))
    df = pd.read_csv(data_p)

    # Plot train, test data.
    for phase in ["train", "test"]:
        slc = df[df.phase == phase]
        plt.scatter(slc.epoch, slc.accuracy, label=phase)
        plt.plot(slc.epoch, slc.accuracy, ls='--')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Percent Accuracy")
    plt.show()

if __name__ == "__main__":
    plot_learning_curve()
