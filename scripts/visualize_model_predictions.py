"""
Visualize random predictions of the serialized model.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
from PIL import Image 

def visualize_predictions():
    """
    Visualize model predictions for a few example test images.
    """

    # Load most recent model.
    print("load most recent model")
    model_p = os.path.join("..", "models", "latest_model.p")
    model = pickle.load(open(model_p, "rb"))

    # Load test data
    print("load test data, labels")
    partition_p = os.path.join("..", "data", "partition", "partition.p")
    partition = pickle.load(open(partition_p, "rb"))

    labels_p = os.path.join("..", "data", "partition", "labels.p")
    labels = pickle.load(open(labels_p, "rb"))

    # Select a pseudo-random sample of test data.
    rand_ids = partition["test"][9:800][::10]
    print(rand_ids)

    # Predict test labels.
    for id in rand_ids:

        try:

            # Load image.
            p = os.path.join("..", "data", "imgs", id)
            img = Image.open(p)
            img = img.resize((255, 255), Image.ANTIALIAS)
            X = np.array(img)

            X = X/255
            X = X.swapaxes(0, 2) # (3, 255, 255)

            X = np.array([X for _ in range(64)])
            print(X.shape)
            X = torch.from_numpy(X).float()

            # Label.
            outputs = model(X)
            _, preds = torch.max(outputs, 1)

            # Plot.
            ypred = preds.numpy()[0]
            ytrue = id.split("/")[0]
            plt.imshow(img)
            plt.title("predict {} true {}".format(ypred, ytrue))
            plt.show()

        except Exception as e:
            print(e)

if __name__ == "__main__":
    visualize_predictions()
