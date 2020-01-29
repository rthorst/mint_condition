"""
Test convolutional neural network based on trading card images
"""

import seaborn as sns
import copy
import boto3
from scipy.stats import spearmanr
import random
import csv
from sklearn.model_selection import train_test_split
import copy
import time
import numpy as np
import os
import pickle 
import torch
from torchvision import transforms
import torchvision.models as models
from torch.utils import data 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from flashtorch.saliency import Backprop
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import confusion_matrix, f1_score
from scipy.stats import pearsonr

def load_and_preprocess_img(img_p):
    """
    img_p (str) path to image
    
    returns
    -------
    X (torch tensor)
    """

    # Load image and resize to (255, 255)
    img = Image.open(img_p)
    img = img.resize((255, 255), Image.ANTIALIAS)

    # Cast to torch tensor of shape (3, 255, 255)
    X = np.array(img)
    assert X.shape == (255, 255, 3)
    X = X.transpose(2, 0, 1) # (3, 255, 255)
    X = torch.from_numpy(X).float() # (3, 255, 255) torch.

    # Normalize.
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    X = normalize(X)


    # Stack into minibatch of 1 images.
    X = torch.stack([X]*1) # (64, 3, 255, 255)

    return X


def test_model():

    # Load model.
    model_p = os.path.join("..", "models", "ebay_model.p")
    print("load {}".format(model_p))
    model = torch.load(model_p, map_location = torch.device("cpu"))

    # Load partition and labels.
    partition_p = os.path.join("..", "data", "partition", "partition.p")
    labels_p = os.path.join("..", "data", "partition", "labels.p")
    print("load {} and {}".format(partition_p, labels_p))
    partition = pickle.load(open(partition_p, "rb"))
    labels = pickle.load(open(labels_p, "rb"))

    # Predict test images and save prediction.
    ytrue_ = [] # shape (ntest, )
    ypred_ = [] # shape (ntest, )
    n_test = len(partition["test"])
    for idx, fname in enumerate(partition["test"]):

        try:


          
            # counter (short..more frequent)
            if idx % 100 == 0:

                # print progress.
                print("{}/{} : {}".format(idx, n_test, fname))

            # counter (longer...less frequent)
            if idx % 500 == 0:
        

                # f score.
                f = f1_score(ytrue_, ypred_, average="micro")
                print("F1 = {:.2f}".format(f))

                # r score.
                r, p = pearsonr(ytrue_, ypred_)
                print("r = {:.2f}".format(r))

                # confusion matrix.
                cm = confusion_matrix(ytrue_, ypred_)
                print("Confusion matrix:\n", cm)

            # load image.
            img_p = os.path.join("..", "data", "cropped_imgs", fname)
            X = load_and_preprocess_img(img_p)

            # predict label.
            pred_logits = model(X)
            ypred = torch.argmax(pred_logits).numpy() # int: {0...4}

            # store ypred and ytrue.
            ytrue = labels[fname]
            ypred_.append(ypred)
            ytrue_.append(ytrue)

        except Exception as e:
            print(e, fname)


    # f score.
    f = f1_score(ytrue_, ypred_, average="micro")
    print("F1 = {:.2f}".format(f))

    # pearson r.
    r, p = pearsonr(ytrue_, ypred_)
    print("r = {:.2f}".format(r))

    # confusion matrix.
    cm = confusion_matrix(ytrue_, ypred_)
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":

    test_model()
