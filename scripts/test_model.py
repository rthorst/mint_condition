"""
Test ML model against gold-standard data, 
currently, a hand-curated set of PSA-graded cards.
"""
import os
from PIL import Image
from sklearn.metrics import classification_report
import numpy as np
import torch
import pickle

def load_and_preprocess_card(img_p):
    """
    Load card from path and return preprocessed numpy array
    """

    img = Image.open(img_p)
    img = img.resize((255, 255), Image.ANTIALIAS)
    X = np.array(img)
    print(X.shape)

    X = X/255
    X = X.swapaxes(0, 2)
    return X

def fname_to_label(fname):
    """ input e.g. ex_8.jpg
        output target label as int (here: 2)
    """

    condition_to_lbl = {
        "mint" : 0,
        "nm" : 1,
        "ex" : 2,
        "vg" : 3,
        "poor" : 4
    }

    str_lbl = fname.split("_")[0] # e.g., "EX"
    int_lbl = condition_to_lbl[str_lbl]

    return int_lbl

def test_against_psa_graded_cards():
    """ Test model against hand-curated PSA cards
    """


    ypred = []
    ytrue = []

    # List test cards to load.
    base_p = os.path.join("..", "data", "curated_test_dataset")
    fnames = os.listdir(base_p)

    # Load cards.
    print("load cards")
    card_arrs = [] # list of numpy array.s
    targets = [] # true labels 

    for fname in fnames:

        p = os.path.join(base_p, fname)

        lbl = fname_to_label(fname)
        X = load_and_preprocess_card(p)

        card_arrs.append(X)
        targets.append(lbl)

    # Load model.
    print("load model")
    model_p = os.path.join("..", "models", "cloud_model.p")
    model = pickle.load(open(model_p, "rb"))

    # Predict label.
    print("label cards")
    predictions = []
    ytrue = []
    for xi, lbl in zip(card_arrs, targets):

        try:


            print(lbl)

            # cast to batch of this image. X 64 times.
            X = np.array([xi for _ in range(64)])
            X = torch.from_numpy(X).float()

            # predict.
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            ypred = preds.numpy()[0]

            # add to predictions and labels.
            predictions.append(ypred)
            ytrue.append(lbl)

        except Exception as e:
            print(e)

    # Score model.
    report = classification_report(predictions, ytrue)
    print(predictions)
    print(ytrue)
    print(report)


def test_against_psa_mickey_mantle():
    """
    Test against example grades of PSA Mickey Mantle card.
    """


    ypred = []
    ytrue = []

    # List test cards to load.
    base_p = os.path.join("..", "data", "mickey_mantle_test")
    fnames = os.listdir(base_p)

    # Load cards.
    print("load cards")
    card_arrs = [] # list of numpy array.s
    targets = [] # true labels 

    for fname in fnames:

        p = os.path.join(base_p, fname)

        # get label.
        lbl_to_int = {
            "mint" : 0,
            "nm" : 1,
            "ex" : 2,
            "vg" : 3,
            "poor" : 1
        }
        lbl = lbl_to_int[fname[:-4]]
        X = load_and_preprocess_card(p)

        card_arrs.append(X)
        targets.append(lbl)

    # Load model.
    print("load model")
    model_p = os.path.join("..", "models", "cloud_model.p")
    model = pickle.load(open(model_p, "rb"))

    # Predict label.
    print("label cards")
    predictions = []
    ytrue = []
    for xi, lbl in zip(card_arrs, targets):

        try:


            print(lbl)

            # cast to batch of this image. X 64 times.
            X = np.array([xi for _ in range(64)])
            X = torch.from_numpy(X).float()

            # predict.
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            ypred = preds.numpy()[0]

            # add to predictions and labels.
            predictions.append(ypred)
            ytrue.append(lbl)

        except Exception as e:
            print(e)

    # Score model.
    report = classification_report(predictions, ytrue)
    print("ypred")
    print(predictions)
    print("ytrue")
    print(ytrue)
    print(report)


if __name__ == "__main__":

    #test_against_psa_graded_cards()
    test_against_psa_mickey_mantle()
