import streamlit
from PIL import Image
import numpy as np
import os
import pickle
import torch 
from scipy.special import softmax
from scipy.ndimage import gaussian_filter
from flashtorch.saliency import Backprop
import seaborn as sns
import matplotlib.pyplot as plt

#####################
## Helper functions #
####################
def preload_ml_model():
    """ helper function to preload ML model """
    model_p = os.path.join("..", "models", "cloud_model.p")
    model = pickle.load(open(model_p, "rb"))
    return model

def load_and_preprocess_img(img_p):
    """
    img_p (str) path to image
    
    returns
    -------
    X (torch tensor)
    """

    # Load image and reshape to (3, 255, 255)
    img = Image.open(img_p)
    img = img.resize((255, 255), Image.ANTIALIAS)

    # Cast to torch tensor. or shape (64, 3, 255, 255)
    X = np.array(img)
    assert X.shape == (255, 255, 3)
    X = X/255
    X = np.array([X for _ in range(64)])
    X = X.swapaxes(2, 3) # (64, 255, 3, 255)
    X = X.swapaxes(1, 2) # (64, 3, 255, 255)
    print(X.shape)
    X = torch.from_numpy(X).float() # (64, 3, 255, 255) torch.

    return X


def get_saliency_map(model, img_p):
    """
    Return saliency map over the image : shape (255, 255)

    Parameters
    ----------
    model (PyTorch model object)
    img_p (str) path to image

    Returns
    -----------
    img_np : img as (255, 255, 3) numpy array
    saliency_map_np : saliency map as (255, 255) numpy array
    """

    # Load and preprocess image as torch array.
    img = Image.open(img_p).resize((255, 255), Image.ANTIALIAS)
    X = np.array(img).reshape((1, 255, 255, 3)) / 255
    X = X.swapaxes(2, 3).swapaxes(1, 2)
    X = torch.from_numpy(X).float() # (1, 3, 255, 255) torch array.
    X.requires_grad_() # This is critical to actually get gradients.

    # Get gradient using flashtorch.
    with torch.set_grad_enabled(True):
        backprop = Backprop(model)
        gradients = backprop.calculate_gradients(X,
                take_max = True, guided = True) # (1, 255, 255)

    # Cast image and saliency maps to numpy arrays.
    X = X.detach()
    img_np = X.numpy()[0].swapaxes(0, 1).swapaxes(1, 2) # (255, 255, 3)
    saliency_map_np = gradients.numpy()[0] # (255, 255)
    
    # Smooth heatmap.
    saliency_map_np = gaussian_filter(saliency_map_np, sigma=10)

    return img_np, saliency_map_np


def predict(model, img_p):
    """ predict image label """

    # Load image as torch tensor of shape (64, 3, 255, 255).
    X = load_and_preprocess_img(img_p) # (64, 3, 255, 255)
        
    # Predict integer label. (ypred_int)
    # Also save confidence.
    with torch.set_grad_enabled(False):
        pred_logits = model(X).numpy()[0] # (1, 5) numpy
        pred_probs = softmax(pred_logits)
        ypred_int = np.argmax(pred_probs) # int. 
        confidence = 100 * np.amax(pred_probs)

    # Get string label.
    int_to_lbl = {
        0 : "Mint",
        1 : "Near Mint",
        2 : "Excellent",
        3 : "Very Good",
        4 : "Poor"
    }
    ypred_str = int_to_lbl[ypred_int]

    return ypred_str, confidence

#####################
###Machine Learning##
#####################

# Preload ML model.
model = preload_ml_model()


######################
###Main App Display###
######################

# Title the app.
streamlit.title("MintCondition")

# Allow the user to upload a file.
FILE_TYPES = [".png", ".jpg", ".jpeg"]
uploader_title = "Upload a picture of a trading card!"
file = streamlit.file_uploader(uploader_title)

# Add a checkbox to control the saliency map.
show_saliency_map = streamlit.checkbox(
    label = "See what the model sees!",
    value = False, # default.
)

# Display the raw image, or the saliency map plus image, 
# Depending on the checkbox value.
if file != None:

    if show_saliency_map:
    
        plt.close("all")
        img_np, saliency_map_np = get_saliency_map(model=model, img_p=file)

        img_np = np.ones((255, 255, 3))

        heatmap = sns.heatmap(saliency_map_np, alpha=0.5, linewidths=0)
        heatmap.imshow(img_np, cmap="YlGnBu")
        plt.axis("off") # remove axes , ticks, etc.
        streamlit.pyplot()

    else:

        plt.close("all")
        img_np, saliency_map_np = get_saliency_map(model=model, img_p=file)
        plt.imshow(img_np)
        plt.axis("off") # remove axes, ticks, etc.
        streamlit.pyplot()

# Score the image.
if file != None:

    # Predict label.
    ypred, confidence = predict(model=model, img_p=file)

    score_markdown = """
    # Grade:
    ## {}
    """.format(ypred)
    streamlit.markdown(score_markdown)
