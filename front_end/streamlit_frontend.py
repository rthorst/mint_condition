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
from torchvision import models, transforms

#####################
## Helper functions #
####################

def preload_ml_model():
    """
    helper function to preload ML model.

    returns:
    ---------
    model (torch model object)

    Note that the model is serialized, as a "state dict", using
    Python 2.7. Thus special flags need to be used to un-serialize
    the model in Python 3.
    """

    # Load model.
    model_p = os.path.join("..", "models", "ebay_model.p")
    model = torch.load(model_p, 
                       map_location = torch.device("cpu")
            )

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


def get_saliency_map(model, img_p, ypred):
    """
    Return saliency map over the image : shape (255, 255)

    Parameters
    ----------
    model (PyTorch model object)
    img_p (str) path to image
    ypred (int) 0-4. Used to control saliency map

    Returns
    -----------
    img_np : img as (255, 255, 3) numpy array
    saliency_map_np : saliency map as (255, 255) numpy array

    TODO:
    Examine this code more closely. At the moment, saliency maps
    don't change much across classes. I think a different saliency
    mapping technique is needed. The Guided=True flag may be possiblw
    but at the moment causes NaN errors.

    """

    # Load and preprocess image: (1, 3, 255, 255) torch tensor.
    X = load_and_preprocess_img(img_p)

    # Require gradient.
    X.requires_grad_() # This is critical to actually get gradients.

    # Get gradient using flashtorch.
    with torch.set_grad_enabled(True):
        backprop = Backprop(model)
        gradients = backprop.calculate_gradients(input_= X, 
                target_class = ypred,
                take_max = True, 
                guided = False) # (1, 255, 255)

    # Cast image and saliency maps to numpy arrays.
    X = X.detach()
    img_np = X.numpy()[0] # (3, 255, 255)
    img_np = img_np.transpose(1, 2, 0) # (255, 255, 3)
    saliency_map_np = gradients[0].numpy()
    #saliency_map_np = np.absolute(saliency_map_np) # absolute value
    
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
        print("predicted probabilities", pred_probs)
        ypred_int = np.argmax(pred_probs) # int. 
        confidence = 100 * np.amax(pred_probs)

    # Get string label.
    int_to_lbl = {
        4 : "Mint",
        3 : "Near Mint",
        2 : "Excellent",
        1 : "Very Good",
        0 : "Poor"
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
#show_saliency_map = streamlit.checkbox(
#    label = "See what the model sees!",
#    value = False, # default.
#)
show_saliency_map = False

# Add a checkbox to add a watermark.
add_watermark = streamlit.checkbox(
    label = "Add watermark to verify grade",
    value = False # default
)

# Display the raw image, or the saliency map plus image, 
# Depending on the checkbox value.
if file != None:

    # Predict label and get confidence.
    ypred, confidence = predict(model, file)

    # Get image and saliency map.
    img_PIL = Image.open(file).resize((255, 255), Image.ANTIALIAS)
    img_np = np.array(img_PIL)
    _, saliency_map_np = get_saliency_map(model=model, 
                                     img_p=file, ypred=0)

    # Initiailize new plot, close old plots.
    plt.close("all")

    # saliency map.
    if show_saliency_map:
    
        heatmap = sns.heatmap(saliency_map_np, alpha=0.8, linewidths=0)
        heatmap.imshow(img_np, cmap="RdBu")

    # no saliency map.
    else:

        plt.imshow(img_np)

    # shared across all plots.
    plt.axis("off")

    # optional watermark.
    if add_watermark:

        ax = plt.gca()
        ax.annotate(
            xy=(0.5, 0.5), s="Verified By Mint Condition",
            alpha = 0.5, color="gray", size=20
        )

    # Title with label.
    plt.title("Grade: {}".format(ypred))

    # show image.
    streamlit.pyplot()


