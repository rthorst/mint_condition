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
import cv2 

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
        4 : "Mint (PSA 9)",
        3 : "Near Mint (PSA 7)",
        2 : "Excellent (PSA 5)",
        1 : "Very Good (PSA 3)",
        0 : "Poor (PSA 1)"
    }

    # Get confidence as string.
    #if confidence > 50:
    #    confidence = "High"
    #elif confidence < 30:
    #    confidence = "Low"
    #else:
    #    confidence = "Medium"

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
uploader_title = """
### Use AI to grade the condition of a trading card! Simply upload a picture of the card.

For advanced options or to use an example card, use the menu on the left.
"""
streamlit.markdown(uploader_title)
file = streamlit.file_uploader(label="")

# Add a checkbox to control the saliency map.
#show_saliency_map = streamlit.checkbox(
#    label = "See what the model sees!",
#    value = False, # default.
#)
show_saliency_map = False

# Title the "advanced options" section of the sidebar.
streamlit.sidebar.markdown("## Advanced Options:")

# Add a checkbox to add a watermark.
add_watermark = streamlit.sidebar.checkbox(
    label = "Add watermark to verify grade",
    value = False # default
)

# Checkbox to show confidence.
display_confidence = streamlit.sidebar.checkbox(
    label = "Show how confident the model is",
    value = False # default
)

# Checkbox to use random sample card.
use_random_card = streamlit.sidebar.checkbox(
    label = "Use a random example card",
    value = False # default
)

# Describe how MintCondition can be used to price a card.
price_md = """
## How To Price Your Card

After your card is graded, there are 2 good ways to find a price.
For a quick 'good-enough' estimate, search
[historical auction data](https://www.ebay.com/b/Sports-Trading-Cards-Accessories/212/bn_1859819)
for similar cards. You can also use 
[paid reference books](https://www.amazon.com/Beckett-Baseball-Card-Price-Guide/dp/1936681331/ref=sr_1_2?keywords=trading+card+guide&qid=1580325809&sr=8-2)
or [subscription services](https://www.beckett.com/online-price-guide) to find a price. 
"""
streamlit.sidebar.markdown(price_md)

## Describe how MintCondition works.
how_it_works_md = """
## How MintCondition Works

MintCondition is trained using photos of over 20,000 expert (PSA)-graded
trading cards. It uses AI to achieve roughly twice the accuracy
of the average human amateur. MintCondition is not perfect, but it usually
produces a reasonable 'good-enough' estimate of the condition of your card.
"""
streamlit.sidebar.markdown(how_it_works_md)

# If specified by user, select a random card to use.
if use_random_card:

    cards_p = "cropped_demo_cards"
    fnames = os.listdir(cards_p)
    fname = np.random.choice(fnames)
    file = os.path.join(cards_p, fname)

# Display the raw image, or the saliency map plus image, 
# Depending on the checkbox value.
if file != None:

    # Predict label and get confidence.
    ypred, confidence = predict(model, file)

    # Show prediction.
    pred_md = "# Grade: {}".format(ypred)
    streamlit.markdown(pred_md)

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

        watermark_s = "{}\nVerified by MintCondition".format(ypred)
        watermark_s = watermark_s.rjust(50, " ")
        ax = plt.gca()
        ax.annotate(
            xy=(0, 0.5), s=watermark_s,
            alpha = 0.5, color="white", size=20,
            xycoords = "axes fraction"
        )

    # Show confidence.
    if display_confidence:
        md = "# Confidence: {:.1f}%".format(confidence)
        md += "\n\n(low confidence? try uploading a different image)"
        streamlit.markdown(md)

    # show image.
    streamlit.pyplot()


