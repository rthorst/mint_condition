import re
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
import requests 
import bs4
from ebaysdk.finding import Connection
from io import BytesIO

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
    #model_p = os.path.join("..", "models", "resnet_62_acc_torchsave.p")
    model_p = os.path.join("..", "models", "10class_42_acc_torchsave.p")
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
    #int_to_lbl = {
    #    4 : "Mint (PSA 9)",
    #    3 : "Near Mint (PSA 7)",
    #    2 : "Excellent (PSA 5)",
    #    1 : "Very Good (PSA 3)",
    #    0 : "Poor (PSA 1)"
    #}
    int_to_lbl = {
        9 : "Gem Mint (PSA 10)",
        8 : "Mint (PSA 9)",
        7 : "Near Mint-Mint (PSA 8)",
        6 : "Near Mint (PSA 7)",
        5 : "Excellent-Mint (PSA 6)",
        4 : "Excellent (PSA 5)",
        3 : "Very Good-Excellent (PSA 4)",
        2 : "Very Good (PSA 3)",
        1 : "Very Good-Excellent (PSA 2)", 
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

hide_upload_text = False # if true, hide upload boxes and relevant text.
file = None

# Title the "advanced options" section of the sidebar.
streamlit.sidebar.markdown("### Advanced Options")

# Add a checkbox to add a watermark.
add_watermark = streamlit.sidebar.checkbox(
    label = "Add watermark to verify grade",
    value = False # default
)
if not hide_upload_text:

    # Allow the user to upload a file.
    FILE_TYPES = [".png", ".jpg", ".jpeg"]  
    uploader_title = """
    ## Use AI to grade the condition of a trading card!
    """
    streamlit.markdown(uploader_title)
    file = streamlit.file_uploader(label="Option 1: Upload a Picture of the Card")

    ## Get an image from ebay.
    ebay_md = """
    Option 2: enter an ebay auction URL e.g. http://www.bit.ly/topps-psa9
    """
    #streamlit.markdown(ebay_md)
    ebay_url = streamlit.text_input(ebay_md)

    if ebay_url not in [None, ""]:

        try: 

            # output loading message to user.
            streamlit.text("Loading image from ebay...please wait")

            # download it.
            print("download {}".format(ebay_url))
            html = requests.get(ebay_url).text

            # get image url.
            url_pat = re.compile(r"bigImage.src ='\S+s-1300.jpg")
            url_pat = re.compile(r"bigImage.src\s{0,2}=\s{0,2}'\S+;")
            mat = re.findall(pattern=url_pat, string=html)[0]
            print(mat)
            img_url = mat.split("'")[1]

            # download image.
            print("download {}".format(img_url))
            img_bytes = requests.get(img_url).content
            file = BytesIO(img_bytes)

        except Exception as e:
            print(e)
            streamlit.markdown("Sorry! We can't retrieve an image from that URL")


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

# Add a checkbox to control the saliency map.
show_saliency_map = streamlit.sidebar.checkbox(
    label = "See what the model sees! (beta)",
    value = False, # default.
)

# Describe how MintCondition can be used to price a card.
price_md = """
### How To Price Your Card

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
### How MintCondition Works
MintCondition learned to grade cards from pictures of over 20,000 expert (PSA)-graded
trading cards. It uses AI to achieve roughly twice the accuracy
of the average human amateur.
"""
streamlit.sidebar.markdown(how_it_works_md)

## What if MintCondition is wrong.
is_it_wrong_md = """
### Can MintCondition Be Wrong?
Yes. For the best results, try uploading a high-quality picture of the card in vertical orientation, similar
to the example cards. With respect to accuracy, MintCondition is twice as accurate as the 
average human amateur but can still make mistakes, even on high-quality photos. MintCondition
also cannot tell if a card has been artificially altered to appear in better condition.
"""
streamlit.sidebar.markdown(is_it_wrong_md)

## Contact information.
streamlit.sidebar.markdown("""
### Contact 
The developer [Robert Thorstad](http://www.robertthorstad.com) can be reached at rthorst (at) gmail (dot) com.""")

# If specified by user, select a random card to use.
if use_random_card:

    cards_p = "cropped_demo_cards"
    fnames = os.listdir(cards_p)
    fname = np.random.choice(fnames)
    file = os.path.join(cards_p, fname)


# Display the raw image, or the saliency map plus image, 
# Depending on the checkbox value.
if file != None:

    hide_upload_text = True

    try:

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
    
            heatmap = sns.heatmap(saliency_map_np, alpha=0.8, linewidths=0,
                  cbar=False)
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

    except Exception as e:
        print(e) # for logging.
        error_md = """
        There was an error processing the picture. Please try again with 
        a different picture. This can happen if the image uses a non-standard
        file format (.JPG is preferred) or sometimes if the image is grayscale
        (try a full-color image).
        """
        streamlit.markdown(e)



