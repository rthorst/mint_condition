import re
from PIL import Image
import numpy as np
import os
import torch 
from scipy.special import softmax
from scipy.ndimage import gaussian_filter
from flashtorch.saliency import Backprop
from torchvision import models, transforms
import requests 
import bs4
from ebaysdk.finding import Connection
from io import BytesIO
import sqlite3
from ebaysdk.finding import Connection
import datetime

"""
Connect to the SQL database for logging nightly cron job.
If the table does not exist, create it.
"""

# Connect to SQL database, for logging.
# This will create the database file if it does not exist
db_p = "cron.db"
conn = sqlite3.connect(db_p)
cursor = conn.cursor()

# If the table does not already exist, create the cron table.
# Note that the table will not be over-written if it already exists.
create_table_command = open("create-cron-table.sql", "r").read()
cursor.execute(create_table_command)

"""
Helper functions
"""

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
    model_p = os.path.join("..", "models", "resnet_62_acc_5_class_torchsave.p")
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
        print("predicted probabilities", [round(100*v, 1) for v in pred_probs])
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
    #int_to_lbl = {
    #    9 : "Gem Mint (PSA 10)",
    #    8 : "Mint (PSA 9)",
    #    7 : "Near Mint-Mint (PSA 8)",
    #    6 : "Near Mint (PSA 7)",
    #    5 : "Excellent-Mint (PSA 6)",
    #    4 : "Excellent (PSA 5)",
    #    3 : "Very Good-Excellent (PSA 4)",
    #    2 : "Very Good (PSA 3)",
    #    1 : "Good (PSA 2)", 
    #    0 : "Poor (PSA 1)"
    #}

    # Get confidence as string.
    #if confidence > 50:
    #    confidence = "High"
    #elif confidence < 30:
    #    confidence = "Low"
    #else:
    #    confidence = "Medium"

    ypred_str = int_to_lbl[ypred_int]

    return ypred_int, ypred_str, confidence

"""
Get 100 most recent cards listed as ungraded.
"""

model = preload_ml_model()

# Get HTML of the ebay page listing all 100 most recent ungraded cards.
base_url = "https://www.ebay.com/b/Ungraded-Single-Baseball-Cards/213/bn_16986411?rt=nc&_sop=10"
html = requests.get(base_url).text

# Extract links to pages for each individual trading card.
soup = bs4.BeautifulSoup(html)
attrs = {"class" : "s-item__link"}
auction_links = soup.findAll("a", attrs)
auction_links = [link["href"] for link in auction_links]

# Parse each individual trading card.
for idx, auction_url in enumerate(auction_links):

    try:

        # Get HTML for this auction.
        auction_html = requests.get(auction_url).text
        soup = bs4.BeautifulSoup(auction_html)

        # extract card price.
        attrs = {"id" : "prcIsum"}
        price_span = soup.findAll("span", attrs)[0]
        price = float(price_span["content"])

        # extract card title
        # we extract it directly from the url .../itm/{title}/...i
        card_title = auction_url.split("/itm/")[1].split("/")[0]
        print(card_title)

        # save current in UTC
        access_date_utc = datetime.datetime.now().timestamp()

        # Extract URL of the picture of the card.
        url_pat = re.compile(r"bigImage.src\s{0,2}=\s{0,2}'\S+;")
        mat = re.findall(pattern=url_pat, string=auction_html)[0]
        img_url = mat.split("'")[1]

        # Download card picture.
        print("download {}".format(img_url))
        img_bytes = requests.get(img_url).content
        f = BytesIO(img_bytes)

        # Grade picture.
        ypred_int, ypred_str, confidence = predict(model, f)

        # Write SQL.
        sql_insert_statement = """
        insert into cron 
        (url, grade, title, price, confidence, access_date_utc)
        
        values
        ( '{}', {}, '{}', {}, {}, {} );
        """.format(
            auction_url,
            ypred_int,
            card_title,
            price,
            confidence,
            access_date_utc
        )
        cursor.execute(sql_insert_statement)

    except Exception as e:
        print(e)

    # sleep (to be nice) regardless of success or failure.
    import time
    time.sleep(.25)

# commit changes.
conn.commit()
