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
# This will create the database file if it does not exist.
log_p = "cron.db"
conn = sqlite3.connect(log_p)
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
TODO: stopped here, API call is not working.
"""

# Load ebay API credentials.
credentials_p = "ebay_sdk_credentials.txt"
f = open(credentials_p, "r")
client_id, dev_id, client_secret = [l.rstrip("\n").lstrip() for l in f.readlines()]

# TODO limit to ungraded cards.
params = {
    "categoryName" : "Baseball Cards",
    "paginationInput" : 
        {
            "keywords" : "psa", # TODO remove me.
            "entriesPerPage" : 100,
            "pageNumber" : 1
        },
    "outputSelector" : "PictureURLLarge"
}
response = api.execute("findItemsAdvanced", params)
j = response.json()

# grade each card.
items = j["searchResult"]["item"] # json[] length 100
for idx, item in enumerate(items):

    try:

        # Get picture url.
        url = item["pictureURLLarge"]

        # Download image.
        res = requests.get(url).content

        # TODO extract price from auction and modify below to 
        # use a real price.

        # TODO extract card title.

        # Grade card.
        ypred_int, ypred_str, confidence = model.predict(res)

        # Log grade.
        time_utc = datetime.datetime.now().timestamp()
        price = 0
        card_title = ""
        insert_row_sql = """
        insert into cron (url, grade, title, price, access_date_utc, confidence)
        values '{}', {}, '{}', {}, {}, {}
        """.format(url, ypred_int, card_title, price, time_utc, confidence)
        
        cursor.execute(insert_row_sql)

        # Sleep 0.25 seconds to be nice.
        time.sleep(0.25)



    except Exception as e:
        print(e)



"""

# Preload ML model.
model = preload_ml_model()
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

ypred_int, ypred_str, confidence = predict(model, file)

"""
