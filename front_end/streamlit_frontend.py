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

def set_parameter_requires_grad(model, feature_extracting):
    """
    Helper function to freeze layers when fine-tuning a model.
    Sets all neurons except those in a top layer to not return gradient.

    From 
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tut    orial.html
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    """
    Helper function to initialize pretrained AlexNet (and other vision)
    models in Pytorch.
    From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


#def preload_ml_model():
#    # helper function to preload ML model
#    model_p = os.path.join("..", "models", "cloud_model.p")
#    model = pickle.load(open(model_p, "rb"))
#    return model

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

    # Load and preprocess image: (1, 3, 255, 255) torch tensor.
    X = load_and_preprocess_img(img_p)

    # Require gradient.
    X.requires_grad_() # This is critical to actually get gradients.

    # Get gradient using flashtorch.
    with torch.set_grad_enabled(True):
        backprop = Backprop(model)
        gradients = backprop.calculate_gradients(input_=X, 
                take_max = True, 
                guided = True) # (1, 255, 255)

    # Cast image and saliency maps to numpy arrays.
    X = X.detach()
    img_np = X.numpy()[0] # (3, 255, 255)
    img_np = img_np.transpose(1, 2, 0) # (255, 255, 3)
    saliency_map_np = gradients.numpy()
    saliency_map_np = np.absolute(saliency_map_np) # absolute value
    
    # Smooth heatmap.
    #saliency_map_np = gaussian_filter(saliency_map_np, sigma=10)

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
show_saliency_map = streamlit.checkbox(
    label = "See what the model sees!",
    value = False, # default.
)

# Add a checkbox to add a watermark.
add_watermark = streamlit.checkbox(
    label = "Add watermark to verify grade",
    value = False # default
)

# Display the raw image, or the saliency map plus image, 
# Depending on the checkbox value.
if file != None:

    plt.close("all")
    img_np, saliency_map_np = get_saliency_map(model=model, img_p=file)

    img_PIL = Image.open(file).resize((255, 255), Image.ANTIALIAS)
    img_np = np.array(img_PIL)

    print(saliency_map_np)

    # saliency map.
    if show_saliency_map:
    
        heatmap = sns.heatmap(saliency_map_np, alpha=0.5, linewidths=0)
        heatmap.imshow(img_np, cmap="YlGnBu")

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

    # Predict label.
    ypred, confidence = predict(model=model, img_p=file)   
    plt.title("Grade: {}".format(ypred))

    # show image.
    streamlit.pyplot()


