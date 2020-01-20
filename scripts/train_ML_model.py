"""
https://pytorch.org/docs/stable/torchvision/models.html

Images must be:
   Shape (3, H, W) where H and W >= 224
   Normalized to range (0, 1) and then normalized with 
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]

    You can use for normalizing in Pytorch:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

Pretrained AlexNet Model:
torchvision.models.alexnet(pretrained=True, progress=True, **kwargs)

Stream data:
    torch.utils.data.DataLoader ; iterable over dataset.

    dataset : IterableDataset object.

Cheat Sheet for streaming data: (really good)
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

"""

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

def split_train_test(test_size = 0.1):
    """
    Partition data into train/test set.

    Parameters:
    ----------
    Test_size (float) : proportion test size (0-1)

    Returns:
    --------
    None

    Creates a data/partition/ directory with two objects.
        - partition.p :: a dictionary with "train" and "test" 
          keys which map to lists of filenames.
        - labels.p :: a mapping from filename to label.
    """

    """
    Create partition dictionary
    """

    # Make partition directory.
    print("make partition directory if not exists")
    partition_p = os.path.join("..", "data", "partition")
    if not os.path.exists(partition_p):
        os.mkdir(partition_p)

    # List all input filenames and shuffle.
    base_p = os.path.join("..", "data", "imgs")
    img_ps = [] # e.g. ../data/imgs/MINT/foo.jpg.

    for dir_name in os.listdir(base_p):

        dir_p = os.path.join(base_p, dir_name)
        for img_fname in os.listdir(dir_p):

            img_p = os.path.join(dir_name, img_fname)
            img_ps.append(img_p)
    random.shuffle(img_ps)

    # Keep only baseball cards in MINT or POOR condition, for now.
    print("Keep only baseball cards, for now")
    baseball_sport = "185223"
    img_ps = [p for p in img_ps if baseball_sport in p]

    # Keep only desired conditions.
    print("Exclude good and fair cards to create a 9, 7, 5, 3, 1 scale")
    img_ps = [p for p in img_ps if "GOOD" not in p and "FAIR" not in p]    

    # Train/test split.
    print("split train, test")
    train_ps, test_ps = train_test_split(img_ps, test_size=test_size)

    # Create partition dictionary.
    print("create partition object")
    partition = {
        "train" : train_ps,
        "test"  : test_ps
    }

    # Pickle partition object.
    of_p = os.path.join(partition_p, "partition.p")
    with open(of_p, "wb") as of:
        pickle.dump(partition, of)
    print("wrote {}".format(of_p))

    """
    Create labels dictionary.
    """ 

    # Map all unique labels to integer IDs.
    # lbl_to_idx is a dictinary mapping e.g. "EX" -> 0, etc.
    #str_labels = ["MINT", "NM", "EX", "VG", "GOOD", "FAIR", "POOR"]
    str_labels = ["MINT", "NM", "EX", "VG", "POOR"]
    lbl_to_idx = {lbl : i for i, lbl in enumerate(str_labels)}

    # Create labels dictionary.
    # fname_to_lbl is a dictionary mapping e.g. "some_fname.npy" -> 2
    # Example filename.100097_sport185226_conditionVG_preprocessed.npy
    fname_to_lbl = {}
    for p in img_ps:

        # Extract label.
        str_lbl = p.split("_")[-1].lstrip("condition").rstrip(".jpg")
        int_lbl = lbl_to_idx[str_lbl]
        
        # Add to dictionary.
        fname_to_lbl[p] = int_lbl

    # Pickle fname_to_lbl mapping.
    of_p = os.path.join(partition_p, "labels.p")
    with open(of_p, "wb") as of:
        pickle.dump(fname_to_lbl, of)
    print("wrote {}".format(of_p))


class Dataset(data.Dataset):
    """ Characterize an iterable dataset 
        Modified from https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, list_IDs, labels):
        self.labels = labels # dictionary. key=fname, value=integer lbl.
        self.list_IDs = list_IDs # list of filenames.

        # Initialize AWS storage (s3) resource.
        self.s3_resource = boto3.resource("s3")

    def __len__(self):
        return len(self.list_IDs)

    def download_from_s3(self, fname):
        # download fname from s3 as "img.jpg"

        # clear old "img.jpg" if exists.
        if os.path.exists("img.jpg"):
            os.remove("img.jpg")
    
        # download image from S3 as "img.jpg"
        resp = self.s3_resource.Object(
            "mintcondition",
            fname
        ).download_file("img.jpg")

        return None

    def __getitem__(self, index):

        try:

            # Select sample.
            img_p = self.list_IDs[index]

            # Get label.
            y = self.labels[img_p]

            # Download image from S3 instance and save as "tmp.jpg"
            self.download_from_s3(img_p)

            # Load image and reshape to (3, 255, 255)
            img = Image.open("img.jpg")
            import time
            time.sleep(2)
            img = img.resize((255, 255), Image.ANTIALIAS)

            # Cast to torch tensor.
            X = np.array(img) # (255, 255, 3) numpy
            assert X.shape == (255, 255, 3)
            X = X/255 # "normalize"
            X = X.swapaxes(0, 2) # (3, 255, 255) numpy
            X = torch.from_numpy(X).float() # (3, 255, 255) torch

        except Exception as e:
            print(e)
            print("exception loading data..using random image, label instead")
            X = np.random.random((3, 255, 255))
            X = torch.from_numpy(X).float()
            y = 0

        return X, y


def set_parameter_requires_grad(model, feature_extracting):
    """
    Helper function. If we are feature extracting, e.g. fine-tuning
    only the top layer of a model, then set existing parameters in the
    model to not return gradients.
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
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
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


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, device="cpu"):
    """ Helper function to train model in PyTorch
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tut    orial.html
    """

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initiailize a file to track accuracy over epochs.
    acc_of_p = os.path.join("..", "data", "model_accuracy.csv")
    acc_of = open(acc_of_p, "w", newline="")
    header = ["epoch", "phase", "accuracy"]
    w = csv.writer(acc_of)
    w.writerow(header)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            batch_num = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # counter.
                batch_num += 1

                if batch_num % 1 == 0:
                    correct = float(torch.sum(preds == labels.data))
                    incorrect = float(torch.sum(preds != labels.data))
                    perc_correct = 100 * correct / (correct + incorrect)
                    msg = """
                    epoch {} batch {} : percent correct {}
                    """.format(epoch, batch_num, perc_correct)
                    print(msg)
                    print(preds)
                    print(labels.data)

                    # rank correlation of predicted, actual.
                    rho, p = spearmanr(preds.numpy(), labels.data.numpy())
                    print("correlation of pred, actual: rho = {:.4f}".format(rho))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            # Write latest train and test accuracies to output file.
            out = [epoch, phase, epoch_acc.numpy()]
            w.writerow(out)
            acc_of.flush()

        # Pickle the model after end of epoch.
        of_p = os.path.join("..", "models", "latest_model.p")
        with open(of_p, "wb") as of:
            pickle.dump(model, of)
        print("wrote {}".format(of_p))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # close output file.
    of.flush()
    of.close()
    print("wrote {}".format(acc_of_p))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def train_CNN_model(num_classes=7, load_latest_model=False):
    """
    Train a CNN to classify card pictures.

    Parameters:
    --------
    num_classes (int) : controls # of output neurons.
    load_latest_model (boolean) : if True load models/latest_model.p
    """ 

    # Use CUDA if available.
    use_cuda = torch.cuda.is_available()
    print("check if CUDA can be used -> {}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Define transformations of the images.
    # Note that the normalization transform with these specific
    # Parameters is necessary for working with pretrained AlexNet.    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        normalize
    ])

    # Parameters
    params = {
        "batch_size" : 64,
        "shuffle" : True,
        "num_workers" : 4
    }
    max_epochs = 100

    # Datasets.
    print("load train/test partition and labels")
    partition_p = os.path.join("..", "data", "partition", "partition.p")
    labels_p = os.path.join("..", "data", "partition", "labels.p")
    
    partition = pickle.load(open(partition_p, "rb"))
    labels = pickle.load(open(labels_p, "rb"))

    # Create generators over train data and test data.
    print("create generators over train, test data")
    train_set = Dataset(partition["train"], labels)
    train_g = data.DataLoader(train_set, **params)

    test_set = Dataset(partition["test"], labels)
    test_g = data.DataLoader(test_set, **params)

    # Initialize pretrained AlexNet model 
    # Replace top layer with a layer of (num_classes) outputs
    # Note that setting feature_extract to False means we fine-tune
    # the whole model, True we finetune only top layer.
    if load_latest_model:
        print("load pickled latest_model.p")
        model_p = os.path.join("..", "models", "latest_model.p")
        model = pickle.load(open(model_p, "rb"))
    else:
        model, _ = initialize_model(
            model_name = "alexnet",
            num_classes = num_classes,
            feature_extract = False, # if True only finetune top layer.
            use_pretrained = True
        )
    model.to(device)

    # Initialize optimizer, loss criterion.
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Train model.
    model, history = train_model(
        model=model,
        dataloaders = {"train" : train_g, "test" : test_g},
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=max_epochs,
        is_inception=False,
        device=device
    )

    # Pickle best performing model.
    of_p = os.path.join("..", "models", "best_model.p")
    with open(of_p, "wb") as of:
        pickle.dump(model, of)
    print("wrote {}".format(of_p))

    # Pickle history of best performing model.
    of_p = os.path.join("..", "models", "history.p")
    with open(of_p, "wb") as of:
        pickle.dump(history, of)
    print("wrote {}".format(of_p))


if __name__ == "__main__":

    split_train_test()
    train_CNN_model(load_latest_model=True, num_classes=5)
