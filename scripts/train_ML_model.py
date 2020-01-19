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
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle 
import torch
from torch.utils import data 
from torch import transforms 

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

    # List all input filenames.
    print("list input filenames")
    preprocessed_p = os.path.join("..", "data", "preprocessed_imgs")
    fnames = os.listdir(preprocessed_p)

    # Train/test split.
    print("split train, test")
    train_fnames, test_fnames = train_test_split(fnames, test_size=test_size)
    # Create partition dictionary.
    print("create partition object")
    partition = {
        "train" : train_fnames,
        "test"  : test_fnames
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
    labels_p = os.path.join("..", "data", "imgs")
    str_labels = os.listdir(labels_p)
    lbl_to_idx = {lbl : i for i, lbl in enumerate(str_labels)}

    # Create labels dictionary.
    # fname_to_lbl is a dictionary mapping e.g. "some_fname.npy" -> 2
    # Example filename.100097_sport185226_conditionVG_preprocessed.npy
    fname_to_lbl = {}
    for fname in fnames:

        # Extract label.
        str_lbl = fname.split("_")[-2].lstrip("condition")
        int_lbl = lbl_to_idx[str_lbl]
        
        # Add to dictionary.
        fname_to_lbl[fname] = int_lbl

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

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        # Select sample.
        ID = self.list_IDs[index]

        # Load X and reshape to (n_channels, height, width)
        # Load data and get label.
        base_p = os.path.join("..", "data", "preprocessed_imgs")
        file_p = os.path.join(base_p, ID)
        X_np = np.load(file_p)
        X_np = np.reshape(X_np, (3, 255, 255))
        print(X_np.shape)
        X = torch.from_numpy(X_np)
        y = self.labels[ID]

        return X, y


def train_CNN_model():
    """
    Train a CNN to classify card pictures.
    """ 

    # Use CUDA if available.
    use_cuda = torch.cuda.is_available()
    print("check if CUDA can be used -> {}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Define a normalization transform, which is necessary for 
    # The pretrained AlexNet. The specific parameters are necessary
    # for AlexNet.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # Parameters
    params = {
        "batch_size" : 64,
        "shuffle" : True,
        "num_workers" : 4,
        "transform" : normalize
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


    # Train model.
    print("train model")
    batch_num = 0
    for epoch in range(max_epochs):

        print("---epoch {}---".format(epoch))

        # Train.
        for local_batch, local_labels in train_g:

            # Transfer to GPU if available.
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(device)

            """
            Model computations. TODO fill me out.
            """


            # Normalize data, a requirement for pretrained alexnet

            # Counter.
            batch_num += 1
            if batch_num % 5 == 0:
                print("...batch number {}".format(batch_num))

        # Test.
        with torch.set_grad_enabled(False):

            for local_batch, local_labels in test_g:


                # Transfer to GPU if available.
                local_batch = local_batch.to(device)
                local_labels = local_labels.to(device)
        
                """
                Model computations. TODO fill me out.
                """
if __name__ == "__main__":

    split_train_test()
    train_CNN_model()
