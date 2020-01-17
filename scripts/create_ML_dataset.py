"""
Create dataset suitable for machine learning.
"""

import os
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt

def preprocess_img(PIL_object, output_shape=(255, 255, 3)):
    """
    Wrapper to preprocess a single image as PIL image object.
    Should generalize to new images eventually inputted online.

    PIL_object (pil) : PILLOW library image object.
    output_shape (width, height, n_channels) tuple.

    returns: 
    array representation.
    """ 

    # If width > height, rotate 90 degrees.
    width, height = PIL_object.size
    if width > height:
        PIL_object = PIL_object.rotate(90)

    # Scale to final shape, for example (255, 255, 3)
    (new_width, new_height, new_channels) = output_shape
    #PIL_object = ImageOps.fit(PIL_object, (new_width, new_height), method=Image.ANTIALIAS)
    PIL_object = PIL_object.resize((new_width, new_height), Image.ANTIALIAS)
    # Scale to int16 numpy array.
    out = np.array(PIL_object, dtype=np.int16)

    # Return new image as numpy array.
    return out

def create_dataset_wrapper():
    """
    Main wrapper to create dataset.
    """

    # List file paths to load.
    # Also extract condition of the card from the filename.

    print("list file paths to images in the dataset")
    print("also extract condition of each card from the filename")
    print("currently keep only baseball images")
    img_paths = []
    conditions = []
    base_p = os.path.join("..", "data", "imgs")
    for dir_name in os.listdir(base_p):
        print("\t...{}".format(dir_name))
        dir_p = os.path.join(base_p, dir_name)

        for img_fname in os.listdir(dir_p):

            # skip non-baseball images.
            if "sport185223" not in img_fname:
                continue

            # get path.
            img_p = os.path.join(dir_p, img_fname)
            img_paths.append(img_p)
            
            # get condition.
            condition = img_fname.rstrip(".jpg").split("_")[-1]
            condition = condition.lstrip("condition")
            conditions.append(condition)

            # break for testing.
            if len(conditions) > 500:
                print("break early for testing")
                break

    # Populate an empty array to hold images
    #of the final shape (255, 255, 3, n)
    # Also populate an empty array of target labels.
    n = len(conditions)
    X_shape = (n, 255, 255, 3)
    final_img_shape = (255, 255, 3) # scale images to this shape.
    y_shape = (n,)

    print("create empty target array")
    y = []
    
    print("create empty X array")
    X = np.zeros(X_shape, dtype=np.int16)

    # Start filling in this array with real images, 
    # preprocessing as necessary.
    for idx, img_p in enumerate(img_paths):

        try:

            # Load the image as PIL object.
            img = Image.open(img_p)

            # Preprocess.
            # Rotate if needed.
            # Scale.
            # Cast to numpy arary.
            img_mtx = preprocess_img(img, final_img_shape)

            # Visualize image for testing.
            print(img_mtx.shape)
            plt.imshow(img_mtx)
            plt.show()

            # Add to X, y array.
            X[idx, :, :, :] = img_mtx
            y.append(conditions[idx])

            # Visualize image for testing.
            to_show = X[idx]
            print(to_show.shape)
            plt.imshow(X[idx, :, :, :])
            plt.show()


        except Exception as e:
            print(e)

        # counter.
        if idx % 10 == 0:
            print("\t...{}/{}".format(idx, n))
        if idx > 100:
            print("break for testing")
            break

    # Write dataset.
    of_base_p = os.path.join("..", "data", "ml_dataset")
    X_p = os.path.join(of_base_p, "X.npy")
    y_p = os.path.join(of_base_p, "y.npy")

    print("save y array")
    np.save(file=y_p, arr=np.array(y))

    print("save x array")
    np.save(file=X_p, arr=X)

    print("done")


if __name__ == "__main__":

    create_dataset_wrapper()

