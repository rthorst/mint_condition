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
    PIL_object = PIL_object.resize((new_width, new_height), Image.ANTIALIAS)
    # Return new image as numpy array.
    return np.array(PIL_object)

def create_dataset_wrapper():
    """
    Main wrapper to create dataset.
    """

    # List file paths to load.
    # Also save original filename.
    print("list file paths to images in the dataset")
    print("also extract filename of each card")
    print("currently keep only baseball images")
    img_paths = []
    filenames = []
    base_p = os.path.join("..", "data", "imgs")

    for dir_name in os.listdir(base_p):

        if dir_name not in ["MINT", "NM", "EX", "VG", "POOR"]:
            print("skip {}".format(dir_name))
            continue

        print("\t...{}".format(dir_name))
        dir_p = os.path.join(base_p, dir_name)

        for img_fname in os.listdir(dir_p):

            # get path.
            img_p = os.path.join(dir_p, img_fname)
            img_paths.append(img_p)
            
            # get filename
            filenames.append(img_fname)

    # Populate an empty array to hold images
    #of the final shape (255, 255, 3, n)
    # Also populate an empty array of target labels.
    n = len(filenames)
    final_img_shape = (255, 255, 3) # scale images to this shape.

    # Preprocess images and write in matrix form.
    for idx, img_p in enumerate(img_paths):

        try:

            # Load the image as PIL object.
            img = Image.open(img_p)

            # Preprocess.
            # Rotate if needed.
            # Scale.
            # Reshape to (3, 255, 255)
            # Cast to numpy arary.
            img_mtx = preprocess_img(img, final_img_shape)

            # Ignore improperly shaped images, usually, due to graysale thus no color channel.
            if img_mtx.shape != final_img_shape:
                print("skip image with improper finals shape {}".format(img_mtx.shape))
                continue

            # Write output.
            base_p = os.path.join("..", "data", "preprocessed_imgs")
            out_fname = filenames[idx].replace(".jpg", "_preprocessed.npy")
            out_p = os.path.join(base_p, out_fname)
            np.save(file=out_p, arr=img_mtx)

        except Exception as e:
            print(e)

        # counter.
        if idx == 0:
            plt.imshow(img_mtx)
            plt.title("Example image: does this look right?")
            plt.show()
        if idx % 10 == 0:
            print("\t...preprocessed {}/{} images".format(idx, n))

    print("done")


if __name__ == "__main__":

    create_dataset_wrapper()

