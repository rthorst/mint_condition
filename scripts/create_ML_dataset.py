"""
Create dataset suitable for machine learning.
"""

from PIL import Image
import numpy as np
from scipy.ndimage.interpolation import rotate

def preprocess_img(img_mtx, output_shape=(255, 255, 3)):
    """
    Wrapper to preprocess a single image as PIL image object.
    Should generalize to new images eventually inputted online.

    img_p (string) : path to image.
    
    """ 

    # If width > height, rotate 90 degrees.
    if img_matrix.shape[1] > img_matrix.shape[0]:
        img_matrix = Image.fromarray(img_matrix).rotate(90)
        img_matrix = np.array(img_matrix)

    # Scale to final shape, for example (255, 255, 3)
    img_matrix = imresize(img_matrix, output_shape)

    # Return new image.
    return img_matrix 

def create_dataset_wrapper():
    """
    Main wrapper to create dataset.
    """

    # List file paths to load.
    # Also extract condition of the card from the filename.

    print("list file paths to images in the dataset")
    print("also extract condition of each card from the filename")
    img_paths = []
    conditions = []
    base_p = os.path.join("..", "data", "imgs")
    for dir_name in os.listdir(imgs):
        print("\t...{}".format(dir_name))
        dir_p = os.path.join(base_p, dir_name)

        for img_fname in os.listdir(dir_p):

            # get path.
            img_p = os.path.join(dir_p, img_fname)
            img_paths.append(img_p)
            
            # get condition.
            condition = img_fname.rstrip(".jpg").split("_")[-1]
            condition = condition.lstrip("condition")
            conditions.append(condition)

    # Populate an empty array to hold images
    #of the final shape (255, 255, 3, n)
    # Also populate an empty array of target labels.
    n = len(conditions)
    X_shape = (255, 255, 3, n)
    final_img_shape = (255, 255, 3) # scale images to this shape.
    y_shape = (n,)

    print("create empty target array")
    y = []
    
    print("create empty X array")
    X = np.zeros(X_shape)

    # Start filling in this array with real images, 
    # preprocessing as necessary.
    for idx, img_p in enumerate(img_paths):

        try:

            # Load the image.
            img = Image.open(img_p)

            # Preprocess.
            # Rotate if needed.
            # Scale.
            img = preprocess_img(img, final_img_shape)

            # Add to X, y array.
            X[idx] = img
            y.append(conditions[idx])

        except Exception as e:

            print(e)

        # counter.
        if idx % 10 == 0:
            print("\t...{}/{}".format(idx, n))

    # Write dataset.
    of_base_p = os.path.join("..", "data", "ML_dataset")
    X_p = os.path.join(of_base_p, "X.npy")
    y_p = os.path.join(of_base_p, "y.npy")

    print("save y array")
    np.save(file=y_p, arr=np.array(y))

    print("save x array")
    np.save(file=x_p, arr=X)

    print("done")


if __name__ == "__main__":

    create_dataset_wrapper()

