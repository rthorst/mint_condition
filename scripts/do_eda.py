"""
Exploratory data analysis.

1. How many examples are there of each condition? Of each sport?
2. Visualize lots of example images.
3. Are there "bad" images of various kinds that shouldn't be in my
data?
"""
import os
from collections import Counter 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def count_where(sport, condition, sports, conditions):
    """
    Helper function to count times where a particular sport/condition
    combination occurs.

    Parameters
    --------
    sport (str) e.g. "basketball"
    condition (str) e.g. "MINT"
    sports (str [])
    conditions (str [])

    Returns
    freq (int) : e.g. number of times a card is "MINT" + "basketball"
    """
    freq = 0
    for sport_i, condition_i in zip(sports, conditions):
        if (sport_i == sport) and (condition_i == condition):
            freq += 1

    return freq

def get_distribution_information():
    """
    Get information about distribution, currently of sports and conditions.
    TODO make a plot if time.
    """

    # List condition and sport of each card.
    base_p = os.path.join("..", "data", "imgs")
    conditions = []
    sports = []

    sport_to_lbl = {
        185223 : "baseball", 
        185226 : "basketball",
        185224 : "football",
        185225 : "hockey"
    }

    for dir_name in os.listdir(base_p):
        dir_p = os.path.join(base_p, dir_name)
        for img_fname in os.listdir(dir_p):
            
            # extract sport.
            sport_int = int(img_fname.split("_")[1].lstrip("sport")) # eg 185225
            sport = sport_to_lbl[sport_int] # e.g. "baseball"

            # add condition, sport to lists.
            conditions.append(dir_name)
            sports.append(sport)

    # Count total cards.
    n = len(conditions)
    print("total number of cards = {}".format(n))
    
    # Count conditions.
    print("conditions: lots of mint, not much fair/poor")
    condition_c = Counter(conditions)
    print(condition_c)
    
    # Count sports.
    print("sports: mostly hockey and baseball")
    sport_c = Counter(sports)
    print(sport_c)

    """
    Visualize sport X condition distribution (2x2)
    """

    # Create a matrix of shape (n_sport, n_conditions) 
    # Where cells are frequencies.
    unique_sports = list(set(sports))
    unique_conditions = list(set(conditions))
    heatmap_mtx = np.zeros((len(unique_sports), len(unique_conditions)),
                            dtype=np.int64)
    
    for i, sport in enumerate(unique_sports):
        for j, condition in enumerate(unique_conditions):
            freq = count_where(sport, condition, sports, conditions)
            heatmap_mtx[i][j] = freq

    # Visualize the 2x2 frequency table as a heatmap.
    print(heatmap_mtx)
    sns.heatmap(heatmap_mtx)
    plt.show()

def visualize_example_images(n_imgs = 9):

    # list all image paths.
    print("list all image paths")
    img_ps = []
    base_p = os.path.join("..", "data", "imgs")
    for dir_name in os.listdir(base_p):    
        dir_p = os.path.join(base_p, dir_name)
        for fname in os.listdir(dir_p):
            f_p = os.path.join(dir_p, fname)
            img_ps.append(f_p)

    # select 9 random images.
    rand_imgs = np.random.choice(img_ps, n_imgs)

    # show those images.
    for p in rand_imgs:

        img = Image.open(p)
        n = np.array(img)
        plt.imshow(n)
        plt.title(p)
        plt.show()

def find_bad_images():
    pass

if __name__ == "__main__":

    #get_distribution_information()
    visualize_example_images(n_imgs=50)
    #find_bad_images()
