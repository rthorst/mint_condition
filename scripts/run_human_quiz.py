import os
import matplotlib.pyplot as plt
from PIL import Image
import time
import csv
import numpy as np
import random

# Create output file.
curr_time = time.time()
of_p = os.path.join("..", "data", "human_quiz", "{}.csv".format(curr_time))
print("make {}".format(of_p))
of = open(of_p, "w", newline="")
header = ["img", "prediction", "true_grade"]
w = csv.writer(of)
w.writerow(header)

# Quiz images.
base_p = os.path.join("..", "data", "human_quiz")
fnames = [fname for fname in os.listdir(base_p) if fname.endswith(".jpg")]
random.shuffle(fnames)

corrects = 0
incorrects = 0
for fname in fnames:

    # Show image.
    p = os.path.join(base_p, fname)
    img = Image.open(p)
    img = np.array(img)
    plt.imshow(img)
    plt.show()

    # Get grade.
    msg = """
    I would grade this (1-5).
    5 = best ... 1 = worst
    """
    grade = input(msg)

    # Get true grade.
    if "mint" in fname:
        true_grade = "5"
    elif "nm" in fname:
        true_grade = "4"
    elif "ex" in fname:
        true_grade = "3"
    elif "vg" in fname:
        true_grade = "2"
    elif "poor" in fname:
        true_grade = "1"

    # Write output.
    out = [fname, grade, true_grade]
    w.writerow(out)

    # Score.
    if grade == true_grade:
        corrects += 1
    else: 
        incorrects += 1

# Done.
of.flush()
of.close()
print("Thanks! Done!")
acc = corrects / (corrects + incorrects)
print("Accuracy", acc)
