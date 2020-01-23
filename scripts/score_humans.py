"""
Score human accuracy classifying images.
"""
import pandas as pd
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

def get_M_human_acc():

    # List paths to data files.
    base_p = os.path.join("..", "data", "human_quiz")
    csv_fnames = [fname for fname in os.listdir(base_p) if 
                  fname.endswith(".csv")]
    csv_ps = [os.path.join(base_p, fname) for fname in csv_fnames]

    # Load and score each file.
    human_accuracies = [] # float[] one score per participant range 0-100.
    for data_p in csv_ps:

        # Load data.
        df = pd.read_csv(data_p)
        df["correct"] = np.where(df.prediction == df.true_grade, 1, 0)

        # Score.
        prop_correct = sum(df.correct) / len(df)
        perc_correct = 100 * prop_correct

        # Add to human accuracy array.
        human_accuracies.append(perc_correct)

    # Return M accuracy.
    M_acc = np.mean(human_accuracies)
    print("M human acc = {:.4f} percent".format(M_acc))


def get_human_percent_agreement():
    """
    For each item, what is the probability that any two random raters 
    would agree.

    Consider all two pairs of raters.
    For each of the 15 items, calculate percent of time they agree.
    Average those.
    """

    # List paths to data files.
    base_p = os.path.join("..", "data", "human_quiz")
    csv_fnames = [fname for fname in os.listdir(base_p) if 
                  fname.endswith(".csv")]
    csv_ps = [os.path.join(base_p, fname) for fname in csv_fnames]

    percent_agreements = [] # overall agreements between each pair of 2 raters.
    for p1, p2 in itertools.combinations(csv_ps, 2):

        # Load data.
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)

        # Compute agreement between these two raters.
        items = set(df1.img)
        num_agree = 0
        num_disagree = 0
        for item in items:

            # get user1, 2 ratings.
            rating1 = df1[df1.img==item].prediction.values[0]
            rating2 = df2[df2.img==item].prediction.values[0]

            # computer agreement and store.
            if rating1 == rating2:
                num_agree += 1
            else: 
                num_disagree += 1

        # Add agreement between these two raters to overall agreement.
        perc_agree = 100*num_agree / (num_agree + num_disagree)
        percent_agreements.append(perc_agree)

    # Output mean interrater agreement to shell.
    M_agreement = np.mean(percent_agreements)
    msg = "M percent agreement = {:.4f}".format(M_agreement)
    print(msg)


def plot_humans_vs_machine():
    """ simple plots of humans vs. machines as SVG"""

    human_acc = 31.111
    human_agreement = 34.074
    machine_acc = 63.0
    machine_agreement = 100

    # Accuracy.
    plt.bar(range(2), [human_acc, machine_acc])    
    plt.xticks(range(2), ["Human", "Machine"])
    plt.ylabel("Percent Accuracy")
    fig_p = os.path.join("..", "results", "human_vs_machine_accuracy.svg")
    plt.savefig(fig_p)

    # Agreement.
    plt.close("all")
    plt.bar(range(2), [human_agreement, machine_agreement])
    plt.xticks(range(2), ["Human", "Machine"])
    plt.ylabel("Percent Agreement")
    fig_p = os.path.join("..", "results", "human_vs_machine_agreement.svg")
    plt.savefig(fig_p)

if __name__ == "__main__":

    #get_M_human_acc()
    #get_human_percent_agreement()
    plot_humans_vs_machine()
