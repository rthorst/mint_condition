"""
Score human accuracy classifying images --- Experiment 1
(Accuracy with insight fellows).
"""
import pandas as pd
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

def get_bootstrapped_ci(arr, n_bootstraps=1000):
    """ 
    get bootstrapped 95% confidence interval for mean of a numerical array.

    parameters:
    arr (numerical [])
    n_boostraps (int) : number of bootstrapped samples to take.

    returns:
    (lower, upper) (tuple of numerical values for 2.5%, 97.5%ile).
    """

    bootstrapped_samples = np.random.choice(arr, size=(n_bootstraps, len(arr)))
    bootstrapped_Ms = np.mean(bootstrapped_samples, axis=1) # shape (n_bootstraps,)

    # get lower, uppper M.
    lower = np.percentile(bootstrapped_Ms, q=2.5)
    upper = np.percentile(bootstrapped_Ms, q=97.5)

    return (lower, upper) 

def get_M_human_acc():

    # List paths to data files.
    base_p = os.path.join("..", "data", "human_quiz1")
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

    # Get M accuracy.
    M_acc = np.mean(human_accuracies)
    print("M human acc = {:.4f} percent".format(M_acc))

    # Get Boostrapped 95% CI for accuracy.
    lower, upper = get_bootstrapped_ci(human_accuracies)
    half_ci = abs(upper-lower)/2
    print("95% CI for humans: lower {} upper {}".format(lower, upper))
    print("Half CI = {}".format(half_ci))

def get_human_percent_agreement():
    """
    For each item, what is the probability that any two random raters 
    would agree.

    Consider all two pairs of raters.
    For each of the 15 items, calculate percent of time they agree.
    Average those.
    """

    # List paths to data files.
    base_p = os.path.join("..", "data", "human_quiz1")
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

    # Output CI on agreements.
    lower, upper = get_bootstrapped_ci(percent_agreements)
    half_ci = 0.5 * abs(upper-lower)
    msg = "CI for agreement: upper {} lower {} half {}".format(upper, lower, half_ci)
    print(msg)

def plot_humans_vs_machine():
    """ simple plots of humans vs. machines as SVG"""

    # Means.
    human_acc = 31.111
    human_agreement = 34.074
    machine_acc = 63.0
    machine_agreement = 100

    # Errors.
    human_acc_ci = 4.44 # half CI.
    human_agreement_ci = 4.53 # half CI.

    # Global plotting parameters.
    sns.set_style("white", {"axes.grid" : False})
    sns.set(font_scale = 1.5)
    sns.set_context("poster")
    figsize = (2, 6)

    # Accuracy.
    plt.bar(range(2), [human_acc, machine_acc], yerr=[human_acc_ci, 0])    
    plt.xticks(range(2), ["Human\nAmateur", "Mint\nCondition"])
    plt.ylabel("Percent Accuracy")

    for file_extension in ["png", "svg"]:
        fig_p = os.path.join("..", "results", "human_vs_machine_accuracy.{}".format(file_extension))
        plt.tight_layout()
        plt.savefig(fig_p, figsize=figsize)


    # Agreement.
    plt.close("all")
    plt.bar(range(2), [human_agreement, machine_agreement], yerr = [human_agreement_ci, 0])
    plt.xticks(range(2), ["Human", "Machine"])
    plt.ylabel("Percent Agreement")

    for file_extension in ["png", "svg"]:
        fig_p = os.path.join("..", "results", "human_vs_machine_agreement.{}".format(file_extension))
        plt.tight_layout()
        plt.savefig(fig_p, figsize=figsize)

    # Grouped Bar.
    plt.close("all")
    human_Ms = [human_acc, human_agreement, 69]
    machine_Ms = [machine_acc, machine_agreement]
    human_errs = [human_acc_ci, human_agreement_ci, 0]
    width = 0.4

    # Plot humans.
    plt.figure(figsize=(16, 9))
    plt.bar(range(3), human_Ms, yerr=human_errs, width=width)
   
    # Plot machines.
    plt.bar(np.arange(2) + width, machine_Ms, width=width, color="green")

    # Show
    plt.gca().grid(False)
    plt.tight_layout()
    fig_p = os.path.join("..", "results", "grouped_bar.png")
    plt.savefig(fig_p, figsize = (16, 9))
    plt.show()


    # Want AI Grade
    plt.close("all")
    plt.bar(range(1), [69])
    plt.xticks(range(2), ["Human", "Machine"])
    plt.ylabel("Wang AI Grade")

    for file_extension in ["png", "svg"]:
        fig_p = os.path.join("..", "results", "want_ai_grade.{}".format(file_extension))
        plt.tight_layout()
        plt.savefig(fig_p, figsize=figsize)


if __name__ == "__main__":

    #get_M_human_acc()
    #get_human_percent_agreement()
    plot_humans_vs_machine()
