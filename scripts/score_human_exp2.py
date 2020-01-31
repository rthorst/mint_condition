import os
import pandas as pd

def calculate_percent_ask_to_see_machine_score():
    """ calculate % of people who ask to see machine score """

    # Load data.
    data_p = os.path.join("..", "data", "human_quiz2", 
            "Grade Trading Cards!.csv")
    df = pd.read_csv(data_p)

    # Get columns containing choices (y/n) about seeing human ratings.
    colnames = [cname for cname in df.columns if "Would you like" in cname]

    # Count number of yes, no responses.
    n_yes = 0
    n_no = 0
    for cname in colnames:
        for resp in df[cname].values:
            
            n_yes += int(resp == "Yes")
            n_no += int(resp == "No")

    # Average and report percent yes.
    perc_yes = 100 *n_yes / (n_yes + n_no)
    msg = "Percent want to see machine score : {:.2f}%".format(perc_yes)
    print(msg)

def calculate_rerating_scores():
    """
    When people see the model's grade, do they change their mind?
    If so, do they change their mind towards the model?
    """

    # Load data.
    data_p = os.path.join("..", "data", "human_quiz2", 
            "Grade Trading Cards!.csv")
    df = pd.read_csv(data_p)

    # Get tuples of (old_score, new_score, model_score).
    # Ignore cases where old_score == model_score (e.g. right initially). 
    initial_cnames = [cname for cname in df.columns if 
        "I rate the condition of card" in cname]
    new_cnames = [cname for cname in df.columns if 
        "I would now grade" in cname]
    machine_ratings = [3, 2, 1, 5, 3, 5, 4, 1, 2, 4]

    # For each tuple, calculate:
    changed_yes = 0 # number of ratings changed after seeing machine.
    changed_no = 0 # num not changed ...
    change_towards_machine = 0 # num changed towards machine rating. 
    change_away_from_machine = 0 # num changed away ...

    for initial_cname, new_cname, machine_rating in zip(
                initial_cnames, new_cnames, machine_ratings):

        initial_ratings = df[initial_cname].values # (n_participants, 1)
        new_ratings = df[new_cname].values # (n_participants, 1)

        for init, new in zip(initial_ratings, new_ratings):

            # Ignore cases where the initial rating was corrct, which
            # imply no opportunity for change.
            if init == machine_rating:
                continue

            # Check if changed rating at all.
            changed = int(init != new)
            changed_yes += changed
            changed_no += (1 - changed)

            # If changed rating, check if changed towards or away from machine
            # rating.
            if changed:

                old_diff = abs(init - machine_rating)
                new_diff = abs(new - machine_rating)

                if new_diff < old_diff: 
                    change_towards_machine += 1

                elif old_diff < new_diff:
                    change_towards_machine += 1

                else: # Could be equal, e.g. 
                    pass

    # Summarize results.
    perc_change = 100 *changed_yes / (changed_yes + changed_no)
    perc_towards_machine = 100 * change_towards_machine / (change_towards_machine + change_away_from_machine)
    msg = """
    Percent change rating based on machine : {:.2f}%,
    Percent change rating towards (rather than away) from machine: {:.2f}%
    """.format(perc_change, perc_towards_machine)

    print(msg)
    print(change_towards_machine, change_away_from_machine)

if __name__ == "__main__":
    calculate_percent_ask_to_see_machine_score()
    calculate_rerating_scores()
