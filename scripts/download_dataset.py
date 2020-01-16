import json
import os
import requests
import time 
from bs4 import BeautifulSoup

"""
Stopped: 
There is a bug in the HTML stopping records from writing as proper JSON
I think the backslashes are the problem.


"""

def download_beckett_htmls():
    """
    Download raw HTML of beckett baseball card pages
    """

    base_url = "https://marketplace.beckett.com/search_new/?sport=185223&rowNum=250&page="
    
    for page_num in range(1, 11):

        # Download page.
        url = base_url + str(page_num)

        # Get text.
        resp = requests.get(url)
        text = resp.text

        # Save.
        fname = "baseball_p{}.html".format(page_num)
        of_p = os.path.join("..", "data", "htmls", fname)
        with open(of_p, "w", newline="") as of:
            of.write(text)
        print(fname)

        # Sleep to be nice.
        time.sleep(1)

def extract_html_of_individual_records():
    """ 
    Extract individual records, currently image URL and condition.
    Create data/individual_records.jsonl
    """

    # Create output file.
    of_p = os.path.join("..", "data", "individual_records.jsonl")
    print("create {}".format(of_p))
    of = open(of_p, "w", newline="")

    # Iterate over input files.
    data_p = os.path.join("..", "data", "htmls")
    for html_fname in os.listdir(data_p):

        # Load HTML.
        print(html_fname)
        html_filepath = os.path.join(data_p, html_fname)
        html = open(html_filepath).read()
    
        # Cast HTML to beautiful soup object.
        soup = BeautifulSoup(html, "html.parser")

        # List all mpl2 and mpl5 tags.
        mpl2s = soup.findAll("li", {"class" : "mpL2"})
        mpl5s = soup.findAll("li", {"class" : "mpL5"})
        assert len(mpl2s) == 250
        assert len(mpl5s) == 250

        # Write output.
        for mpl2_tag, mpl5_tag in zip(mpl2s, mpl5s):

            out = json.dumps({
                "img_html" : remove_slashes(str(mpl2_tag)),
                "condition_html" : remove_slashes(str(mpl5_tag)),
            })
            of.write(json.dumps(out) + "\n")

    # Done.
    of.flush()
    of.close()
    print("Done, wrote {}".format(of_p))

def remove_slashes(s):
    """ helper function to remove slashes """

    s = s.replace("/", "")
    s = s.replace("\\", "")
    return s

def extract_url_and_condition_from_individual_records():
    """
    Create a dataset of image URL and condition.
    """

    # Load input file.
    in_p = os.path.join("..", "data", "individual_records.jsonl")
    print("read {}".format(in_p))
    rows = open(in_p, "r").readlines()

    # Create output file.
    of_p = os.path.join("..", "data", "clean_individual_records.jsonl")
    of = open(of_p, "w", newline="")

    # Clean each line, extracting image URL and condition.
    for row in rows:

        # Get image url.
        j = json.loads(row)
        print(j)
        img_soup = BeautifulSoup(j["img_html"])
        img_url = soup.find("image", {"li" : "data-original"})
        print(img_url)

"""
image.
li
data-original

condition.
li title
"""

if __name__ == "__main__":
    #download_beckett_htmls()
    extract_html_of_individual_records()    
    extract_url_and_condition_from_individual_records()

