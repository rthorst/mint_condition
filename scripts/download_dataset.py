import json
import os
import requests
import time 
from bs4 import BeautifulSoup
import itertools
import random 


def download_beckett_htmls():
    """
    Download raw HTML of beckett baseball card pages
    """

    base_url = "https://marketplace.beckett.com/search_new/?sport=185223&rowNum=250"
    page_nums = range(1, 11)
    conditions = ["NM", "MINT", "EX", "VG", "GOOD", "FAIR", "POOR"]
    
    for page_num in range(1, 11):
        for condition in conditions: 

            # Download page.
            url = "{}&page={}&condition_id={}".format(base_url, page_num, condition)
            print(url)
            import time

            # Get text.
            resp = requests.get(url)
            text = resp.text

            # Save.
            fname = "baseball_p{}_condition{}.html".format(page_num, condition)
            of_p = os.path.join("..", "data", "htmls", fname)
            with open(of_p, "w", newline="") as of:
                of.write(text)
            print(fname)

            # Sleep to be nice.
            time.sleep(1)

def extract_individual_records():
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
    html_fnames = os.listdir(data_p)
    random.shuffle(html_fnames)
    for html_fname in html_fnames:

        # Load HTML.
        print(html_fname)
        html_filepath = os.path.join(data_p, html_fname)
        html = open(html_filepath).read()

        # Extract condition from the filename.
        # Note that before I had to extract it from mpl5-span-title.
        condition = html_fname.rstrip(".html").split("_")[-1]
        condition = condition.lstrip("condition")
        print(condition)

        # Cast HTML to beautiful soup object.
        soup = BeautifulSoup(html, "html.parser")

        # List all mpl2 tags which contain the image URLs.
        mpl2s = soup.findAll("li", {"class" : "mpL2"}) # contains image
        assert len(mpl2s) == 250

        # Write output.
        for mpl2_tag in mpl2s:

            try:

                # Get image URL
                img_tag = mpl2_tag.findChildren("img")[0]
                img_url = img_tag["data-original"]

                # Write.
                out = json.dumps({
                    "image_url" : img_url,
                    "img_grade" : condition # same for every img in file
                })
                of.write(out + "\n")

            except Exception as e:

                print(e)

    # Done.
    of.flush()
    of.close()
    print("Done, wrote {}".format(of_p))

"""
image.
li
data-original

condition.
li title
"""

if __name__ == "__main__":
    
    #download_beckett_htmls()
    extract_individual_records()    

