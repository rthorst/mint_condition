import json
import os
import requests
import time 
from bs4 import BeautifulSoup
import itertools
import random 
import shutil

def download_beckett_htmls():
    """
    Download raw HTML of beckett baseball card pages
    """

    base_url = "https://marketplace.beckett.com/search_new/?rowNum=250"
    page_nums = range(1, 51)
    conditions = ["NM", "MINT", "EX", "VG", "GOOD", "FAIR", "POOR"]
    sports = [
        185223, #baseball
        185226, #basketball
        185224, #football
        185225  #hockey
    ]    
    
    for page_num in page_nums:
        for condition in conditions: 
            for sport in sports:

                # Download page.
                url = "{}&page={}&condition_id={}&sport={}".format(base_url, page_num, condition, sport)
                print(url)
                import time

                # Get text.
                resp = requests.get(url)
                text = resp.text

                # Save.
                fname = "{}_p{}_condition{}.html".format(sport, page_num, condition)
                of_p = os.path.join("..", "data", "htmls", fname)
                with open(of_p, "w", newline="") as of:
                    of.write(text)
                print(fname)

                # Sleep to be nice.
                time.sleep(1)

def extract_individual_records():
    """ 
    Extract individual records, currently image URL and condition.
    Also assign each record a unique numerical ID.
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
    id = 1 # counter: unique numerical ID for each record. 
    for html_fname in html_fnames:

        # Load HTML.
        print("extract records from {}".format(html_fname))
        html_filepath = os.path.join(data_p, html_fname)
        html = open(html_filepath).read()

        # Extract condition from the filename.
        # Note that before I had to extract it from mpl5-span-title.
        condition = html_fname.rstrip(".html").split("_")[-1]
        condition = condition.lstrip("condition")
        print(condition)

        # Extract sport from the filename.
        sport = html_fname.split("_")[0]

        # Cast HTML to beautiful soup object.
        soup = BeautifulSoup(html, "html.parser")

        # List all mpl2 tags which contain the image URLs.
        mpl2s = soup.findAll("li", {"class" : "mpL2"}) # contains image

        # Write output.
        for mpl2_tag in mpl2s:

            try:

                # Get image URL
                img_tag = mpl2_tag.findChildren("img")[0]
                img_url = img_tag["data-original"]

                # Write.
                out = json.dumps({
                    "id" : id, # unique numerical ID
                    "image_url" : img_url,
                    "condition" : condition, # same for every img in file
                    "sport" : sport
                })
                of.write(out + "\n")

                # Increment numerical ID.
                id += 1

            except Exception as e:

                print(e)

    # Done.
    of.flush()
    of.close()
    print("Done, wrote {}".format(of_p))

def clean_individual_records():
    """
    Clean individual records -- currently, skip images with no-image
    placeholder image.

    Also skip stock photos (with items_stock in the filepath) which
    are the vast majority.

    """

    print("clean individual records -- remove images with no-image placeholder")

    # load input file
    print("load input file")
    in_p = os.path.join("..", "data", "individual_records.jsonl")
    js = [json.loads(row) for row in open(in_p, "r").readlines()]

    # create output file.
    print("create output file")
    of_p = os.path.join("..", "data", "cleaned_individual_records.jsonl")
    of = open(of_p, "w", newline="")

    print("clean and write")
    missing_images_skipped = 0 # skip if no-image
    stock_photos_skipped = 0 # skip if stock image.
    images_wrote = 0 # images not skipped -> successfully wrote.
    for j in js:

        if "no-image.jpg" in j["image_url"]:
            missing_images_skipped += 1

        elif "items_stock" in j["image_url"]:
            stock_photos_skipped += 1

        else:
            of.write(json.dumps(j) + "\n")
            images_wrote += 1
    # done.
    of.flush()
    of.close()
    msg = """
    Done. Ignored {} missing images and {} stock photos.
    Wrote {} total photos.
    """.format(missing_images_skipped, stock_photos_skipped, images_wrote)
    print(msg)

def download_images(start_at_idx=0):
    """
    For each record in cleaned_individual_records.jsonl, download the associated image.
    It should be saved in data/{condition}/id_condition.jpg

    Start_at_idx (int) :: used to resume scraping from idx > 0.
    """

    # make 1 directory per conditin to hold images.
    conditions = ["NM", "MINT", "EX", "VG", "GOOD", "FAIR", "POOR"]
    for condition in conditions:
       dir_p = os.path.join("..", "data", "imgs", condition)
       if not os.path.exists(dir_p):
            os.mkdir(dir_p)

    # load individual records to get images for.
    in_p = os.path.join("..", "data", "cleaned_individual_records.jsonl")
    print("read {}".format(in_p))
    js = [json.loads(row) for row in open(in_p, "r").readlines()]

    # get images.
    print("get images")
    for idx, j in enumerate(js):

        # check if we need to skip this index because already processed.
        if idx < start_at_idx:
            msg = "skip {} because already crawled"
            print(msg)
            continue

        # get url, id, condition.
        img_url = j["image_url"]
        id = j["id"]
        condition = j["condition"]
        sport = j["sport"]

        # designate output file path.
        fname = "{}_sport{}_condition{}.jpg".format(id, sport, condition)
        of_p = os.path.join("..", "data", "imgs", condition, fname)

        # download image.
        try:
            resp = requests.get(img_url).content
            with open(of_p, "wb") as of:
                of.write(resp)

        except Exception as e:
            print(e, img_url)

        # sleep 1s to be nice.
        time.sleep(1)

        # counter.
        if idx % 10 == 0:
            print(idx)

if __name__ == "__main__":
    
    download_beckett_htmls()
    extract_individual_records()    
    clean_individual_records()
    download_images()

