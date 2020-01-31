from ebaysdk.finding import Connection
import json 
import pickle
import requests 
import time 
from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt

def find_cards():
    # List card URLs from Ebay API

    client_id = "RobertTh-insight-PRD-8ca8111fb-03bf72d2"
    dev_id = "c4cb3652-b695-4442-8c9f-7b14f62faecc"
    client_secret = "PRD-ca8111fbb25b-ea35-40b0-9532-55d7"
    api = Connection(appid=client_id, config_file=None)

    grades = [10, 8, 6, 4, 2]
    for grade in grades:
        for page_num in range(1, 100):

            # get data.
            params = {
             "categoryName": "Baseball Cards",
             "keywords" : "psa {}".format(grade),
             "paginationInput" : 
                    {
                        "entriesPerPage" : 100,
                        "pageNumber" : page_num
                    },
             "outputSelector" : "PictureURLLarge"
             }
            response = api.execute("findItemsAdvanced", params)
            j = response.json()

            # save
            of_p = "psa_grade_{}_page_{}.pkl".format(grade, page_num)
            of_p = os.path.join("page_jsons", of_p)
            print(of_p)
            with open(of_p, "wb") as of:
                pickle.dump(j, of)

def download_images():
    # Download card Images from Ebay.

    # Iterate over json files.
    for page_num in range(1, 100):
        for grade in [10, 8, 6, 4, 2]:

            # Load appropriate json file.
            in_p = "page_jsons/psa_grade_{}_page_{}.pkl".format(grade, page_num)
            j = pickle.load(open(in_p, "rb"))
            j = json.loads(j)
            print(j.keys())

            # Get data.
            items = j["searchResult"]["item"] # json[] length 100.
            for idx, item in enumerate(items):

                try: 

                    # List URL.
                    #url = item["galleryURL"]
                    url = item["pictureURLLarge"]

                    # Download img.
                    of_p = "imgs/grade_{}_page_{}_id_{}.jpg".format(grade, page_num, idx)
                    res = requests.get(url).content
                    print(of_p)

                    # Write to file.
                    with open(of_p, "wb") as of:
                        of.write(res)

                    # Sleep 1s.
                    time.sleep(0.25)

                except Exception as e:
                    print(e)
                    time.sleep(0.25)


def auto_crop_images():


    counter = 0   
    n = len(os.listdir("imgs"))
    for fname in os.listdir("imgs"):

        try:

            # Counter.
            counter += 1
            if counter % 100 == 0:
                print("{}/{}".format(counter, n))

            # Skip if grade not in [2, 4, 6, 8, 10].
            grade = fname.split("_")[1]
            if grade not in ["2", "4", "6", "8", "10"]:
                continue 

            # Load
            p = os.path.join("imgs", fname)
            img = Image.open(p)
            img = np.array(img)

            # Crop
            width, height, channels = img.shape
            new_top = int(height/3)
            cropped = img[new_top:, :, :]

            of_p = os.path.join("cropped_imgs", fname)
            PIL_img = Image.fromarray(cropped)
            PIL_img.save(of_p)
            #np.save(file=of_p, arr=cropped)

        except Exception as e:

            print(e)

if __name__ == "__main__":

    #find_cards()
    #download_images()
    auto_crop_images()
