"""
Upload images to Amazon S3
"""

import os

base_p = os.path.join("..", "data", "imgs")
for dir_name in os.listdir(base_p):

    dir_p = os.path.join(base_p, dir_name)
    for fname in os.listdir(dir_p):

        img_p = os.path.join(dir_p, fname)
        cmd = "aws s3 cp \"{}\" s3://mintcondition/imgs".format(img_p)
        print(cmd)
        os.system(cmd)
        break
