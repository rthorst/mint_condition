import requests
import urllib

base_url = "https://2rhehmesf4.execute-api.us-east-1.amazonaws.com/api/grade_url"
image_url = "https://images.psacard.com/s3/cu-psa/psacard/images/photograde/gretzky18_9xl.jpg"

image_url = urllib.parse.quote(image_url)
print(image_url)
