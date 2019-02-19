# encoding: utf-8
import requests
URL = "http://localhost:5000/predict"
IMAGE_PATH = "cat.jpg"
image = open(IMAGE_PATH,'rb').read()
payload = {"image":image}

r = requests.post(URL,files=payload).json()

if r["success"]:
    for (idx,result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(idx+1, result["label"], result["probability"]))
else:
    print("Request failed.")
