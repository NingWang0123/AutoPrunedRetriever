import json


with open("meta_codebook_new2.json", "w") as f:
    data = json.load(f)


print(data.keys())

#python checking_retrievals.py