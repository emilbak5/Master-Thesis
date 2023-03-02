import os
import json



with open('annotations.json', 'r') as f:
    annotations = json.load(f)

for image in annotations['images']:
    start_idx = image['file_name'].rfind('\\')
    new_name = image['file_name'][start_idx+1:]
    new_name = new_name.replace('.png', '.jpg')
    image['file_name'] = new_name

with open('annotations.json', 'w') as f:
    json.dump(annotations, f, indent=4)
    

