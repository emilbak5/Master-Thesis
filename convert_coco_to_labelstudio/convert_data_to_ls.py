from label_studio_converter.imports.coco import convert_coco_to_ls
import json


DATA_PATH = '/data/local-files/?d=images%5C'

INFILE = 'annotations.json'
OUTFILE = 'annotations.ls.json'


def main():

    with open(INFILE, 'r') as f:
        data = json.load(f)
    
    # for annotation in data['annotations']:
    #     # remove segmentation
    #     annotation.pop('segmentation', None)
    
    # save the modified json
    with open(OUTFILE, 'w') as f:
        json.dump(data, f, indent=4)
    

    convert_coco_to_ls(input_file=OUTFILE, out_file=OUTFILE, image_root_url=DATA_PATH)



if __name__ == '__main__':
    main()




