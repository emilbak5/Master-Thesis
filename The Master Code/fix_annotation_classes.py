import json

ANNOTATIONS = ["data_stickers/train/annotations.coco.json", "data_stickers/valid/annotations.coco.json", "data_stickers/test/annotations.coco.json"]

def fix_annotation_classes(annotation_paths):

    annotation_files = []
    for annotation_path in annotation_paths:
        with open(annotation_path) as f:
            annotation_files.append(json.load(f))

    

    for annotation_file in annotation_files:
        annotation_file['categories'] = [
            {
                "id": 1,
                "name": "logo"
            },
            {
                "id": 2,
                "name": "sticker"
            }
        ]
    for annotation_file in annotation_files:
        for annotation in annotation_file['annotations']:
            if annotation['category_id'] == 1:
                annotation['category_id'] = 2
            elif annotation['category_id'] == 0:
                annotation['category_id'] = 1

    x = 5

    for annotation_file, annotation_path in zip(annotation_files, annotation_paths):
        with open(annotation_path, 'w') as f:
            json.dump(annotation_file, f, indent=4)


if __name__ == '__main__':
    fix_annotation_classes(ANNOTATIONS)
    