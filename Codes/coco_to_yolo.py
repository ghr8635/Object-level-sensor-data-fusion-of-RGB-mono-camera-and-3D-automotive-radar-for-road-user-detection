# coco to yolo one folder


import json
import os


def coco2yolo_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            coco_file = os.path.join(input_folder, file_name)
            coco2yolo(coco_file, output_folder)


def coco2yolo(coco_file, output_folder):
    class_def = {
        "1": 0,
        "4": 1,
        "5": 2,
        "6": 3,
        "7": 4,

    }
    with open(coco_file, 'r') as f:
        data = json.load(f)
    image = data['image']
    annotations = data['annotations']
    categories = data['categories']

    # Extract image information
    image_id = image['id']
    file_name = image['file_name']
    width = image['width']
    height = image['height']

    # Write annotations to YOLO format txt file
    yolo_file_path = os.path.join(output_folder, file_name.replace('.png', '.txt'))
    with open(yolo_file_path, 'w') as yolo_file:
        for ann in annotations:
            category_id = ann['category_id']
            category = next((cat for cat in categories if cat['category_id'] == str(category_id)), None)
            if category is None:
                print(f"Category with id {category_id} not found.")
                continue

            class_name = category['name']

            bbox = ann['bbox']
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2
            width_norm = bbox[2] / width
            height_norm = bbox[3] / height

            yolo_line = f"{class_def[str(category_id)]} {x_center / width} {y_center / height} {width_norm} {height_norm}\n"
            yolo_file.write(yolo_line)


# Example usage:
input_folder = r'D:\AUF-SEM 2\Project\Python\pythonProject3\INFRA-3DRC_scene-04\camera_01\camera_01__annotation'
output_folder = r'D:\AUF-SEM 2\Project\Python\Output_folder\yolo_output'

coco2yolo_folder(input_folder, output_folder)

# coco to yolo all folder together
