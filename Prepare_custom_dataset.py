import glob
import io
from pathlib import Path
import os
import shutil
import uuid
import shutil
import random
import xml.etree.ElementTree as ET
import yaml



annotation_files = glob.glob('data/annotations' + '/**/*', recursive=True)#.shuffle()
image_files = glob.glob('data/images' + '/**/*.jpg', recursive=True)

# Shuffle the list so the datasets are randomly in the list
random.shuffle(annotation_files)

dog_races = list()


def process_single_item(annotation_file: str, datatype: str):
    try:

        if(not Path(annotation_file).is_file()):
            return
        print(annotation_file)
        filename = os.path.basename(annotation_file)
        #f = io.open(annotation_file, mode="r", encoding="utf-8")
        #text = f.read()
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        image_folder = root.find("folder").text
        image_name = root.find("filename").text + ".jpg"

        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        output_string = list()

        for item in root.findall("object"):
            race = item.find("name").text
            if not dog_races.__contains__(race):
                dog_races.append(race)
            race_index = dog_races.index(race)


            x1 = int(item.find("bndbox/xmin").text) # Top-Left
            y1 = int(item.find("bndbox/ymin").text) # Top-Left
            x2 = int(item.find("bndbox/xmax").text) # Bottom-Right
            y2 = int(item.find("bndbox/ymax").text) # Bottom-Right

            x_width = x2 - x1
            y_height = y2 - y1

            x_middle = (x1 + (x_width) / 2) / image_width
            y_middle = (y1 + (y_height) / 2) / image_height

            item_width = x_width / image_width
            item_height = y_height / image_height

            data_string = f"{race_index} {x_middle} {y_middle} {item_width} {item_height}"
            output_string.append(data_string)

        random_uuid = str(uuid.uuid4())
        output_filename = f"{filename}-{random_uuid}"
        output_file_path = f"dataset/{datatype}/labels/{output_filename}.txt"
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        output_file = io.open(output_file_path, mode="w+", encoding="utf-8")
        single_output_string = "\n".join(output_string)
        output_file.write(single_output_string)
        #for line in output_string:
        #    output_file.write(line + "\n")

        image_input_path = [image for image in image_files if image.__contains__(f"{image_folder}_") and image.__contains__(image_name) ][0]
        image_output_path = f"dataset/{datatype}/images/{output_filename}.jpg"
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
        shutil.copyfile(image_input_path, image_output_path)
    except Exception as e:
        print(f"Could not process {annotation_file} with the error {e}")

def process_group(items, datatype: str):
    for annotation_file in items:
        process_single_item(annotation_file, datatype)


training = annotation_files[:int(len(annotation_files)*0.7)] # 70%
validation = annotation_files[int(len(annotation_files)*0.7):int(len(annotation_files)*0.9)] # 20%
testing = annotation_files[int(len(annotation_files)*0.9):] # 10%

if os.path.exists('dataset') and os.path.isdir('dataset'):
    shutil.rmtree('dataset')

process_group(training, "train")
process_group(validation, "valid")
process_group(testing, "test")

# Config file for train yolo-model
config_yaml_dict = dict()
config_yaml_dict["nc"] = len(dog_races)
config_yaml_dict["names"] = dog_races
config_yaml_dict["train"] = "/dataset/train"
config_yaml_dict["val"] = "/dataset/valid"
config_yaml_dict["test"] = "/dataset/test"

#os.makedirs(os.path.dirname("config.yaml"), exist_ok=True)
with open("config.yaml", 'w') as file:
    yaml.dump(config_yaml_dict, file, default_flow_style=True)