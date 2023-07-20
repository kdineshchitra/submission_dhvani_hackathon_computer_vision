import os
import cv2
import shutil
import pathlib
import zipfile
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def get_annotations(tree):
    root = tree.getroot()
    size_el = root.find("size")
    width = int(size_el.find("width").text)
    height = int(size_el.find("height").text)
    objects = root.findall("object")
    object_bbox_dict = dict()
    for obj_idx, obj in enumerate(objects):
        class_name = obj.find("name").text
        if class_name not in CLASSES:
            continue
        bbox_el = obj.find("bndbox")
        xmin = int(bbox_el.find("xmin").text)
        ymin = int(bbox_el.find("ymin").text)
        xmax = int(bbox_el.find("xmax").text)
        ymax = int(bbox_el.find("ymax").text)
        object_bbox_dict[obj_idx] = {
            "class": class_name,
            "bbox": [xmin, ymin, xmax, ymax],
        }
    annotation_dict = {
        "height": height,
        "width": width,
        "num_objects": len(object_bbox_dict),
        "objects": object_bbox_dict,
    }
    return annotation_dict


def create_dataset(images, dataset="training"):
    root_data_dir = pathlib.Path("custom_dataset")
    image_dir = root_data_dir/"images"
    label_dir = root_data_dir/"labels"

    dataset_image_dir = image_dir/dataset
    dataset_label_dir = label_dir/dataset

    os.makedirs(dataset_image_dir, exist_ok=True)
    os.makedirs(dataset_label_dir, exist_ok=True)

    for image in images:
        img_np = cv2.imread(str(train_dir/image))
        height, width, _ = img_np.shape
        label = image[:image.rfind(".")] + ".txt"
        label_file = open(dataset_label_dir/label, "w")
        annotation = images_meta[image]
        for obj in annotation["objects"].values():
            label_id = CLASSES.index(obj["class"])
            xmin, ymin, xmax, ymax = obj["bbox"]
            cx = ((xmin + xmax) / 2) / width
            cy = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            label_file.write(" ".join(map(str, [label_id, cx, cy, w, h]))+"\n")
        label_file.close()
        shutil.copy(train_dir/image, dataset_image_dir)

data_archive_file = "archive.zip"
data_dir = pathlib.Path(".temp_vehicle_data")
os.makedirs(data_dir, exist_ok=True)

with zipfile.ZipFile(data_archive_file, "r") as zip_ref:
    zip_ref.extractall(data_dir)

train_dir = data_dir/"train"/"Final Train Dataset"
test_dir = data_dir/"test1"/"test"

CLASSES = ("bus", "car", "rickshaw", "three wheelers (CNG)")
NUM_CLASSES = len(CLASSES)

images_meta = dict()
for filepath in train_dir.rglob("*"):
    if filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        if filepath.suffix.lower() != ".xml":
            print(f"Unknown file format: {filepath}")
        continue
    xml_filepath = filepath.with_suffix(".xml")
    if not xml_filepath.exists():
        print(f"No annotation file for {filepath}")
        continue
    try:
        tree = ET.parse(xml_filepath)
    except Exception as e:
        print(f"Exception happend in {xml_filepath}\n\t- Exception: {e}")
        continue
    annotation_dict = get_annotations(tree)
    annotation_dict.update({"xml_filepath": xml_filepath.name})
    if annotation_dict["num_objects"] > 0:
        images_meta[filepath.name] = annotation_dict

train_images, val_images = train_test_split(list(images_meta.keys()), test_size=0.20)
create_dataset(train_images, dataset="training")
create_dataset(val_images, dataset="validation")
os.makedirs("custom_dataset/test_images/", exist_ok=True)
for test_image in test_dir.rglob("*"):
        shutil.copy(test_image, "custom_dataset/test_images/")

shutil.rmtree(data_dir)
