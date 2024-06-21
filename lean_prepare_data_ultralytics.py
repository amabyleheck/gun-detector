import os
import sys
import random
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

def write_txt(txt_file, data_list):
    with open(txt_file, 'w') as fd:
        for data in data_list:
            fd.write(f"{data}\n")

def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    width = float(root.find('size').find('width').text)
    height = float(root.find('size').find('height').text)

    bboxs = []
    for obj in root.findall('object'):
        bbox = [float(obj.find('bndbox').find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
        xc = (bbox[0] + bbox[2]) / 2 / width
        yc = (bbox[1] + bbox[3]) / 2 / height
        w = (bbox[2] - bbox[0]) / width
        h = (bbox[3] - bbox[1]) / height
        bboxs.append((xc, yc, w, h))
    return bboxs

def process_images(set_type, base_dir, output_root):
    set_type_lower = set_type.lower()  # Convert set type to lowercase
    jpeg_images_path = os.path.join(base_dir, set_type, 'JPEGImages')
    base_images = glob.glob(os.path.join(jpeg_images_path, '*.jpg'))
    random.shuffle(base_images)
    set_list = []

    labels_path = os.path.join(output_root, 'labels', set_type_lower)  # Use lowercase for directory names
    images_path = os.path.join(output_root, 'images', set_type_lower)  # Use lowercase for directory names
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    for img in tqdm(base_images):
        imgname = os.path.basename(img)
        imgbasename = imgname.rsplit('.', 1)[0]

        print(f"Processing {imgname}")
        anno_fn = os.path.join(base_dir, set_type, 'Annotations', f"{imgbasename}.xml")
        labels = ''.join([f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n" for bbox in parse_xml(anno_fn)])

        imgpath = os.path.abspath(img)
        des_fn = os.path.join(images_path, imgname)
        if not os.path.isabs(des_fn):
            des_fn = os.path.abspath(des_fn)

        sl_cmd = f"ln -s '{imgpath}' '{des_fn}'"
        print(f"Creating symlink: {sl_cmd}")
        os.system(sl_cmd)

        annofile = os.path.join(labels_path, f"{imgbasename}.txt")
        with open(annofile, 'wt') as fp:
            fp.write(labels)

        set_list.append(des_fn)

def create_data_yaml(output_root):
    os.makedirs(output_root, exist_ok=True)
    yaml_content = f"""
path: {output_root}  # Dataset root directory
train: images/train
val: images/val
test: images/test
# Classes
names:
  0: gun
"""
    with open(os.path.join(output_root, 'data.yaml'), 'w') as yaml_file:
        yaml_file.write(yaml_content.strip())

if __name__ == "__main__":
    base_dir = sys.argv[1]
    output_root = '~/datasets/real-time-gun-detection/ultralytics_converted/automatic_dataset'
    create_data_yaml(output_root)
    for set_type in ['Train', 'Test', 'Val']:
        process_images(set_type, base_dir, output_root)

