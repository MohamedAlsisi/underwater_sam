import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import shutil

def convert_coco_to_yolo_seg(json_path, image_dir, output_dir, class_mapping=None):
    """
    Convert COCO format annotations to YOLO segmentation format
    
    Args:
        json_path: Path to COCO json file
        image_dir: Directory containing the images
        output_dir: Directory to save YOLO format labels
        class_mapping: Optional dictionary mapping COCO class names to YOLO class indices
    """
    # Create output directories
    labels_dir = os.path.join(output_dir, 'labels')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Load COCO format annotations
    with open(json_path, 'r') as f:
        coco = json.load(f)
    
    # Create class mapping if not provided
    if class_mapping is None:
        categories = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}
    else:
        categories = {cat['id']: class_mapping[cat['name']] for cat in coco['categories']}
    
    # Create image id to filename mapping
    image_dict = {img['id']: img for img in coco['images']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Process each image
    for img_id, img_info in tqdm(image_dict.items(), desc='Converting annotations'):
        if img_id not in image_annotations:
            continue
            
        # Copy image to new directory
        src_path = os.path.join(image_dir, img_info['file_name'])
        dst_path = os.path.join(images_dir, img_info['file_name'])
        shutil.copy2(src_path, dst_path)
        
        # Get image dimensions
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Process annotations for this image
        txt_filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for ann in image_annotations[img_id]:
                # Get class id
                class_id = categories[ann['category_id']]
                
                # Get segmentation points and normalize them
                if isinstance(ann['segmentation'], dict):
                    # RLE format - skip for now (you might want to add RLE handling)
                    continue
                    
                for segment in ann['segmentation']:
                    # Convert segmentation to normalized coordinates
                    points = np.array(segment).reshape(-1, 2)
                    points[:, 0] = points[:, 0] / img_width
                    points[:, 1] = points[:, 1] / img_height
                    
                    # Format for YOLO segmentation:
                    # class_id x1 y1 x2 y2 ... xn yn
                    points_str = ' '.join([f'{p[0]:.6f} {p[1]:.6f}' for p in points])
                    f.write(f'{class_id} {points_str}\n')

def create_data_yaml(output_dir, train_path, val_path, test_path, nc, names):
    """Create data.yaml file for YOLO training"""
    yaml_content = f"""
path: {output_dir}  # dataset root dir
train: {train_path}  # train images (relative to 'path')
val: {val_path}  # val images (relative to 'path')
test: {test_path}  # test images (optional)

# Classes
nc: {nc}  # number of classes
names: {names}  # class names
    """
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())

def main():
    parser = argparse.ArgumentParser(description='Convert COCO format to YOLO segmentation format')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for YOLO format dataset')
    args = parser.parse_args()
    
    # Your class names
    class_names = ['wrecks/ruins', 'fish', 'reefs', 'aquatic plants', 'human divers', 'robots', 'sea-floor']
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    # Convert each split
    for split in ['train', 'val', 'test']:
        # split = 'val'
        print(f"\nProcessing {split} split...")
        json_path = os.path.join(args.data_dir, 'multi_class_annotations', f'multi_class_{split}_annotations.json')
        image_dir = os.path.join(args.data_dir, split)
        split_output_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        convert_coco_to_yolo_seg(json_path, image_dir, split_output_dir, class_mapping)
    
    # Create data.yaml
    create_data_yaml(
        args.output_dir,
        'train/images',  # Relative paths as used by YOLO
        'val/images',
        'test/images',
        len(class_names),
        class_names
    )

if __name__ == '__main__':
    main()