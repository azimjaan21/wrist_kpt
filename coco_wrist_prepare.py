import json
from pathlib import Path
import shutil

# Constants: COCO keypoint indices
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10

def extract_wrist_keypoints(src_json_path, dst_json_path, src_img_dir, dst_img_dir):
    with open(src_json_path, 'r') as f:
        data = json.load(f)

    # Update categories
    if 'categories' in data and len(data['categories']) > 0:
        data['categories'][0]['keypoints'] = ['left_wrist', 'right_wrist']
        data['categories'][0]['skeleton'] = []

    new_annotations = []
    valid_image_ids = set()
    image_id_to_file = {}

    for img in data['images']:
        image_id_to_file[img['id']] = img['file_name']

    for ann in data['annotations']:
        if 'keypoints' not in ann:
            continue
        kps = ann['keypoints']
        left = kps[LEFT_WRIST_IDX*3:LEFT_WRIST_IDX*3+3]
        right = kps[RIGHT_WRIST_IDX*3:RIGHT_WRIST_IDX*3+3]
        new_kps = left + right

        # Count visible wrists
        num_visible = int(left[2] > 0) + int(right[2] > 0)
        if num_visible > 0:
            ann['keypoints'] = new_kps
            ann['num_keypoints'] = num_visible
            new_annotations.append(ann)
            valid_image_ids.add(ann['image_id'])

    # Filter images and copy them
    filtered_images = []
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    for img in data['images']:
        if img['id'] in valid_image_ids:
            filtered_images.append(img)
            src_img_path = src_img_dir / img['file_name']
            dst_img_path = dst_img_dir / img['file_name']
            if not dst_img_path.exists():  # avoid duplicate copy
                shutil.copy2(src_img_path, dst_img_path)

    # Save filtered data
    data['annotations'] = new_annotations
    data['images'] = filtered_images

    with open(dst_json_path, 'w') as f:
        json.dump(data, f)

    print(f"✅ Processed {src_json_path.name}: {len(new_annotations)} annotations, {len(filtered_images)} images.")

if __name__ == '__main__':
    base_dir = Path('coco2017')
    annotation_dir = base_dir / 'annotations'
    output_dir = Path('coco_wrist')
    output_ann_dir = output_dir / 'annotations'
    output_ann_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train2017', 'val2017']:
        src_json = annotation_dir / f'person_keypoints_{split}.json'
        dst_json = output_ann_dir / f'person_wrist_only_{split}.json'

        src_img_dir = base_dir / split
        dst_img_dir = output_dir / split

        extract_wrist_keypoints(src_json, dst_json, src_img_dir, dst_img_dir)
