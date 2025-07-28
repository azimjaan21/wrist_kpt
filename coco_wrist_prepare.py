import json
from pathlib import Path

# Constants: COCO keypoint indices
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10

def extract_wrist_keypoints(src_json_path, dst_json_path):
    with open(src_json_path, 'r') as f:
        data = json.load(f)

    # Update categories
    if 'categories' in data and len(data['categories']) > 0:
        data['categories'][0]['keypoints'] = ['left_wrist', 'right_wrist']
        data['categories'][0]['skeleton'] = []  # No skeleton for wrist-only

    new_annotations = []
    valid_image_ids = set()

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

    data['annotations'] = new_annotations
    data['images'] = [img for img in data['images'] if img['id'] in valid_image_ids]

    with open(dst_json_path, 'w') as f:
        json.dump(data, f)

    print(f"Processed {src_json_path.name}: {len(new_annotations)} annotations, {len(data['images'])} valid images.")

if __name__ == '__main__':
    annotation_dir = Path('coco/annotations')
    output_dir = Path('coco_wrist/annotations')
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train2017', 'val2017']:
        src_path = annotation_dir / f'person_keypoints_{split}.json'
        dst_path = output_dir / f'person_wrist_only_{split}.json'
        extract_wrist_keypoints(src_path, dst_path)
