import json
from pathlib import Path

LEFT_WRIST_IDX = 0
RIGHT_WRIST_IDX = 1

def coco_to_yolo_pose(coco_json_path, output_label_dir, image_dir):
    with open(coco_json_path) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]

    output_label_dir = Path(output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary: image_id -> list of annotation lines (strings)
    image_id_to_lines = {}

    for ann in annotations:
        image_id = ann["image_id"]
        image_info = images[image_id]
        file_name = image_info["file_name"]
        width, height = image_info["width"], image_info["height"]
        keypoints = ann["keypoints"]

        # Check wrist visibility: v > 0 means labeled
        left_vis = keypoints[LEFT_WRIST_IDX * 3 + 2] > 0
        right_vis = keypoints[RIGHT_WRIST_IDX * 3 + 2] > 0

        if not (left_vis or right_vis):
            continue  # skip if neither wrist is visible/labeled

        # bbox normalized
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height

        # Prepare keypoints: zero out coordinates if visibility == 0
        def get_kp(idx):
            x = keypoints[idx * 3] / width
            y = keypoints[idx * 3 + 1] / height
            v = keypoints[idx * 3 + 2]
            if v == 0:
                x, y = 0.0, 0.0
            return [x, y, v]

        left_kp = get_kp(LEFT_WRIST_IDX)
        right_kp = get_kp(RIGHT_WRIST_IDX)

        line = [0, x_center, y_center, w, h] + left_kp + right_kp
        line_str = " ".join(map(str, line))

        if image_id not in image_id_to_lines:
            image_id_to_lines[image_id] = []

        image_id_to_lines[image_id].append(line_str)

    # Save all label files (one file per image, multiple persons each line)
    for image_id, lines in image_id_to_lines.items():
        file_name = images[image_id]["file_name"]
        out_file = output_label_dir / (Path(file_name).stem + ".txt")
        with open(out_file, "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"Saved {len(image_id_to_lines)} label files to {output_label_dir}")

# === RUN ===
coco_to_yolo_pose(
    coco_json_path="coco_wrist/annotations/person_wrist_only_train2017.json",
    output_label_dir="coco_wrist/labels/train2017",
    image_dir="coco_wrist/train2017"
)

coco_to_yolo_pose(
    coco_json_path="coco_wrist/annotations/person_wrist_only_val2017.json",
    output_label_dir="coco_wrist/labels/val2017",
    image_dir="coco_wrist/val2017"
)
