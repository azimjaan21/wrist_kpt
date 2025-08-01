import json
from pathlib import Path

NUM_KEYPOINTS = 17  # Standard COCO

def coco_to_yolo_pose(coco_json_path, output_label_dir, image_dir):
    with open(coco_json_path) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]

    output_label_dir = Path(output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary: image_id -> list of annotation lines
    image_id_to_lines = {}

    for ann in annotations:
        image_id = ann["image_id"]
        image_info = images[image_id]
        file_name = image_info["file_name"]
        width, height = image_info["width"], image_info["height"]
        keypoints = ann["keypoints"]

        if len(keypoints) != NUM_KEYPOINTS * 3:
            continue  # skip invalid annotations

        # bbox normalized
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height

        # Normalize keypoints
        keypoints_norm = []
        for i in range(NUM_KEYPOINTS):
            x_kp = keypoints[i * 3] / width
            y_kp = keypoints[i * 3 + 1] / height
            v_kp = keypoints[i * 3 + 2]

            if v_kp == 0:
                x_kp, y_kp = 0.0, 0.0  # Unlabeled → set to 0

            keypoints_norm.extend([x_kp, y_kp, v_kp])

        # Format line
        line = [0, x_center, y_center, w, h] + keypoints_norm
        line_str = " ".join(map(str, line))

        if image_id not in image_id_to_lines:
            image_id_to_lines[image_id] = []

        image_id_to_lines[image_id].append(line_str)

    # Save label files
    for image_id, lines in image_id_to_lines.items():
        file_name = images[image_id]["file_name"]
        out_file = output_label_dir / (Path(file_name).stem + ".txt")
        with open(out_file, "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"Saved {len(image_id_to_lines)} label files to {output_label_dir}")

# === RUN ===
coco_to_yolo_pose(
    coco_json_path="coco2017/annotations/person_keypoints_train2017.json",
    output_label_dir="coco_17kpts/labels/train2017",
    image_dir="coco_17kpts/train2017"
)

coco_to_yolo_pose(
    coco_json_path="coco2017/annotations/person_keypoints_val2017.json",
    output_label_dir="coco_17kpts/labels/val2017",
    image_dir="coco_17kpts/val2017"
)
