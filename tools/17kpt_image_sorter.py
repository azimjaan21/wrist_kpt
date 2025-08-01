import os
import shutil

def match_images(label_dir, coco_img_dir, output_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    label_ids = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    for img_id in label_ids:
        img_name = f"{img_id}.jpg"
        src = os.path.join(coco_img_dir, img_name)
        dst = os.path.join(output_img_dir, img_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"❌ Image not found: {src}")

# Paths
coco_train_img_dir = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\coco2017\train2017"
coco_val_img_dir = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\coco2017\val2017"

label_train_dir = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\coco_17kpts\labels\train2017"
label_val_dir = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\coco_17kpts\labels\val2017"

output_train_img_dir = "coco_17kpts/images/train2017"
output_val_img_dir = "coco_17kpts/images/val2017"

# Run for both train and val
match_images(label_train_dir, coco_train_img_dir, output_train_img_dir)
match_images(label_val_dir, coco_val_img_dir, output_val_img_dir)

print("✅ All matching images copied successfully.")
