import torch
from ultralytics import YOLO

def train_pose_scratch(
    model_yaml,
    data_yaml,
    exp_name,
    kpt_shape=[2, 3],
    epochs=120,
    batch=16,
    imgsz=640,
    patience=30,
    workers=5,
    device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training on {device}, from scratch")

    # Load from YAML, no pretrained weights
    model = YOLO(model_yaml)
    model.model.kpt_shape = kpt_shape

    # Ensure all layers trainable
    for name, param in model.model.named_parameters():
        param.requires_grad = True

    print(f"[INFO] Experiment: {exp_name}")
    model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        workers=workers,
        device=device,
        project='runs/pose',
        name=exp_name,
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        close_mosaic=10,
        warmup_epochs=3,
        lrf=0.01,
        cos_lr=True,  
        pretrained=False,
        save=True
    )

if __name__ == "__main__":
    config = {
        'model_yaml': r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\ultralytics\ultralytics\cfg\models\11\ab3.yaml',
        'data_yaml': 'coco_wrist/wrist_data.yaml',
        'exp_name': '#',
        'kpt_shape': [2, 3],
        'epochs': 70,
        'batch': 16,
        'imgsz': 640,
        'patience': 30,
        'workers': 5,
    }
    train_pose_scratch(**config)
