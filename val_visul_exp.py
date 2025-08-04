from ultralytics import YOLO


model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\runs\pose\base_yolo11s_pose\weights\best.pt")  

image_folder = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\coco_wrist\images\val2017'  

results = model.predict(
    source=image_folder,
    imgsz=640,              
    save=True,              
    show=False,            
    project='visual_wrist_valids/',
    name='trans_laern_base_pose',   
    exist_ok=True          
)
