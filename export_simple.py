from ultralytics import YOLO 

model = YOLO('runs/detect/experiment_stage2/weights/best.pt')

model.export(
    format = "tflite", 
    imgsz=640,
    int8=False,
    half=False, 
    nms=False
    #extra_args={"onnx2tf_option": "-onwdt"}
)



