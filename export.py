from ultralytics import YOLO
pt_path = r"C:\Users\Lenovo\Desktop\ultralytics-main\runs\detect\train-11\weights\best.pt"
model = YOLO(pt_path)
model.export(
    format="onnx",
    imgsz=640,
    opset=16,
    simplify=True,
    device=0,
    half=False
)
print("导出成功，生成 best.onnx 在weights文件夹内")