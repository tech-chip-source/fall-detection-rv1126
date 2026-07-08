import onnx
import onnxruntime as ort
import numpy as np
model = onnx.load("rknn_convert/best_fixed.onnx")
sess = ort.InferenceSession(model.SerializeToString())
onnx.save(model, "rknn_convert/best_final.onnx")
