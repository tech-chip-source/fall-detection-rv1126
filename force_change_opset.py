import onnx
model = onnx.load("rknn_convert/best_no_resize.onnx")
model.opset_import[0].version = 11
onnx.save(model, "rknn_convert/best_opset11_forced.onnx")
