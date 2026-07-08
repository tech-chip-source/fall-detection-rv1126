import onnxruntime as ort
import numpy as np
import time
sess = ort.InferenceSession('./best.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
start = time.time()
outputs = sess.run([output_name], {input_name: input_data})
end = time.time()
print(f'推理耗时: {(end-start)*1000:.2f} ms')
print(f'输出形状: {outputs[0].shape}')