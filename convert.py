from rknn.api import RKNN
import numpy as np
import cv2
rknn = RKNN(verbose=True)
rknn.config(
    target_platform='rv1103',
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]]
)
ret = rknn.load_onnx(
    model='best.onnx',
    input_size_list=[[1, 3, 640, 640]]
)
print(f"load_onnx ret: {ret}")
if ret == 0:
    ret = rknn.build(
        do_quantization=True,
        dataset='dataset.txt'
    )
    print(f"build ret: {ret}")
    if ret == 0:
        rknn.export_rknn('model_out_rv1103.rknn')
        print('✓ 转换成功')
    else:
        print('构建失败')
else:
    print('加载失败')
rknn.release()
