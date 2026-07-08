# 先单独配置平台，不能写进build
from rknn import RKNN

rknn = RKNN(verbose=True)
# 平台配置单独一行
rknn.config(target_platform="rv1126b")
# 加载onnx
rknn.load_onnx(model="best.onnx")
# build只保留量化参数dataset
rknn.build(do_quantization=True, dataset="calib")
# 导出模型
rknn.export_rknn("out.rknn")
# 释放
rknn.release()
