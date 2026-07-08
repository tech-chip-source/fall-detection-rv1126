from PIL import Image
import numpy as np
for i in range(100):
    img = Image.fromarray(
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    )
    img.save(f'calib_img{i+1}.jpg')
print('校准图片生成完成')
