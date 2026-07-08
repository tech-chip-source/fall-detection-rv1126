#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include "yolov8.h"
#include "postprocess.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <rknn_model> <image_path>\n", argv[0]);
        return -1;
    }
    const char* model_path = argv[1];
    const char* image_path = argv[2];
    // 初始化 RKNN 模型
    rknn_app_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(app_ctx));
    int ret = init_yolov8_model(model_path, &app_ctx);
    if (ret != 0) {
        printf("init_yolov8_model fail! ret=%d\n", ret);
        return -1;
    }
    // 读取图片
    image_buffer_t src_img;
    memset(&src_img, 0, sizeof(src_img));
    ret = read_image(image_path, &src_img);
    if (ret != 0) {
        printf("read_image fail! ret=%d\n", ret);
        return -1;
    }
    // 推理
    object_detect_result_list results;
    ret = inference_yolov8_model(&app_ctx, &src_img, &results);
    if (ret != 0) {
        printf("inference_yolov8_model fail! ret=%d\n", ret);
        return -1;
    }
    // 打印检测结果
    printf("检测到 %d 个目标:\n", results.count);
    for (int i = 0; i < results.count; i++) {
        printf("类别: %s, 置信度: %.2f, 坐标: (%d, %d, %d, %d)\n",
               coco_cls_to_name(results.results[i].cls_id),
               results.results[i].prop,
               results.results[i].box.left,
               results.results[i].box.top,
               results.results[i].box.right,
               results.results[i].box.bottom);
    }
    // 释放资源
    release_image(&src_img);
    deinit_yolov8_model(&app_ctx);
    return 0;
}
