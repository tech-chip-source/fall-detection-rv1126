#ifndef _YOLOV8_H_
#define _YOLOV8_H_
#include <stdint.h>
#include "rknn_api.h"
#include "postprocess.h"
#include "image_utils.h"
typedef struct {
    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
} rknn_app_context_t;
int init_yolov8_model(const char* model_path, rknn_app_context_t* app_ctx);
int inference_yolov8_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, object_detect_result_list* results);
void deinit_yolov8_model(rknn_app_context_t* app_ctx);
#endif
