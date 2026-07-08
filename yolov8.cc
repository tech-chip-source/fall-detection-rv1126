#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "yolov8.h"
#include "postprocess.h"
#include "rknn_api.h"
int init_yolov8_model(const char* model_path, rknn_app_context_t* app_ctx) {
    int ret;
    // 读取模型文件
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("打开模型文件失败: %s\n", model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    size_t model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* model_data = (uint8_t*)malloc(model_size);
    fread(model_data, 1, model_size, fp);
    fclose(fp);
    // 初始化 RKNN
    ret = rknn_init(&app_ctx->ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("rknn_init 失败! ret=%d\n", ret);
        return -1;
    }
    // 获取输入输出信息
    ret = rknn_query(app_ctx->ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));
    if (ret < 0) {
        printf("rknn_query 获取输入输出数量失败! ret=%d\n", ret);
        return -1;
    }
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));

    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->input_attrs[i].index = i;
        rknn_query(app_ctx->ctx, RKNN_QUERY_INPUT_ATTR, &app_ctx->input_attrs[i], sizeof(rknn_tensor_attr));
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->output_attrs[i].index = i;
        rknn_query(app_ctx->ctx, RKNN_QUERY_OUTPUT_ATTR, &app_ctx->output_attrs[i], sizeof(rknn_tensor_attr));
    }
    app_ctx->model_width = app_ctx->input_attrs[0].dims[2];
    app_ctx->model_height = app_ctx->input_attrs[0].dims[1];
    app_ctx->model_channel = app_ctx->input_attrs[0].dims[3];
    printf("模型加载成功: 输入尺寸 %dx%d\n", app_ctx->model_width, app_ctx->model_height);
    return 0;
}
int inference_yolov8_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, object_detect_result_list* results) {
    int ret;
    // 预处理图像
    image_buffer_t resized_img;
    memset(&resized_img, 0, sizeof(resized_img));
    resized_img.width = app_ctx->model_width;
    resized_img.height = app_ctx->model_height;
    resized_img.size = app_ctx->model_width * app_ctx->model_height * 3;
    resized_img.virt_addr = (uint8_t*)malloc(resized_img.size);
    ret = resize_image(src_img, &resized_img);
    if (ret != 0) {
        printf("resize_image 失败! ret=%d\n", ret);
        return -1;
    }
    // 设置输入
    rknn_input inputs[1];
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = resized_img.size;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = resized_img.virt_addr;
    ret = rknn_inputs_set(app_ctx->ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set 失败! ret=%d\n", ret);
        free(resized_img.virt_addr);
        return -1;
    }
    // 执行推理
    struct timeval start, end;
    gettimeofday(&start, NULL);
    ret = rknn_run(app_ctx->ctx, NULL);
    if (ret < 0) {
        printf("rknn_run 失败! ret=%d\n", ret);
        free(resized_img.virt_addr);
        return -1;
    }
    gettimeofday(&end, NULL);
    printf("推理耗时: %.2f ms\n", (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0);
    // 获取输出
    rknn_output outputs[app_ctx->io_num.n_output];
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(app_ctx->ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get 失败! ret=%d\n", ret);
        free(resized_img.virt_addr);
        return -1;
    }
    // 后处理
    post_process(outputs, app_ctx->io_num.n_output, results);
    // 释放资源
    rknn_outputs_release(app_ctx->ctx, app_ctx->io_num.n_output, outputs);
    free(resized_img.virt_addr);
    return 0;
}
void deinit_yolov8_model(rknn_app_context_t* app_ctx) {
    if (app_ctx->ctx) {
        rknn_destroy(app_ctx->ctx);
    }
    if (app_ctx->input_attrs) {
        free(app_ctx->input_attrs);
    }
    if (app_ctx->output_attrs) {
        free(app_ctx->output_attrs);
    }
}
