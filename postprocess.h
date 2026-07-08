#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "postprocess.h"
#include "rknn_api.h"
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static float iou_calc(const box_rect_t* box1, const box_rect_t* box2) {
    int x1 = max(box1->left, box2->left);
    int y1 = max(box1->top, box2->top);
    int x2 = min(box1->right, box2->right);
    int y2 = min(box1->bottom, box2->bottom);
    int w = max(0, x2 - x1);
    int h = max(0, y2 - y1);
    float inter_area = w * h;
    float area1 = (box1->right - box1->left) * (box1->bottom - box1->top);
    float area2 = (box2->right - box2->left) * (box2->bottom - box2->top);
    return inter_area / (area1 + area2 - inter_area);
}
static void nms(object_detect_result_t* results, int* count) {
    for (int i = 0; i < *count; i++) {
        for (int j = i + 1; j < *count; j++) {
            if (iou_calc(&results[i].box, &results[j].box) > NMS_THRESH) {
                for (int k = j; k < *count - 1; k++) {
                    results[k] = results[k + 1];
                }
                (*count)--;
                j--;
            }
        }
    }
}

int post_process(rknn_output* outputs, int output_count, object_detect_result_list* result_list) {
    memset(result_list, 0, sizeof(*result_list));
    float* output = (float*)outputs[0].buf;
    int stride = 84;
    int num_boxes = outputs[0].size / sizeof(float) / stride;
    for (int i = 0; i < num_boxes; i++) {
        float* ptr = output + i * stride;
        float cls_conf = ptr[4];
        int cls_id = 0;
        float max_conf = 0;
        for (int j = 5; j < stride; j++) {
            if (ptr[j] > max_conf) {
                max_conf = ptr[j];
                cls_id = j - 5;
            }
        }
        float conf = cls_conf * max_conf;
        if (conf < BOX_THRESH) continue;
        float x = ptr[0];
        float y = ptr[1];
        float w = ptr[2];
        float h = ptr[3];
        int left = (int)(x - w / 2);
        int top = (int)(y - h / 2);
        int right = (int)(x + w / 2);
        int bottom = (int)(y + h / 2);
        object_detect_result_t* res = &result_list->results[result_list->count];
        res->cls_id = cls_id;
        res->prop = conf;
        res->box.left = left;
        res->box.top = top;
        res->box.right = right;
        res->box.bottom = bottom;
        result_list->count++;
    }
    nms(result_list->results, &result_list->count);
    return 0;
}
char* coco_cls_to_name(int cls_id) {
    static const char* names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
    if (cls_id < 0 || cls_id >= 80) return "null";
    return (char*)names[cls_id];
}
