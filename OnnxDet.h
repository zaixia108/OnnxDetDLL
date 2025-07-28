#pragma once

#ifdef ONNXDET_EXPORTS
#define ONNXDET_API __declspec(dllexport)
#else
#define ONNXDET_API __declspec(dllimport)
#endif

// C风格导出接口
extern "C" {
    // 创建检测器实例
    ONNXDET_API void* CreateDetector();

    // 释放检测器实例
    ONNXDET_API void DestroyDetector(void* detector);

    // 初始化检测器
    ONNXDET_API bool InitDetector(void* detector, const char* model_path, float conf_threshold, float iou_threshold, bool use_dml);

    // 执行检测
    ONNXDET_API bool Detect(void* detector, unsigned char* image_data, int width, int height, int channels,
        float** out_boxes, float** out_scores, int** out_classes, int* out_count);

    // 释放结果内存
    ONNXDET_API void ReleaseResults(float* boxes, float* scores, int* classes);
}