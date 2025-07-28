#pragma once

#ifdef ONNXDET_EXPORTS
#define ONNXDET_API __declspec(dllexport)
#else
#define ONNXDET_API __declspec(dllimport)
#endif

// C��񵼳��ӿ�
extern "C" {
    // ���������ʵ��
    ONNXDET_API void* CreateDetector();

    // �ͷż����ʵ��
    ONNXDET_API void DestroyDetector(void* detector);

    // ��ʼ�������
    ONNXDET_API bool InitDetector(void* detector, const char* model_path, float conf_threshold, float iou_threshold, bool use_dml);

    // ִ�м��
    ONNXDET_API bool Detect(void* detector, unsigned char* image_data, int width, int height, int channels,
        float** out_boxes, float** out_scores, int** out_classes, int* out_count);

    // �ͷŽ���ڴ�
    ONNXDET_API void ReleaseResults(float* boxes, float* scores, int* classes);
}