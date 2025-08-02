#include "onnxdet.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// 这里包含您的OnnxDet类的完整定义
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>

class OnnxDet {
public:
    OnnxDet() = default;

    void Init(const std::string& path, double confThres = 0.3, double iouThres = 0.5, bool use_dml = true) {
        conf_threshold = confThres;
        iou_threshold = iouThres;

        // 初始化ONNX运行时
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (use_dml) {
            session_options.AppendExecutionProvider("DmlExecutionProvider");
        }
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Detect");

        std::wstring wpath(path.begin(), path.end());
        session = Ort::Session(env, wpath.c_str(), session_options);

        GetInputDetails();
        GetOutputDetails();
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> operator()(const cv::Mat& image) {
        return DetectObjects(image);
    }

private:
    Ort::Env env{ nullptr };
    Ort::Session session{ nullptr };
    double conf_threshold = 0.3;
    double iou_threshold = 0.5;

    std::vector<std::string> input_names;
    std::vector<int64_t> input_shape;
    int input_height = 0;
    int input_width = 0;

    std::vector<std::string> output_names;
    std::vector<int64_t> output_shape;

    int img_height = 0;
    int img_width = 0;

    void GetInputDetails() {
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session.GetInputCount();
        input_names.clear();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_names.push_back(name.get());
        }
        auto type_info = session.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shape = tensor_info.GetShape();
        if (input_shape.size() >= 4) {
            input_height = static_cast<int>(input_shape[2]);
            input_width = static_cast<int>(input_shape[3]);
        }
    }

    void GetOutputDetails() {
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_outputs = session.GetOutputCount();
        output_names.clear();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_names.push_back(name.get());
        }

        // 获取输出形状信息（用于调试）
        if (session.GetOutputCount() > 0) {
            auto type_info = session.GetOutputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shape = tensor_info.GetShape();
            /*std::cout << "模型输出形状: [";
            for (size_t i = 0; i < output_shape.size(); ++i) {
                std::cout << output_shape[i] << (i < output_shape.size() - 1 ? ", " : "");
            }
            std::cout << "]" << std::endl;*/
        }
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> DetectObjects(const cv::Mat& image) {
        std::vector<float> input_tensor;
        float ratio;
        std::tie(input_tensor, ratio) = PrepareInput(image);

        // 构造输入张量
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor.data(), input_tensor.size(),
            input_shape.data(), input_shape.size()
        );

        // 准备输入输出名称
        std::vector<const char*> input_names_cstr{ input_names[0].c_str() };
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names) {
            output_names_cstr.push_back(name.c_str());
        }

        // 执行推理
        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names_cstr.data(), &input_tensor_ort, 1,
            output_names_cstr.data(), output_names_cstr.size()
        );

        // 处理输出结果
        return ProcessOutput(output_tensors, ratio);
    }

    std::pair<std::vector<float>, float> PrepareInput(const cv::Mat& image) {
        img_height = image.rows;
        img_width = image.cols;

        // BGR转RGB
        cv::Mat rgb_img;
        cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);

        // 等比例缩放并填充
        cv::Mat resized_img;
        float ratio = RatioResize(rgb_img, resized_img);

        // 归一化到0-1
        std::vector<float> input_tensor;
        input_tensor.resize(3 * input_height * input_width);

        // HWC -> CHW (NCHW)
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < input_height; ++h) {
                for (int w = 0; w < input_width; ++w) {
                    float pixel = resized_img.at<cv::Vec3b>(h, w)[c] / 255.0f;
                    input_tensor[c * input_height * input_width + h * input_width + w] = pixel;
                }
            }
        }

        return { input_tensor, ratio };
    }

    float RatioResize(const cv::Mat& src, cv::Mat& dst, int color = 114) {
        dst = cv::Mat(input_height, input_width, CV_8UC3, cv::Scalar(color, color, color));

        // 计算缩放比例，保持长宽比
        float r = std::min(static_cast<float>(input_height) / src.rows, static_cast<float>(input_width) / src.cols);
        int new_unpad_w = static_cast<int>(std::round(src.cols * r));
        int new_unpad_h = static_cast<int>(std::round(src.rows * r));

        // 缩放图像
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

        // 填充
        resized.copyTo(dst(cv::Rect(0, 0, new_unpad_w, new_unpad_h)));

        // 返回1/r用于坐标还原
        return 1.0f / r;
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>
        ProcessOutput(const std::vector<Ort::Value>& output_tensors, float ratio) {
        // 获取输出张量数据
        const float* output_data = output_tensors[0].GetTensorData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        /*std::cout << "原始输出形状: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i] << (i < output_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;*/

        // 假定输出为 [1, n_ch, n_anchors]，如 [1, 84, 8400]
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        int batch_size = output_shape[0];
        int n_ch = output_shape[1];      // 通道数（类别数+5）
        int n_anchors = output_shape[2]; // 锚点数量

        int num_classes = n_ch - 4; // 减去xywh和obj_conf
        /*std::cout << "处理输出: 锚点数=" << n_anchors << ", 通道数=" << n_ch
            << ", 类别数=" << num_classes << std::endl;*/

        // 针对 [1, n_ch, n_anchors]，channel在第2维，anchor在第3维
        for (int i = 0; i < n_anchors; ++i) {
            std::vector<float> anchor_data(n_ch);
            for (int j = 0; j < n_ch; ++j) {
                // 先遍历channel，再anchor
                anchor_data[j] = output_data[j * n_anchors + i];
            }

            // 提取边界框坐标和置信度
            float x = anchor_data[0];
            float y = anchor_data[1];
            float w = anchor_data[2];
            float h = anchor_data[3];
            //float obj_conf = anchor_data[4];

            // 找出最高类别分数
            float max_class_score = 0;
            int max_class_id = 0;
            for (int j = 0; j < num_classes - 1; ++j) {
                float class_score = anchor_data[4 + j];
                if (class_score > max_class_score) {
                    max_class_score = class_score;
                    max_class_id = j;
                }
            }


            // 计算最终置信度
            float confidence = max_class_score;

            // 过滤低置信度的边界框
            if (confidence < conf_threshold) {
                continue;
            }

            // 应用比例因子
            x *= ratio;
            y *= ratio;
            w *= ratio;
            h *= ratio;

            // 转换为左上右下坐标
            float x1 = x - w / 2;
            float y1 = y - h / 2;
            float x2 = x + w / 2;
            float y2 = y + h / 2;

            // 保存结果
            boxes.push_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)
            ));
            confidences.push_back(confidence);
            class_ids.push_back(max_class_id);
        }

        // 应用NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

        // 提取NMS后的结果
        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;

        for (int idx : indices) {
            nms_boxes.push_back(boxes[idx]);
            nms_scores.push_back(confidences[idx]);
            nms_class_ids.push_back(class_ids[idx]);
        }

        //std::cout << "检测到 " << nms_boxes.size() << " 个物体" << std::endl;

        return { nms_boxes, nms_scores, nms_class_ids };
    }
};


// 导出函数实现
extern "C" {
    ONNXDET_API void* CreateDetector() {
        try {
            return new OnnxDet();
        }
        catch (const std::exception& e) {
            std::cerr << "Error creating detector: " << e.what() << std::endl;
            return nullptr;
        }
    }

    ONNXDET_API void DestroyDetector(void* detector) {
        if (detector) {
            delete static_cast<OnnxDet*>(detector);
        }
    }

    ONNXDET_API bool InitDetector(void* detector, const char* model_path, float conf_threshold, float iou_threshold, bool use_dml = true) {
        if (!detector || !model_path) return false;

        try {
            static_cast<OnnxDet*>(detector)->Init(model_path, conf_threshold, iou_threshold, use_dml);
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Initialization error: " << e.what() << std::endl;
            return false;
        }
    }

    ONNXDET_API bool Detect(void* detector, unsigned char* image_data, int width, int height, int channels,
        float** out_boxes, float** out_scores, int** out_classes, int* out_count) {
        if (!detector || !image_data || !out_boxes || !out_scores || !out_classes || !out_count) return false;

        try {
            cv::Mat image(height, width, CV_8UC3, image_data);
            auto results = (*static_cast<OnnxDet*>(detector))(image);

            const auto& boxes = std::get<0>(results);
            const auto& scores = std::get<1>(results);
            const auto& classes = std::get<2>(results);

            *out_count = static_cast<int>(boxes.size());
            if (*out_count == 0) {
                *out_boxes = nullptr;
                *out_scores = nullptr;
                *out_classes = nullptr;
                return true;
            }

            // 为结果分配内存
            *out_boxes = (float*)malloc(boxes.size() * 4 * sizeof(float));
            *out_scores = (float*)malloc(scores.size() * sizeof(float));
            *out_classes = (int*)malloc(classes.size() * sizeof(int));

            // 复制检测结果 (x1, y1, x2, y2格式)
            for (size_t i = 0; i < boxes.size(); i++) {
                (*out_boxes)[i * 4] = static_cast<float>(boxes[i].x);
                (*out_boxes)[i * 4 + 1] = static_cast<float>(boxes[i].y);
                (*out_boxes)[i * 4 + 2] = static_cast<float>(boxes[i].x + boxes[i].width);
                (*out_boxes)[i * 4 + 3] = static_cast<float>(boxes[i].y + boxes[i].height);
                (*out_scores)[i] = scores[i];
                (*out_classes)[i] = classes[i];
            }

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Detection error: " << e.what() << std::endl;
            *out_count = 0;
            return false;
        }
    }

    ONNXDET_API void ReleaseResults(float* boxes, float* scores, int* classes) {
        if (boxes) free(boxes);
        if (scores) free(scores);
        if (classes) free(classes);
    }
}
