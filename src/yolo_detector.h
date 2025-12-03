#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "utils.h"

class YOLODetector {
public:
    YOLODetector(const std::string& model_path, 
                 float conf_threshold = 0.25f, 
                 float iou_threshold = 0.45f);
    ~YOLODetector();
    
    std::vector<Detection> detect(const cv::Mat& image);
    
    // Get model type
    ModelType getModelType() const { return model_type_; }
    
private:
    // Preprocessing
    cv::Mat preprocess(const cv::Mat& image);
    
    // Postprocessing
    std::vector<Detection> postprocess(const std::vector<float>& output, 
                                      int img_width, int img_height);
    
    // Segmentation-specific postprocessing
    std::vector<Detection> postprocess_seg(const std::vector<float>& det_output,
                                          const std::vector<float>& proto_output,
                                          int img_width, int img_height);
    
    // Generate instance mask from coefficients and prototypes
    cv::Mat generate_mask(const std::vector<float>& mask_coeffs,
                         const std::vector<float>& prototypes,
                         const cv::Rect& bbox,
                         int img_width, int img_height);
    
    // ONNX Runtime components
    Ort::Env env_;
    Ort::Session* session_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    // Model info
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
    std::vector<int64_t> input_shape_;
    
    // Model type
    ModelType model_type_;
    int num_outputs_;
    
    // Detection parameters
    float conf_threshold_;
    float iou_threshold_;
    int input_width_;
    int input_height_;
    
    // Constants
    static const int NUM_CLASSES = 80;  // COCO dataset
    static const int MASK_DIM = 32;     // Mask coefficients dimension
};

#endif // YOLO_DETECTOR_H
