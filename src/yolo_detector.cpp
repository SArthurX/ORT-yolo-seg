#include "yolo_detector.h"
#include <iostream>
#include <algorithm>

YOLODetector::YOLODetector(const std::string& model_path, 
                           float conf_threshold, 
                           float iou_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "YOLODetector"),
      session_(nullptr),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold),
      input_width_(640),
      input_height_(640) {
    
    // Configure session options
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // Create session
    session_ = new Ort::Session(env_, model_path.c_str(), session_options_);
    
    // Get input info
    size_t num_input_nodes = session_->GetInputCount();
    if (num_input_nodes > 0) {
        auto input_name = session_->GetInputNameAllocated(0, allocator_);
        input_names_.push_back(std::string(input_name.get()));
        
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();
        
        // Update input dimensions if dynamic
        if (input_shape_.size() == 4) {
            input_height_ = input_shape_[2] > 0 ? input_shape_[2] : 640;
            input_width_ = input_shape_[3] > 0 ? input_shape_[3] : 640;
        }
    }
    
    // Get output info and detect model type
    num_outputs_ = session_->GetOutputCount();
    
    if (num_outputs_ > 0) {
        auto output_name = session_->GetOutputNameAllocated(0, allocator_);
        output_names_.push_back(std::string(output_name.get()));
        
        // Get first output shape to determine model type
        auto output_type_info = session_->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_shape = output_tensor_info.GetShape();
        
        // Detect model type based on output shape
        if (num_outputs_ == 2 && output_shape.size() >= 2) {
            // Segmentation model: output0=[1, 116, 8400], output1=[1, 32, 160, 160]
            int num_attributes = output_shape[1];
            if (num_attributes == 116 || num_attributes == 84 + MASK_DIM) {
                model_type_ = ModelType::SEGMENTATION;
                std::cout << "Detected SEGMENTATION model" << std::endl;
                
                // Get second output (mask prototypes)
                auto proto_name = session_->GetOutputNameAllocated(1, allocator_);
                output_names_.push_back(std::string(proto_name.get()));
            } else {
                model_type_ = ModelType::DETECTION;
                std::cout << "Detected DETECTION model" << std::endl;
            }
        } else {
            // Detection model: single output [1, 84, 8400]
            model_type_ = ModelType::DETECTION;
            std::cout << "Detected DETECTION model" << std::endl;
        }
    }
    
    // Prepare C-string pointers for ONNX Runtime API
    input_names_cstr_.clear();
    for (const auto& name : input_names_) {
        input_names_cstr_.push_back(name.c_str());
    }
    output_names_cstr_.clear();
    for (const auto& name : output_names_) {
        output_names_cstr_.push_back(name.c_str());
    }
    
    std::cout << "YOLODetector initialized successfully" << std::endl;
    std::cout << "Model type: " << (model_type_ == ModelType::SEGMENTATION ? "Segmentation" : "Detection") << std::endl;
    std::cout << "Number of outputs: " << num_outputs_ << std::endl;
    std::cout << "Input name: " << input_names_[0] << std::endl;
    std::cout << "Input shape: [" << input_shape_[0] << ", " 
              << input_shape_[1] << ", " << input_shape_[2] << ", " 
              << input_shape_[3] << "]" << std::endl;
}

YOLODetector::~YOLODetector() {
    if (session_) {
        delete session_;
    }
}

cv::Mat YOLODetector::preprocess(const cv::Mat& image) {
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    
    // Resize image
    cv::Mat resized;
    cv::resize(rgb_image, resized, cv::Size(input_width_, input_height_));
    
    // Convert to float and normalize to [0, 1]
    cv::Mat float_image;
    resized.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
    
    return float_image;
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
    // Store original image dimensions
    int orig_width = image.cols;
    int orig_height = image.rows;
    
    // Preprocess
    cv::Mat preprocessed = preprocess(image);
    
    // Prepare input tensor
    std::vector<float> input_tensor_values;
    input_tensor_values.resize(input_width_ * input_height_ * 3);
    
    // Convert HWC to CHW format
    std::vector<cv::Mat> channels(3);
    cv::split(preprocessed, channels);
    
    size_t channel_size = input_width_ * input_height_;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_tensor_values.data() + c * channel_size, 
                   channels[c].data, 
                   channel_size * sizeof(float));
    }
    
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_tensor_values.data(), 
        input_tensor_values.size(),
        input_shape.data(), 
        input_shape.size()
    );
    
    // Run inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr_.data(),
        &input_tensor,
        1,
        output_names_cstr_.data(),
        num_outputs_
    );
    
    // Handle different model types
    std::vector<Detection> detections;
    
    if (model_type_ == ModelType::SEGMENTATION) {
        // Get detection output [1, 116, 8400]
        float* det_data = output_tensors[0].GetTensorMutableData<float>();
        auto det_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t det_size = 1;
        for (auto dim : det_shape) det_size *= dim;
        std::vector<float> det_output(det_data, det_data + det_size);
        
        // Get mask prototypes [1, 32, 160, 160]
        float* proto_data = output_tensors[1].GetTensorMutableData<float>();
        auto proto_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        size_t proto_size = 1;
        for (auto dim : proto_shape) proto_size *= dim;
        std::vector<float> proto_output(proto_data, proto_data + proto_size);
        
        // Postprocess segmentation
        detections = postprocess_seg(det_output, proto_output, orig_width, orig_height);
    } else {
        // Detection model
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t output_size = 1;
        for (auto dim : output_shape) output_size *= dim;
        std::vector<float> output(output_data, output_data + output_size);
        
        // Postprocess detection
        detections = postprocess(output, orig_width, orig_height);
    }
    
    // Apply NMS
    detections = nms(detections, iou_threshold_);
    
    return detections;
}

std::vector<Detection> YOLODetector::postprocess(const std::vector<float>& output, 
                                                 int img_width, int img_height) {
    std::vector<Detection> detections;
    
    // YOLOv11 output format: [1, 84, 8400]
    // 84 = 4 (bbox) + 80 (classes)
    // 8400 = number of anchor points
    // Data layout: transposed, so we access as [attribute][anchor]
    
    const int num_anchors = 8400;
    const int num_attributes = 84;
    
    for (int i = 0; i < num_anchors; ++i) {
        // YOLOv11 output is transposed: shape is [1, 84, 8400]
        // So for anchor i, bbox data is at indices: [0*8400+i, 1*8400+i, 2*8400+i, 3*8400+i]
        float x_center = output[0 * num_anchors + i];
        float y_center = output[1 * num_anchors + i];
        float width = output[2 * num_anchors + i];
        float height = output[3 * num_anchors + i];
        
        // Find class with maximum confidence
        float max_class_score = 0.0f;
        int max_class_id = -1;
        
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float class_score = output[(4 + c) * num_anchors + i];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                max_class_id = c;
            }
        }
        
        // Filter by confidence threshold
        if (max_class_score < conf_threshold_) {
            continue;
        }
        
        // Convert from normalized coordinates to pixel coordinates
        // YOLOv11 outputs are already in pixel space relative to input size
        float x1 = (x_center - width / 2.0f) * img_width / input_width_;
        float y1 = (y_center - height / 2.0f) * img_height / input_height_;
        float x2 = (x_center + width / 2.0f) * img_width / input_width_;
        float y2 = (y_center + height / 2.0f) * img_height / input_height_;
        
        // Clamp to image boundaries
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_height)));
        
        Detection det;
        det.bbox = cv::Rect(
            static_cast<int>(x1), 
            static_cast<int>(y1),
            static_cast<int>(x2 - x1), 
            static_cast<int>(y2 - y1)
        );
        det.confidence = max_class_score;
        det.class_id = max_class_id;
        
        detections.push_back(det);
    }
    
    return detections;
}

// Segmentation-specific postprocessing
std::vector<Detection> YOLODetector::postprocess_seg(const std::vector<float>& det_output,
                                                      const std::vector<float>& proto_output,
                                                      int img_width, int img_height) {
    std::vector<Detection> detections;
    
    // YOLOv11-seg output format: [1, 116, 8400]
    // 116 = 4 (bbox) + 80 (classes) + 32 (mask coefficients)
    const int num_anchors = 8400;
    const int num_attributes = 116;
    
    for (int i = 0; i < num_anchors; ++i) {
        // Extract bbox coordinates (transposed format)
        float x_center = det_output[0 * num_anchors + i];
        float y_center = det_output[1 * num_anchors + i];
        float width = det_output[2 * num_anchors + i];
        float height = det_output[3 * num_anchors + i];
        
        // Find class with maximum confidence
        float max_class_score = 0.0f;
        int max_class_id = -1;
        
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float class_score = det_output[(4 + c) * num_anchors + i];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                max_class_id = c;
            }
        }
        
        // Filter by confidence threshold
        if (max_class_score < conf_threshold_) {
            continue;
        }
        
        // Convert from normalized coordinates to pixel coordinates
        float x1 = (x_center - width / 2.0f) * img_width / input_width_;
        float y1 = (y_center - height / 2.0f) * img_height / input_height_;
        float x2 = (x_center + width / 2.0f) * img_width / input_width_;
        float y2 = (y_center + height / 2.0f) * img_height / input_height_;
        
        // Clamp to image boundaries
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_height)));
        
        Detection det;
        det.bbox = cv::Rect(
            static_cast<int>(x1), 
            static_cast<int>(y1),
            static_cast<int>(x2 - x1), 
            static_cast<int>(y2 - y1)
        );
        det.confidence = max_class_score;
        det.class_id = max_class_id;
        
        // Extract mask coefficients
        std::vector<float> mask_coeffs(MASK_DIM);
        for (int j = 0; j < MASK_DIM; ++j) {
            mask_coeffs[j] = det_output[(4 + NUM_CLASSES + j) * num_anchors + i];
        }
        
        // Generate instance mask
        det.mask = generate_mask(mask_coeffs, proto_output, det.bbox, img_width, img_height);
        
        detections.push_back(det);
    }
    
    return detections;
}

// Generate instance mask from coefficients and prototypes
cv::Mat YOLODetector::generate_mask(const std::vector<float>& mask_coeffs,
                                   const std::vector<float>& prototypes,
                                   const cv::Rect& bbox,
                                   int img_width, int img_height) {
    // Prototype shape: [32, 160, 160]
    const int proto_h = 160;
    const int proto_w = 160;
    
    // Linear combination of prototypes
    cv::Mat mask = cv::Mat::zeros(proto_h, proto_w, CV_32F);
    
    for (int i = 0; i < MASK_DIM; ++i) {
        float coeff = mask_coeffs[i];
        for (int y = 0; y < proto_h; ++y) {
            for (int x = 0; x < proto_w; ++x) {
                int idx = i * proto_h * proto_w + y * proto_w + x;
                mask.at<float>(y, x) += coeff * prototypes[idx];
            }
        }
    }
    
    // Apply sigmoid activation
    cv::Mat mask_sigmoid;
    cv::exp(-mask, mask_sigmoid);
    mask_sigmoid = 1.0 / (1.0 + mask_sigmoid);
    
    // Resize to original image size
    cv::Mat mask_resized;
    cv::resize(mask_sigmoid, mask_resized, cv::Size(img_width, img_height));
    
    // Crop to bounding box
    cv::Rect safe_bbox = bbox & cv::Rect(0, 0, img_width, img_height);
    if (safe_bbox.area() > 0) {
        cv::Mat mask_cropped = mask_resized(safe_bbox).clone();
        
        // Threshold to binary mask
        cv::Mat binary_mask;
        cv::threshold(mask_cropped, binary_mask, 0.5, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);
        
        return binary_mask;
    }
    
    return cv::Mat();
}
