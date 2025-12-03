#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Model type
enum class ModelType {
    DETECTION,
    SEGMENTATION
};

// Detection structure
struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    cv::Mat mask;  // Segmentation mask (empty for detection models)
};

// NMS (Non-Maximum Suppression)
std::vector<Detection> nms(std::vector<Detection>& detections, float iou_threshold);

// Calculate IoU (Intersection over Union)
float calculate_iou(const cv::Rect& box1, const cv::Rect& box2);

// COCO class names (80 classes)
extern const std::vector<std::string> COCO_CLASSES;

// Generate random colors for visualization
cv::Scalar get_class_color(int class_id);

#endif // UTILS_H
