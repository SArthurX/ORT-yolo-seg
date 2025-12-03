#include "utils.h"
#include <algorithm>

// COCO dataset class names (80 classes)
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

float calculate_iou(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    int intersection_width = std::max(0, x2 - x1);
    int intersection_height = std::max(0, y2 - y1);
    int intersection_area = intersection_width * intersection_height;
    
    int box1_area = box1.width * box1.height;
    int box2_area = box2.width * box2.height;
    int union_area = box1_area + box2_area - intersection_area;
    
    if (union_area == 0) return 0.0f;
    return static_cast<float>(intersection_area) / union_area;
}

std::vector<Detection> nms(std::vector<Detection>& detections, float iou_threshold) {
    // Sort detections by confidence (descending)
    std::sort(detections.begin(), detections.end(), 
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });
    
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            // Suppress boxes with same class and high IoU
            if (detections[i].class_id == detections[j].class_id) {
                float iou = calculate_iou(detections[i].bbox, detections[j].bbox);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return result;
}

cv::Scalar get_class_color(int class_id) {
    // Generate deterministic colors based on class_id
    int offset = class_id * 123457;
    int r = (offset * 37) % 256;
    int g = (offset * 59) % 256;
    int b = (offset * 97) % 256;
    return cv::Scalar(b, g, r);
}
