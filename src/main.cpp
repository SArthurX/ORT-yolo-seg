#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolo_detector.h"
#include "utils.h"

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // Get class name and color
        std::string class_name = det.class_id < COCO_CLASSES.size() ? 
                                COCO_CLASSES[det.class_id] : "Unknown";
        cv::Scalar color = get_class_color(det.class_id);
        
        // Draw segmentation mask if available
        if (!det.mask.empty()) {
            // Create colored mask overlay
            cv::Mat colored_mask = cv::Mat::zeros(det.bbox.size(), CV_8UC3);
            colored_mask.setTo(color);
            
            // Apply mask as alpha channel
            cv::Mat mask_3c;
            cv::cvtColor(det.mask, mask_3c, cv::COLOR_GRAY2BGR);
            cv::Mat weighted_mask;
            cv::multiply(colored_mask, mask_3c, weighted_mask, 0.5 / 255.0);
            
            // Blend with original image
            cv::Rect safe_bbox = det.bbox & cv::Rect(0, 0, image.cols, image.rows);
            if (safe_bbox.area() > 0) {
                cv::Mat roi = image(safe_bbox);
                cv::add(roi, weighted_mask, roi);
            }
        }
        
        // Draw bounding box
        cv::rectangle(image, det.bbox, color, 2);
        
        // Prepare label text
        std::string label = class_name + " " + 
                          std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        
        // Get text size for background
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                             0.5, 1, &baseline);
        
        // Draw label background
        cv::Point label_pos(det.bbox.x, det.bbox.y - 5);
        if (label_pos.y < 0) label_pos.y = det.bbox.y + text_size.height + 5;
        
        cv::rectangle(image, 
                     cv::Point(label_pos.x, label_pos.y - text_size.height - baseline),
                     cv::Point(label_pos.x + text_size.width, label_pos.y + baseline),
                     color, cv::FILLED);
        
        // Draw label text
        cv::putText(image, label, label_pos, 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <input_image> <output_image>" << std::endl;
        std::cerr << "Example: ./yolo_inference yolov11n.onnx input.jpg output.jpg" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string input_image_path = argv[2];
    std::string output_image_path = argv[3];
    
    // Load image
    std::cout << "Loading image: " << input_image_path << std::endl;
    cv::Mat image = cv::imread(input_image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }
    std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
    
    // Initialize detector
    std::cout << "Initializing YOLO detector with model: " << model_path << std::endl;
    try {
        YOLODetector detector(model_path, 0.25f, 0.45f);
        
        // Run detection
        std::cout << "Running inference..." << std::endl;
        auto start = cv::getTickCount();
        std::vector<Detection> detections = detector.detect(image);
        auto end = cv::getTickCount();
        double inference_time = (end - start) / cv::getTickFrequency() * 1000.0;
        
        std::cout << "Inference completed in " << inference_time << " ms" << std::endl;
        std::cout << "Found " << detections.size() << " objects:" << std::endl;
        
        // Print detections
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            std::string class_name = det.class_id < COCO_CLASSES.size() ? 
                                    COCO_CLASSES[det.class_id] : "Unknown";
            std::cout << "  " << (i + 1) << ". " << class_name 
                     << " (conf: " << det.confidence << ")" 
                     << " bbox: [" << det.bbox.x << ", " << det.bbox.y << ", "
                     << det.bbox.width << ", " << det.bbox.height << "]"
                     << std::endl;
        }
        
        // Draw detections on image
        draw_detections(image, detections);
        
        // Save result
        cv::imwrite(output_image_path, image);
        std::cout << "Result saved to: " << output_image_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
