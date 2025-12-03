# YOLO ONNX Runtime ARM Deployment

Deploy YOLO object detection and instance segmentation models on ARM boards using ONNX Runtime.

## Project Structure

```
cv25yolo/
├── models/
│   ├── yolov11n.onnx          # YOLOv11 detection model
│   └── yolov11n-seg.onnx      # YOLOv11 segmentation model
├── lib/
│   ├── onnxruntime-linux-aarch64-1.23.2/  # ONNX Runtime for ARM
│   └── opencv/                 # OpenCV source code
├── src/
│   ├── main.cpp               # Main program
│   ├── yolo_detector.h/cpp    # YOLO detector implementation
│   └── utils.h/cpp            # Utility functions
├── tools/
│   └── build_opencv.sh        # OpenCV cross-compilation script
├── CMakeLists.txt             # CMake configuration
├── toolchain-aarch64.cmake    # ARM cross-compilation toolchain
├── build.sh                   # Build script
└── create_deploy_package.sh   # Deployment package creator
```

## Supported Model Types

### Detection Models
- **Files**: `yolov11n.onnx`, `yolov11s.onnx`, `yolov11m.onnx`, etc.
- **Output**: Bounding boxes + class labels
- **Output Shape**: `[1, 84, 8400]`
- **Visualization**: Colored bounding boxes with labels and confidence scores

### Segmentation Models
- **Files**: `yolov11n-seg.onnx`, `yolov11s-seg.onnx`, `yolov11m-seg.onnx`, etc.
- **Output**: Bounding boxes + class labels + instance segmentation masks
- **Output Shape**: 
  - Detection: `[1, 116, 8400]`
  - Mask Prototypes: `[1, 32, 160, 160]`
- **Visualization**: Semi-transparent colored masks (50% alpha) + bounding boxes with labels

**The same command works for both model types!** The application automatically detects the model format.

## Prerequisites

### Development Machine (x86_64)

1. **Cross-Compilation Toolchain**
   - Linaro aarch64 toolchain at: `/usr/local/linaro-aarch64-2020.09-gcc10.2-linux5.4`

2. **ONNX Runtime** (included)
   - Pre-built ARM64 version in `lib/onnxruntime-linux-aarch64-1.23.2`

3. **OpenCV**
   - Cross-compiled ARM64 version required
   - Use the provided script: `tools/build_opencv.sh`

### Target ARM Board

- **Architecture**: ARM aarch64 (64-bit)
- **OS**: Linux kernel 3.7.0 or later
- **GLIBC**: 2.17 or later
- **Runtime Libraries**: libpthread, libdl (usually pre-installed)

## Quick Start

### 1. Build OpenCV for ARM (if not already compiled)

```bash
cd tools
./build_opencv.sh
```

### 2. Build the Inference Application

```bash
./build.sh
```

After successful compilation, you'll have:
- `build/yolo_inference` - ARM executable

### 3. Create Deployment Package

```bash
./create_deploy_package.sh
```

This creates a `yolo_arm_deploy/` directory with everything needed for deployment.

### 4. Deploy to ARM Board

```bash
scp -r yolo_arm_deploy user@arm-board:~/

ssh user@arm-board
cd yolo_arm_deploy

# Run inference (detection model)
./run_inference.sh models/yolov11n.onnx input.jpg output.jpg

# Run inference (segmentation model) - same command!
./run_inference.sh models/yolov11n-seg.onnx input.jpg output_seg.jpg
```

## Usage

### Command Syntax

```bash
./yolo_inference <model_path> <input_image> <output_image>
```

**Parameters:**
- `<model_path>`: Path to YOLOv11 ONNX model file
- `<input_image>`: Input image path (supports jpg, png, etc.)
- `<output_image>`: Output image path with detection/segmentation results

### Examples

```bash
# Detection model
./yolo_inference yolov11n.onnx street.jpg result_det.jpg

# Segmentation model
./yolo_inference yolov11n-seg.onnx street.jpg result_seg.jpg

# With different model sizes
./yolo_inference yolov11s-seg.onnx people.jpg people_seg.jpg
```

## Output Examples

### Detection Model Output

```
Loading image: ./car.jpg
Image loaded: 720x480
Initializing YOLO detector with model: models/yolov11n.onnx
Detected DETECTION model
YOLODetector initialized successfully
Model type: Detection
Number of outputs: 1
Input name: images
Input shape: [1, 3, 640, 640]
Running inference...
Inference completed in 3593.41 ms
Found 3 objects:
  1. car (conf: 0.663401) bbox: [61, 95, 590, 282]
  2. car (conf: 0.43073) bbox: [580, 147, 136, 90]
  3. car (conf: 0.280718) bbox: [0, 147, 81, 66]
Result saved to: output.jpg
```

**Output Image**: Colored bounding boxes with class labels and confidence scores

### Segmentation Model Output

```
Loading image: ./car.jpg
Image loaded: 720x480
Initializing YOLO detector with model: models/yolo11n-seg.onnx
Detected SEGMENTATION model
YOLODetector initialized successfully
Model type: Segmentation
Number of outputs: 2
Input name: images
Input shape: [1, 3, 640, 640]
Running inference...
Inference completed in 5280.58 ms
Found 2 objects:
  1. car (conf: 0.656971) bbox: [580, 148, 136, 90]
  2. car (conf: 0.555717) bbox: [74, 94, 602, 284]
Result saved to: output.jpg
```

**Output Image**: Semi-transparent colored instance masks + bounding boxes with labels

## Configuration

### Detection Parameters

You can adjust detection parameters in `src/main.cpp`:

```cpp
YOLODetector detector(model_path, 
                      0.25f,  // conf_threshold: Confidence threshold
                      0.45f); // iou_threshold: NMS IoU threshold
```

**Parameter Guidelines:**
- **Confidence Threshold** (0.1-0.9): Higher = fewer but more confident detections
- **IOU Threshold** (0.3-0.7): Higher = fewer overlapping boxes

### Mask Transparency (Segmentation)

To adjust mask transparency, modify in `src/main.cpp`:

```cpp
// Current: 50% transparency
cv::multiply(colored_mask, mask_3c, weighted_mask, 0.5 / 255.0);

// Example: 30% transparency (more visible)
cv::multiply(colored_mask, mask_3c, weighted_mask, 0.3 / 255.0);
```

## Technical Details

### Model Preprocessing

1. Convert BGR to RGB
2. Resize to 640x640 (letterbox padding)
3. Normalize to [0, 1]
4. Convert HWC to CHW format
5. Add batch dimension

### Model Postprocessing

**Detection:**
1. Parse output tensor `[1, 84, 8400]`
   - 84 = 4 bbox coords + 80 class scores
   - 8400 anchor points
2. Apply confidence filtering
3. Non-Maximum Suppression (NMS)
4. Transform coordinates to original image scale

**Segmentation:**
1. Same as detection for bounding boxes
2. Extract 32-dimensional mask coefficients
3. Matrix multiplication with mask prototypes `[32, 160, 160]`
4. Sigmoid activation to generate probability map
5. Resize mask to original image size
6. Crop mask to bounding box region
7. Binarize with threshold 0.5
8. Apply colored overlay with transparency

### Automatic Model Detection

The application detects model type by checking:
1. **Number of outputs**: 1 output = Detection, 2 outputs = Segmentation
2. **First output dimensions**: 84 attributes = Detection, 116 attributes = Segmentation

```cpp
if (num_outputs_ == 2 && num_attributes == 116) {
    model_type_ = ModelType::SEGMENTATION;
} else {
    model_type_ = ModelType::DETECTION;
}
```

### Performance Optimization

1. **Multi-threading** - Adjust thread count in `yolo_detector.cpp`:
   ```cpp
   session_options_.SetIntraOpNumThreads(4);
   ```

2. **Graph Optimization** - Already enabled (extended optimization level):
   ```cpp
   session_options_.SetGraphOptimizationLevel(
       GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
   ```

3. **Model Selection**:
   - Use `yolov11n` for fastest inference (nano model)
   - Use `yolov11s` for balance of speed and accuracy (small model)
   - Use `yolov11m` for higher accuracy (medium model)

4. **Quantization** - Consider INT8 quantized models for 2-4x speedup


## 📦 Exporting Models

### Export Detection Model

```python
from ultralytics import YOLO

# Load pre-trained or custom model
model = YOLO('yolov11n.pt')

# Export to ONNX
model.export(format='onnx', simplify=True)
```

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [YOLOv11 Official Documentation](https://docs.ultralytics.com/models/yolo11/)
- [OpenCV Documentation](https://opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.

## Contributing

Issues and Pull Requests are welcome to improve this project!


