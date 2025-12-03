#!/bin/bash

DEPLOY_DIR="yolo_arm_deploy"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Creating deployment package...${NC}"

rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR/{bin,lib,models}

echo "Copying executable..."
cp build/yolo_inference $DEPLOY_DIR/bin/

echo "Copying ONNX Runtime libraries..."
cp lib/onnxruntime-linux-aarch64-1.23.2/lib/libonnxruntime.so.1.23.2 $DEPLOY_DIR/lib/
cd $DEPLOY_DIR/lib/
ln -sf libonnxruntime.so.1.23.2 libonnxruntime.so.1
ln -sf libonnxruntime.so.1 libonnxruntime.so
cd - > /dev/null

echo "Copying YOLOv11 model..."
cp models/* $DEPLOY_DIR/models/

cat > $DEPLOY_DIR/run_inference.sh << 'EOF'
#!/bin/bash

# Set library path
SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
export LD_LIBRARY_PATH=$SCRIPT_DIR/lib:$LD_LIBRARY_PATH

# Run inference
if [ -f "$SCRIPT_DIR/lib/ld-linux-aarch64.so.1" ]; then
    $SCRIPT_DIR/lib/ld-linux-aarch64.so.1 --library-path $SCRIPT_DIR/lib $SCRIPT_DIR/bin/yolo_inference "$@"
else
    # Fallback to system loader
    $SCRIPT_DIR/bin/yolo_inference "$@"
fi
EOF

chmod +x $DEPLOY_DIR/run_inference.sh

cat > $DEPLOY_DIR/README.txt << 'EOF'
YOLOv11 ARM Deployment Package
===============================

Contents:
- bin/yolo_inference     : Main executable (ARM aarch64)
- lib/                   : ONNX Runtime libraries
- models/yolov11n.onnx   : YOLOv11 nano model
- run_inference.sh       : Helper script to run inference

Usage:
------
1. Transfer this directory to your ARM board

2. Run inference:
   ./run_inference.sh models/yolov11n.onnx input.jpg output.jpg

   Parameters:
   - First argument: path to ONNX model
   - Second argument: input image path
   - Third argument: output image path (with bounding boxes)

3. Alternative (manual):
   export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
   ./bin/yol o_inference models/yolov11n.onnx input.jpg output.jpg

Requirements on ARM board:
--------------------------
- ARM aarch64 architecture
- Linux kernel 3.7.0 or later
- GLIBC 2.17 or later
- libpthread, libdl (usually pre-installed)

Output:
-------
The program will:
- Load the ONNX model
- Process the input image
- Detect objects (80 COCO classes)
- Draw bounding boxes and labels
- Save result to output image
- Print detection results to console

For more information, see the main project README.md
EOF

echo -e "${GREEN}Deployment package created successfully!${NC}"
echo -e "${YELLOW}Package location: $DEPLOY_DIR${NC}"
echo ""
echo "Contents:"
ls -lh $DEPLOY_DIR
echo ""
echo "To deploy:"
echo "  scp -r $DEPLOY_DIR user@arm-board:~/"
echo "  ssh user@arm-board"
echo "  cd $DEPLOY_DIR"
echo "  ./run_inference.sh models/yolov11n.onnx input.jpg output.jpg"
