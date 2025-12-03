#!/bin/bash

# Build script for YOLOv11 inference on ARM

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building YOLOv11 Inference for ARM aarch64${NC}"

# Set variables
BUILD_DIR="build"
TOOLCHAIN_FILE="tools/toolchain-aarch64.cmake"

# Check if toolchain file exists
if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo -e "${RED}Error: Toolchain file not found: $TOOLCHAIN_FILE${NC}"
    exit 1
fi

# Create build directory
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Removing existing build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring CMake...${NC}"
cmake -DCMAKE_TOOLCHAIN_FILE=../$TOOLCHAIN_FILE \
      -DCMAKE_BUILD_TYPE=Release \
      ..

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

# Build
echo -e "${GREEN}Building...${NC}"
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}Executable: ${BUILD_DIR}/yolo_inference${NC}"

# Check the binary architecture
echo -e "\n${GREEN}Verifying binary architecture:${NC}"
file yolo_inference

echo -e "\n${YELLOW}To deploy to ARM board:${NC}"
echo "1. Copy the following files to your ARM board:"
echo "   - build/yolo_inference"
echo "   - models/yolov11n.onnx"
echo "   - lib/onnxruntime-linux-aarch64-1.23.2/lib/libonnxruntime.so*"
echo ""
echo "2. On the ARM board, set LD_LIBRARY_PATH:"
echo "   export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:\$LD_LIBRARY_PATH"
echo ""
echo "3. Run inference:"
echo "   ./yolo_inference yolov11n.onnx input.jpg output.jpg"
