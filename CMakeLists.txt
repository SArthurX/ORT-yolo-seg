cmake_minimum_required(VERSION 3.16)
project(yolo_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static linking flags to avoid GLIBC version issues
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT ${CMAKE_SOURCE_DIR}/lib/onnxruntime-linux-aarch64-1.23.2)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)

# OpenCV paths
set(OPENCV_INSTALL_DIR ${CMAKE_SOURCE_DIR}/lib/opencv/build/install)
set(OPENCV_INCLUDE_DIR ${OPENCV_INSTALL_DIR}/include/opencv4)
set(OPENCV_LIB_DIR ${OPENCV_INSTALL_DIR}/lib)

# Include directories
include_directories(
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

# Link directories
link_directories(
    ${ONNXRUNTIME_LIB_DIR}
    ${OPENCV_LIB_DIR}   
)

# Source files
set(SOURCES
    src/main.cpp
    src/yolo_detector.cpp
    src/utils.cpp
)

# Create executable
add_executable(yolo_inference ${SOURCES})

# Link libraries
set(OPENCV_3RDPARTY_DIR ${OPENCV_LIB_DIR}/opencv4/3rdparty)
target_link_libraries(yolo_inference
    onnxruntime
    ${OPENCV_LIB_DIR}/libopencv_imgcodecs.a
    ${OPENCV_LIB_DIR}/libopencv_imgproc.a
    ${OPENCV_LIB_DIR}/libopencv_core.a
    ${OPENCV_3RDPARTY_DIR}/liblibjpeg-turbo.a
    ${OPENCV_3RDPARTY_DIR}/liblibpng.a
    ${OPENCV_3RDPARTY_DIR}/liblibtiff.a
    ${OPENCV_3RDPARTY_DIR}/liblibwebp.a
    ${OPENCV_3RDPARTY_DIR}/liblibopenjp2.a
    ${OPENCV_3RDPARTY_DIR}/libzlib.a
    dl
    pthread
)

# Set RPATH for runtime library search
set_target_properties(yolo_inference PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)
