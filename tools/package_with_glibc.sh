#!/bin/bash

# Script to package GLIBC libraries with the deployment

#if you have error like this:
#./yolo_inference: /lib/libm.so.6: version `GLIBC_2.29' not found
#./yolo_inference: /usr/lib/libstdc++.so.6: version `GLIBCXX_3.4.26' not found

TOOLCHAIN=/usr/local/linaro-aarch64-2020.09-gcc10.2-linux5.4
DEPLOY_DIR="../yolo_arm_deploy"
GLIBC_SRC="${TOOLCHAIN}/aarch64-linux-gnu/libc/lib"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}=========================${NC}"
echo -e "${YELLOW}GLIBC Library Packaging${NC}"
echo -e "${YELLOW}=========================${NC}"
echo ""
echo -e "${RED}⚠️  WARNING: This bundles system libraries from the toolchain${NC}"
echo -e "${RED}   This may cause compatibility issues on some systems.${NC}"
echo -e "${RED}   Only use if you understand the risks!${NC}"
echo ""
read -p "Do you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${GREEN}Checking GLIBC source location...${NC}"

if [ ! -d "$GLIBC_SRC" ]; then
    echo -e "${RED}Error: GLIBC library directory not found at: $GLIBC_SRC${NC}"
    echo "Please check your toolchain installation."
    exit 1
fi

echo -e "${GREEN}Creating lib directory in deployment package...${NC}"
mkdir -p "${DEPLOY_DIR}/lib"

echo -e "${GREEN}Copying GLIBC libraries...${NC}"

# Core GLIBC libraries
LIBS_TO_COPY=(
    "libc.so.6"
    "libm.so.6"
    "libpthread.so.0"
    "libdl.so.2"
    "ld-linux-aarch64.so.1"
)

for lib in "${LIBS_TO_COPY[@]}"; do
    if [ -f "${GLIBC_SRC}/${lib}" ]; then
        echo "  Copying ${lib}..."
        cp -L "${GLIBC_SRC}/${lib}" "${DEPLOY_DIR}/lib/" 2>/dev/null || \
        cp "${GLIBC_SRC}/${lib}" "${DEPLOY_DIR}/lib/"
    else
        echo -e "  ${YELLOW}⚠️  ${lib} not found, skipping...${NC}"
    fi
done

echo ""
echo -e "${GREEN}✓ GLIBC libraries packaged successfully!${NC}"
echo ""
echo -e "${YELLOW}Deployment package contents:${NC}"
ls -lh ${DEPLOY_DIR}/lib/