#!/bin/bash
# Build script for OnnxDet Rust project

set -e

echo "=== OnnxDet Build Script ==="

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed. Please install from https://rustup.rs/"
    exit 1
fi

# Check if ONNX Runtime is available
if [ -z "$ORT_LIB_LOCATION" ]; then
    echo "ORT_LIB_LOCATION not set. Attempting to download ONNX Runtime..."
    
    # Determine OS and architecture
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    if [ "$OS" == "Linux" ] && [ "$ARCH" == "x86_64" ]; then
        ORT_VERSION="1.20.1"
        ORT_PACKAGE="onnxruntime-linux-x64-${ORT_VERSION}"
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PACKAGE}.tgz"
        
        if [ ! -d "$ORT_PACKAGE" ]; then
            echo "Downloading ONNX Runtime ${ORT_VERSION}..."
            wget -q "$ORT_URL"
            tar -xzf "${ORT_PACKAGE}.tgz"
            rm "${ORT_PACKAGE}.tgz"
        fi
        
        export ORT_LIB_LOCATION="$(pwd)/${ORT_PACKAGE}"
        export LD_LIBRARY_PATH="${ORT_LIB_LOCATION}/lib:${LD_LIBRARY_PATH}"
        echo "ONNX Runtime location: $ORT_LIB_LOCATION"
    else
        echo "Unsupported platform: $OS $ARCH"
        echo "Please manually install ONNX Runtime and set ORT_LIB_LOCATION"
        exit 1
    fi
fi

# Build the project
echo "Building Rust library..."
cargo build --release

# Check if Python is available for building Python bindings
if command -v python3 &> /dev/null; then
    echo "Building Python bindings..."
    
    # Install maturin if not present
    if ! command -v maturin &> /dev/null; then
        echo "Installing maturin..."
        pip3 install maturin
    fi
    
    # Build Python wheel
    maturin build --release
    
    echo "Python wheel built successfully!"
    echo "Install with: pip install target/wheels/*.whl"
else
    echo "Python3 not found. Skipping Python bindings build."
fi

echo ""
echo "=== Build Complete ==="
echo "Rust library: target/release/libonnxdet.so (or .dylib/.dll)"
echo ""
echo "To use the library, make sure to set:"
echo "  export LD_LIBRARY_PATH=${ORT_LIB_LOCATION}/lib:\$LD_LIBRARY_PATH"
