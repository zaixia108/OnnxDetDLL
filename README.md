# OnnxDetDLL - Rust Edition

这是原C++ OnnxDetDLL项目的Rust实现版本，提供高性能的ONNX对象检测功能和Python绑定。

A high-performance ONNX-based object detection library written in Rust with Python bindings.

This is a Rust translation of the original C++ OnnxDetDLL project.

## 特性 Features

- ✅ 使用ONNX Runtime进行模型推理 / ONNX Runtime integration for model inference
- ✅ 保持长宽比的图像预处理 / Image preprocessing with aspect ratio preservation
- ✅ 非极大值抑制(NMS)过滤重叠检测 / Non-Maximum Suppression (NMS) for filtering
- ✅ Python绑定，易于集成 / Python bindings for easy integration
- ✅ 支持YOLOv8等检测模型 / Support for YOLOv8 and similar models
- ✅ 可配置的置信度和IoU阈值 / Configurable confidence and IoU thresholds
- ✅ 跨平台支持 / Cross-platform support (Windows, Linux, macOS)

## 文档 Documentation

- [English README](README.md) - This file
- [中文文档](README_CN.md) - Complete Chinese documentation
- [**Windows Build Guide**](WINDOWS_BUILD.md) - **Windows系统构建指南 (必读)**
- [**Windows Linker Fix**](WINDOWS_LINKER_FIX.md) - **链接器错误快速修复**
- [Migration Guide](MIGRATION.md) - Guide for migrating from C++ version
- [Development Guide](DEVELOPMENT.md) - For contributors and developers

## 快速链接 Quick Links

- 🪟 [**Windows用户看这里**](WINDOWS_BUILD.md) - Complete Windows build guide
- ⚠️ [**链接器错误修复**](WINDOWS_LINKER_FIX.md) - Fix linker error 1181
- 🇨🇳 [查看中文文档](README_CN.md) - Complete documentation in Chinese
- 📖 [查看迁移指南](MIGRATION.md) - How to migrate from C++ version  
- 👨‍💻 [开发者指南](DEVELOPMENT.md) - Contributing and development

---

### 构建环境 For Building

- Rust 1.70+ (从 https://rustup.rs/ 安装 / install from https://rustup.rs/)
- Python 3.7+ (用于Python绑定 / for Python bindings)
- ONNX Runtime 1.20+ (自动下载或手动安装 / auto-downloaded or manual install)

### 运行环境 For Running

- ONNX Runtime库 (与Python包一起打包 / bundled with the Python package)

## 安装说明 Installation

### Windows用户 For Windows Users

**Windows系统请查看详细的构建指南：[WINDOWS_BUILD.md](WINDOWS_BUILD.md)**

⚠️ **遇到链接器错误？** 查看：[WINDOWS_LINKER_FIX.md](WINDOWS_LINKER_FIX.md)

简要步骤 Quick steps:
```cmd
# 克隆仓库
git clone https://github.com/zaixia108/OnnxDetDLL.git
cd OnnxDetDLL

# 运行Windows构建脚本（使用Release模式）
build.bat

# 如果遇到链接器错误，先运行：
# cargo clean
# 然后再次运行 build.bat

# 安装Python包
pip install target\wheels\*.whl
```

### Linux/macOS用户 For Linux/macOS Users

### 快速开始 Quick Start

使用提供的构建脚本（推荐） / Use the provided build script (recommended):

```bash
# Clone the repository / 克隆仓库
git clone https://github.com/zaixia108/OnnxDetDLL.git
cd OnnxDetDLL

# Run build script (Linux/macOS) / 运行构建脚本
./build.sh

# For Python bindings / Python绑定安装
pip install target/wheels/*.whl
```

### 手动构建 Manual Build

1. 克隆仓库 Clone the repository:
```bash
git clone https://github.com/zaixia108/OnnxDetDLL.git
cd OnnxDetDLL
```

2. 下载ONNX Runtime (Linux示例 / Linux example):
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-1.20.1.tgz
export ORT_LIB_LOCATION=$(pwd)/onnxruntime-linux-x64-1.20.1
export LD_LIBRARY_PATH=$ORT_LIB_LOCATION/lib:$LD_LIBRARY_PATH
```

3. 构建Rust库 Build the Rust library:
```bash
cargo build --release
```

4. 构建Python模块 Build the Python module:
```bash
pip install maturin
maturin develop --release
```

## 使用方法 Usage

### Python示例 Python Examples

#### 基本用法 Basic Usage

```python
from onnxdet import PyOnnxDetector

# 初始化检测器 Initialize detector
detector = PyOnnxDetector(
    model_path="model.onnx",
    conf_threshold=0.3,
    iou_threshold=0.5,
    use_dml=False  # Windows上设为True使用DirectML / Set to True for DirectML on Windows
)

# 从文件检测 Detect from file
boxes, scores, classes = detector.detect_from_file("image.jpg")

# 处理结果 Process results
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes[i]
    confidence = scores[i]
    class_id = classes[i]
    print(f"对象 {i}: 类别={class_id}, 置信度={confidence:.3f}")
    print(f"  边界框: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
```

#### 使用OpenCV With OpenCV

```python
import cv2
import numpy as np
from onnxdet import PyOnnxDetector

# 初始化检测器 Initialize detector
detector = PyOnnxDetector("model.onnx", 0.3, 0.5)

# 加载图像 Load image
img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channels = img_rgb.shape

# 展平并检测 Flatten and detect
img_data = img_rgb.flatten().astype(np.uint8)
boxes, scores, classes = detector.detect(img_data, width, height, channels)

# 绘制结果 Draw results
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes[i]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

cv2.imwrite("result.jpg", img)
```

### Rust

```rust
use onnxdet::OnnxDetector;
use image::open;

fn main() -> anyhow::Result<()> {
    // Create detector
    let detector = OnnxDetector::new(
        "model.onnx",
        0.3,  // confidence threshold
        0.5,  // IoU threshold
        false // use_dml
    )?;

    // Load image
    let image = open("image.jpg")?;

    // Detect objects
    let detections = detector.detect(&image)?;

    // Process results
    for det in detections {
        println!("Class: {}, Confidence: {:.3}", det.class_id, det.confidence);
        println!("  Box: ({:.1}, {:.1}, {:.1}, {:.1})", 
                 det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2);
    }

    Ok(())
}
```

## API Reference

### Python API

#### `PyOnnxDetector`

```python
PyOnnxDetector(model_path: str, conf_threshold: float = 0.3, 
               iou_threshold: float = 0.5, use_dml: bool = False)
```

**Parameters:**
- `model_path`: Path to the ONNX model file
- `conf_threshold`: Confidence threshold for filtering detections (default: 0.3)
- `iou_threshold`: IoU threshold for Non-Maximum Suppression (default: 0.5)
- `use_dml`: Use DirectML acceleration (Windows only, default: False)

**Methods:**

##### `detect(image_data, width, height, channels)`
Detect objects from a flattened numpy array.

**Parameters:**
- `image_data`: 1D numpy array of uint8 (flattened RGB image)
- `width`: Image width
- `height`: Image height  
- `channels`: Number of channels (should be 3 for RGB)

**Returns:** Tuple of (boxes, scores, classes)
- `boxes`: numpy array of shape (N, 4) with [x1, y1, x2, y2] coordinates
- `scores`: numpy array of shape (N,) with confidence scores
- `classes`: numpy array of shape (N,) with class IDs

##### `detect_from_file(image_path)`
Detect objects from an image file.

**Parameters:**
- `image_path`: Path to the image file

**Returns:** Tuple of (boxes, scores, classes) (same format as above)

## Comparison with Original C++ Version

| Feature | C++ Version | Rust Version |
|---------|-------------|--------------|
| Language | C++ | Rust |
| Memory Safety | Manual | Automatic |
| Python Bindings | ctypes/cffi | PyO3 (native) |
| Build System | Visual Studio | Cargo |
| Cross-platform | Windows focused | Windows/Linux/macOS |
| Dependencies | OpenCV, ONNX RT | image, ort crates |
| Performance | High | High (comparable) |

## Performance

The Rust version provides comparable performance to the C++ version:
- Zero-cost abstractions
- Efficient memory management
- Optimized release builds with LTO

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Insert your license here]

## Acknowledgments

- Original C++ implementation by the OnnxDetDLL project
- ONNX Runtime by Microsoft
- PyO3 for Python bindings
