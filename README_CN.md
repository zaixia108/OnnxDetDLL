# OnnxDetDLL - Rust版本

原C++ OnnxDetDLL项目的Rust实现，提供高性能的ONNX对象检测功能和直接的Python调用接口。

## 项目简介

这个项目将原有的C++对象检测DLL项目完全重写为Rust实现，并通过PyO3提供原生Python绑定。相比原C++版本：

- **更安全**: Rust的内存安全保证消除了许多常见的C++问题
- **性能相当**: Release构建性能与C++版本持平
- **更易用**: 原生Python绑定，无需ctypes或cffi
- **跨平台**: 一套代码支持Windows、Linux和macOS

## 特性

- ✅ 使用ONNX Runtime进行高性能模型推理
- ✅ 图像预处理，保持长宽比
- ✅ 非极大值抑制(NMS)算法过滤重叠检测
- ✅ 原生Python绑定，直接调用无需封装
- ✅ 支持YOLOv8等主流检测模型
- ✅ 可配置的置信度和IoU阈值
- ✅ 完整的类型安全和错误处理

## 系统要求

### 构建环境

- Rust 1.70或更高版本 - 从 https://rustup.rs/ 安装
- Python 3.7或更高版本 - 用于Python绑定
- ONNX Runtime 1.20+ - 可以自动下载或手动安装

### 运行环境

- ONNX Runtime动态库 - 在构建时会自动处理

## 快速开始

### Windows用户特别说明 ⚠️

**如果您使用Windows系统，请先查看详细的Windows构建指南：[WINDOWS_BUILD.md](WINDOWS_BUILD.md)**

Windows快速构建步骤：
```cmd
# 克隆项目
git clone https://github.com/zaixia108/OnnxDetDLL.git
cd OnnxDetDLL

# 运行Windows构建脚本
build.bat

# 安装Python包
pip install target\wheels\*.whl
```

### Linux/macOS用户

### 1. 安装Rust

如果还没有安装Rust，请访问 https://rustup.rs/ 并按照说明安装。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. 克隆项目

```bash
git clone https://github.com/zaixia108/OnnxDetDLL.git
cd OnnxDetDLL
```

### 3. 使用自动构建脚本

项目提供了自动构建脚本，会自动下载ONNX Runtime并构建项目：

```bash
chmod +x build.sh
./build.sh
```

构建脚本会：
1. 检查Rust是否已安装
2. 下载ONNX Runtime（如果需要）
3. 构建Rust库
4. 构建Python wheel包

### 4. 安装Python包

```bash
pip install target/wheels/*.whl
```

## 使用方法

### Python示例

#### 基础用法 - 从文件检测

```python
from onnxdet import PyOnnxDetector

# 创建检测器实例
detector = PyOnnxDetector(
    model_path="你的模型.onnx",  # ONNX模型路径
    conf_threshold=0.3,           # 置信度阈值
    iou_threshold=0.5,            # NMS的IoU阈值
    use_dml=False                 # Windows上可设为True使用DirectML加速
)

# 从图片文件检测
boxes, scores, classes = detector.detect_from_file("测试图片.jpg")

# 打印检测结果
print(f"检测到 {len(boxes)} 个对象")
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes[i]
    confidence = scores[i]
    class_id = classes[i]
    print(f"对象 {i}:")
    print(f"  类别ID: {class_id}")
    print(f"  置信度: {confidence:.3f}")
    print(f"  边界框: ({x1:.1f}, {y1:.1f}) 到 ({x2:.1f}, {y2:.1f})")
```

#### 与OpenCV集成

```python
import cv2
import numpy as np
from onnxdet import PyOnnxDetector

# 创建检测器
detector = PyOnnxDetector("模型.onnx", 0.3, 0.5, False)

# 使用OpenCV读取图像
img = cv2.imread("测试图片.jpg")
if img is None:
    print("无法加载图片")
    exit(1)

# BGR转RGB（ONNX模型通常使用RGB格式）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channels = img_rgb.shape

# 将图像转换为一维数组
img_data = img_rgb.flatten().astype(np.uint8)

# 执行检测
boxes, scores, classes = detector.detect(img_data, width, height, channels)

print(f"检测到 {len(boxes)} 个对象")

# 在图像上绘制检测结果
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes[i]
    conf = scores[i]
    cls = classes[i]
    
    # 绘制边界框
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                 (0, 255, 0), 2)
    
    # 绘制标签
    label = f"类别 {cls}: {conf:.2f}"
    cv2.putText(img, label, (int(x1), int(y1) - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 保存结果
cv2.imwrite("检测结果.jpg", img)
print("结果已保存到 检测结果.jpg")
```

### Rust示例

如果你想在Rust代码中使用：

```rust
use onnxdet::OnnxDetector;
use image::open;

fn main() -> anyhow::Result<()> {
    // 创建检测器
    let mut detector = OnnxDetector::new(
        "model.onnx",
        0.3,  // 置信度阈值
        0.5,  // IoU阈值
        false // use_dml
    )?;

    // 加载图像
    let image = open("test.jpg")?;

    // 执行检测
    let detections = detector.detect(&image)?;

    // 处理结果
    for det in detections {
        println!("类别: {}, 置信度: {:.3}", det.class_id, det.confidence);
        println!("  边界框: ({:.1}, {:.1}, {:.1}, {:.1})", 
                 det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2);
    }

    Ok(())
}
```

## API文档

### Python API

#### PyOnnxDetector类

**构造函数**
```python
PyOnnxDetector(model_path: str, conf_threshold: float = 0.3, 
               iou_threshold: float = 0.5, use_dml: bool = False)
```

参数：
- `model_path`: ONNX模型文件的路径
- `conf_threshold`: 置信度阈值，低于此值的检测会被过滤掉（默认：0.3）
- `iou_threshold`: NMS的IoU阈值，用于过滤重叠的边界框（默认：0.5）
- `use_dml`: 是否使用DirectML加速（仅Windows，默认：False）

**方法**

##### detect_from_file(image_path: str)

从图片文件进行检测。

参数：
- `image_path`: 图片文件路径

返回：元组 (boxes, scores, classes)
- `boxes`: numpy数组，形状为(N, 4)，每行包含[x1, y1, x2, y2]坐标
- `scores`: numpy数组，形状为(N,)，包含置信度分数
- `classes`: numpy数组，形状为(N,)，包含类别ID

##### detect(image_data, width, height, channels)

从numpy数组进行检测。

参数：
- `image_data`: 1D numpy数组（uint8类型），包含展平的RGB图像数据
- `width`: 图像宽度
- `height`: 图像高度
- `channels`: 通道数（应为3，表示RGB）

返回：同detect_from_file

## 与C++版本的对比

| 特性 | C++版本 | Rust版本 |
|------|---------|----------|
| 语言 | C++ | Rust |
| 内存安全 | 手动管理 | 自动保证 |
| Python绑定 | ctypes/cffi | PyO3（原生） |
| 构建系统 | Visual Studio | Cargo |
| 跨平台 | 主要Windows | Windows/Linux/macOS |
| 依赖管理 | 手动 | Cargo自动 |
| 性能 | 高 | 高（相当） |

## 性能特点

Rust版本通过以下方式保证性能：

- 零成本抽象
- 高效的内存管理
- Release构建启用LTO优化
- 与C++版本相当的推理速度

## 故障排除

### ONNX Runtime加载失败

如果遇到ONNX Runtime库加载失败，请确保：

1. Linux上设置了LD_LIBRARY_PATH：
```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

2. macOS上设置了DYLD_LIBRARY_PATH：
```bash
export DYLD_LIBRARY_PATH=/path/to/onnxruntime/lib:$DYLD_LIBRARY_PATH
```

3. Windows上，确保onnxruntime.dll在系统PATH中

### 编译错误

如果遇到编译错误，请尝试：

1. 更新Rust到最新版本：
```bash
rustup update
```

2. 清理并重新构建：
```bash
cargo clean
cargo build --release
```

## 贡献

欢迎提交问题和拉取请求！

## 许可证

[请在此处插入你的许可证信息]

## 致谢

- 原始C++实现由OnnxDetDLL项目提供
- ONNX Runtime由Microsoft提供
- PyO3用于Python绑定
