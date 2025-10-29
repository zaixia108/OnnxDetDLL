# 迁移指南：从C++版本到Rust版本

Migration Guide: From C++ to Rust Version

## 概述 Overview

本文档帮助从C++版本迁移到Rust版本的用户理解两个版本的主要区别。

This document helps users migrating from the C++ version understand the key differences.

## 主要改进 Key Improvements

### 1. 内存安全 Memory Safety

**C++ 版本:**
- 手动内存管理（malloc/free）
- 潜在的内存泄漏风险
- 需要显式调用ReleaseResults

**Rust 版本:**
- 自动内存管理
- 编译时保证无内存泄漏
- 无需手动释放内存

### 2. Python集成 Python Integration

**C++ 版本:**
```python
# 需要ctypes封装
import ctypes
lib = ctypes.CDLL('OnnxDet.dll')
# 复杂的类型转换...
```

**Rust 版本:**
```python
# 原生Python模块
from onnxdet import PyOnnxDetector
detector = PyOnnxDetector("model.onnx")
# 直接使用，无需类型转换
```

### 3. 构建系统 Build System

**C++ 版本:**
- Visual Studio项目文件
- 手动配置依赖
- 平台特定

**Rust 版本:**
- Cargo统一构建
- 自动依赖管理
- 跨平台一致

## API对比 API Comparison

### 创建检测器 Creating Detector

**C++ API:**
```cpp
void* detector = CreateDetector();
bool success = InitDetector(detector, "model.onnx", 0.3, 0.5, true);
```

**Rust/Python API:**
```python
detector = PyOnnxDetector("model.onnx", 0.3, 0.5, False)
```

### 执行检测 Running Detection

**C++ API:**
```cpp
float* boxes;
float* scores;
int* classes;
int count;
Detect(detector, image_data, width, height, 3, 
       &boxes, &scores, &classes, &count);
// 使用结果...
ReleaseResults(boxes, scores, classes);
```

**Rust/Python API:**
```python
boxes, scores, classes = detector.detect(image_data, width, height, 3)
# 自动内存管理，无需释放
```

### 清理资源 Cleanup

**C++ API:**
```cpp
ReleaseResults(boxes, scores, classes);
DestroyDetector(detector);
```

**Rust/Python API:**
```python
# 自动清理，无需显式操作
```

## 迁移步骤 Migration Steps

### 1. 安装环境 Setup Environment

```bash
# 安装Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 克隆Rust版本
git clone https://github.com/zaixia108/OnnxDetDLL.git
cd OnnxDetDLL

# 构建
./build.sh
```

### 2. 更新Python代码 Update Python Code

**旧代码（C++版本）:**
```python
import ctypes
import numpy as np

# 加载DLL
lib = ctypes.CDLL('OnnxDet.dll')

# 设置函数签名
lib.CreateDetector.restype = ctypes.c_void_p
lib.InitDetector.argtypes = [ctypes.c_void_p, ctypes.c_char_p, 
                              ctypes.c_float, ctypes.c_float, ctypes.c_bool]
# ... 更多设置 ...

# 创建检测器
detector = lib.CreateDetector()
lib.InitDetector(detector, b"model.onnx", 0.3, 0.5, False)

# 检测
boxes_ptr = ctypes.POINTER(ctypes.c_float)()
scores_ptr = ctypes.POINTER(ctypes.c_float)()
classes_ptr = ctypes.POINTER(ctypes.c_int)()
count = ctypes.c_int()

img_data = np.ascontiguousarray(img.flatten(), dtype=np.uint8)
lib.Detect(detector, img_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
           width, height, 3, 
           ctypes.byref(boxes_ptr), ctypes.byref(scores_ptr),
           ctypes.byref(classes_ptr), ctypes.byref(count))

# 转换结果
boxes = np.ctypeslib.as_array(boxes_ptr, shape=(count.value, 4))
# ... 复制数据 ...

# 释放内存
lib.ReleaseResults(boxes_ptr, scores_ptr, classes_ptr)
lib.DestroyDetector(detector)
```

**新代码（Rust版本）:**
```python
from onnxdet import PyOnnxDetector
import numpy as np

# 创建检测器
detector = PyOnnxDetector("model.onnx", 0.3, 0.5, False)

# 检测
img_data = img.flatten().astype(np.uint8)
boxes, scores, classes = detector.detect(img_data, width, height, 3)

# 直接使用结果，无需手动内存管理
```

### 3. 性能对比测试 Performance Testing

建议进行以下测试以验证迁移后的性能：

```python
import time
import numpy as np
from onnxdet import PyOnnxDetector

detector = PyOnnxDetector("model.onnx", 0.3, 0.5, False)

# 预热
img_data = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8).flatten()
detector.detect(img_data, 640, 640, 3)

# 性能测试
times = []
for _ in range(100):
    start = time.time()
    boxes, scores, classes = detector.detect(img_data, 640, 640, 3)
    times.append(time.time() - start)

print(f"平均推理时间: {np.mean(times)*1000:.2f}ms")
print(f"FPS: {1.0/np.mean(times):.2f}")
```

## 功能对比表 Feature Comparison

| 功能 Feature | C++ | Rust | 备注 Notes |
|-------------|-----|------|-----------|
| ONNX Runtime | ✅ | ✅ | 相同版本支持 |
| 图像预处理 | ✅ | ✅ | Rust使用image crate |
| NMS | ✅ | ✅ | 算法相同 |
| Python绑定 | ⚠️ | ✅ | Rust版本更原生 |
| 内存安全 | ❌ | ✅ | Rust编译时保证 |
| 跨平台 | ⚠️ | ✅ | Rust更容易 |
| 性能 | ✅ | ✅ | 相当 |
| DirectML | ✅ | 🚧 | 计划支持 |

## 常见问题 FAQ

### Q: Rust版本的性能如何？
**A:** Release构建下，Rust版本的性能与C++版本相当，通常在误差范围内。

### Q: 能否直接替换DLL？
**A:** 不能直接替换。需要重新编译Python绑定并更新调用代码。

### Q: 支持的模型格式有变化吗？
**A:** 没有。两个版本都支持标准ONNX格式，模型文件可以通用。

### Q: 内存占用有差异吗？
**A:** 通常相当。Rust的内存管理更高效，但差异不明显。

### Q: Windows上的DirectML支持如何？
**A:** 当前版本的use_dml参数预留但未实现。可通过ONNX Runtime的执行提供程序配置。

## 技术细节 Technical Details

### 类型映射 Type Mapping

| C++ | Rust | Python |
|-----|------|--------|
| float* | Vec<f32> | numpy.ndarray |
| int* | Vec<i32> | numpy.ndarray |
| void* | Box<T> | 对象引用 |
| const char* | &str / String | str |

### 错误处理 Error Handling

**C++:** 返回bool，错误信息输出到stderr
**Rust:** Result<T, E>类型，使用anyhow进行错误传播
**Python:** 抛出Python异常（PyRuntimeError等）

## 获取帮助 Getting Help

遇到问题？

1. 查看example_simple.py示例
2. 阅读完整文档：README.md / README_CN.md
3. 提交Issue：https://github.com/zaixia108/OnnxDetDLL/issues

## 贡献 Contributing

欢迎提交改进建议和代码！特别是：

- DirectML支持
- 更多模型格式支持
- 性能优化
- 文档改进
