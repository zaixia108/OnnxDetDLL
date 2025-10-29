# 开发者指南 Developer Guide

## 项目结构 Project Structure

```
OnnxDetDLL/
├── src/
│   └── lib.rs              # 主要Rust源代码 / Main Rust source
├── Cargo.toml              # Rust项目配置 / Rust project config
├── pyproject.toml          # Python包配置 / Python package config
├── build.sh                # 构建脚本 / Build script
├── example.py              # Python示例（完整） / Python example (full)
├── example_simple.py       # Python示例（简单） / Python example (simple)
├── README.md               # 英文文档 / English docs
├── README_CN.md            # 中文文档 / Chinese docs
├── MIGRATION.md            # 迁移指南 / Migration guide
└── OnnxDet.cpp/h           # 原始C++代码（参考） / Original C++ (reference)
```

## 开发环境设置 Development Setup

### 1. 安装必需工具 Required Tools

```bash
# Rust工具链 / Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python开发工具 / Python dev tools
pip install maturin pytest numpy opencv-python

# 代码格式化工具 / Code formatting
rustup component add rustfmt clippy
```

### 2. 开发模式构建 Development Build

```bash
# 下载ONNX Runtime（如果需要）
export ORT_LIB_LOCATION=/path/to/onnxruntime
export LD_LIBRARY_PATH=$ORT_LIB_LOCATION/lib:$LD_LIBRARY_PATH

# 快速构建（调试模式）
cargo build

# Python绑定开发模式（自动重新加载）
maturin develop
```

### 3. 代码检查 Code Linting

```bash
# Rust代码格式化
cargo fmt

# Rust代码检查
cargo clippy -- -D warnings

# 运行测试
cargo test
```

## 架构说明 Architecture

### 核心组件 Core Components

#### 1. OnnxDetector (Rust核心)

```rust
pub struct OnnxDetector {
    session: Session,           // ONNX Runtime会话
    input_height: usize,        // 输入高度
    input_width: usize,         // 输入宽度
    conf_threshold: f32,        // 置信度阈值
    iou_threshold: f32,         // IoU阈值
}
```

主要方法：
- `new()`: 初始化检测器
- `detect()`: 执行对象检测
- `prepare_input()`: 图像预处理
- `process_output()`: 处理模型输出
- `non_max_suppression()`: NMS算法

#### 2. PyOnnxDetector (Python绑定)

```rust
#[pyclass]
struct PyOnnxDetector {
    detector: OnnxDetector,
}
```

Python可见方法：
- `__new__()`: 构造函数
- `detect()`: 从numpy数组检测
- `detect_from_file()`: 从文件检测

### 数据流 Data Flow

```
图像输入 Image Input
    ↓
resize_with_padding()  # 保持长宽比缩放
    ↓
RGB归一化 RGB Normalization (0-1)
    ↓
转换为NCHW格式 Convert to NCHW
    ↓
ONNX Runtime推理 ONNX Runtime Inference
    ↓
process_output()  # 解析输出
    ↓
non_max_suppression()  # NMS过滤
    ↓
返回检测结果 Return Detections
```

## 添加新功能 Adding New Features

### 示例：添加批量处理 Example: Adding Batch Processing

1. **修改Rust核心 Modify Rust Core**

```rust
impl OnnxDetector {
    pub fn detect_batch(&mut self, images: &[DynamicImage]) 
        -> Result<Vec<Vec<Detection>>> 
    {
        let mut all_detections = Vec::new();
        for image in images {
            all_detections.push(self.detect(image)?);
        }
        Ok(all_detections)
    }
}
```

2. **添加Python绑定 Add Python Binding**

```rust
#[pymethods]
impl PyOnnxDetector {
    fn detect_batch<'py>(
        &mut self,
        py: Python<'py>,
        image_paths: Vec<String>,
    ) -> PyResult<Vec<(...)>> {
        // 实现批量检测
    }
}
```

3. **编写测试 Write Tests**

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_batch_detection() {
        // 测试代码
    }
}
```

4. **更新文档 Update Documentation**

在README.md和README_CN.md中添加使用示例。

## 性能优化指南 Performance Optimization

### 1. 编译优化 Compilation Optimizations

Cargo.toml已配置：

```toml
[profile.release]
opt-level = 3        # 最高优化级别
lto = true           # 链接时优化
```

### 2. 运行时优化 Runtime Optimizations

- 使用`&mut self`而非`&self`用于session.run()（避免不必要的锁）
- 预分配内存用于重复操作
- 使用引用避免不必要的克隆

### 3. 性能分析 Profiling

```bash
# Linux下使用perf
cargo build --release
perf record --call-graph dwarf target/release/your_binary
perf report

# 使用flamegraph
cargo install flamegraph
cargo flamegraph
```

## 测试指南 Testing Guide

### 单元测试 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_calculation() {
        let detector = /* ... */;
        let box1 = BBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
        let box2 = BBox { x1: 5.0, y1: 5.0, x2: 15.0, y2: 15.0 };
        let iou = detector.calculate_iou(&box1, &box2);
        assert!((iou - 0.1428).abs() < 0.001);
    }
}
```

### Python集成测试 Python Integration Tests

```python
import pytest
from onnxdet import PyOnnxDetector
import numpy as np

def test_basic_detection():
    detector = PyOnnxDetector("model.onnx", 0.3, 0.5)
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8).flatten()
    boxes, scores, classes = detector.detect(img, 640, 640, 3)
    assert boxes.shape[1] == 4
```

## 调试技巧 Debugging Tips

### Rust调试 Debugging Rust

```bash
# 使用rust-lldb或rust-gdb
rust-lldb target/debug/onnxdet
# 或
rust-gdb target/debug/onnxdet
```

### Python调试 Debugging Python

```python
import pdb
from onnxdet import PyOnnxDetector

detector = PyOnnxDetector("model.onnx")
pdb.set_trace()  # 设置断点
boxes, scores, classes = detector.detect_from_file("test.jpg")
```

### 日志输出 Logging

在Rust代码中添加调试输出：

```rust
#[cfg(debug_assertions)]
eprintln!("Debug: shape = {:?}", shape);
```

## 常见问题解决 Troubleshooting

### 问题1：编译错误 "ort::Session not found"

**解决方案：**
```bash
# 确保启用std feature
cargo clean
cargo build --features "ort/std"
```

### 问题2：运行时错误 "ONNX Runtime library not found"

**解决方案：**
```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

### 问题3：Python导入错误

**解决方案：**
```bash
# 重新构建Python绑定
maturin develop --release
# 或安装wheel
pip install --force-reinstall target/wheels/*.whl
```

## 代码风格指南 Code Style Guide

### Rust代码风格

遵循Rust标准风格：

```rust
// 好的示例
pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<Detection>> {
    let (input, ratio) = self.prepare_input(image)?;
    // ...
}

// 避免
pub fn detect(&mut self,image:&DynamicImage)->Result<Vec<Detection>>{
    let(input,ratio)=self.prepare_input(image)?;
    //...
}
```

### Python代码风格

遵循PEP 8：

```python
# 好的示例
def example_function():
    detector = PyOnnxDetector("model.onnx", 0.3, 0.5)
    return detector

# 避免
def exampleFunction():
    detector=PyOnnxDetector("model.onnx",0.3,0.5)
    return detector
```

## 贡献流程 Contribution Process

1. Fork项目
2. 创建feature分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 创建Pull Request

### Pull Request检查清单

- [ ] 代码通过`cargo fmt`格式化
- [ ] 代码通过`cargo clippy`检查
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 测试全部通过
- [ ] 添加了更改日志条目

## 发布流程 Release Process

1. 更新版本号（Cargo.toml和pyproject.toml）
2. 更新CHANGELOG.md
3. 创建git tag：`git tag -a v0.1.0 -m "Release v0.1.0"`
4. 构建release：`cargo build --release`
5. 构建Python wheel：`maturin build --release`
6. 发布到crates.io（可选）：`cargo publish`
7. 发布到PyPI（可选）：`maturin publish`

## 资源链接 Resources

### Rust
- [Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)

### PyO3
- [PyO3 User Guide](https://pyo3.rs/)
- [Maturin Guide](https://maturin.rs/)

### ONNX Runtime
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [ort crate docs](https://docs.rs/ort/)

## 获取帮助 Getting Help

- 提交Issue：[GitHub Issues](https://github.com/zaixia108/OnnxDetDLL/issues)
- 讨论：[GitHub Discussions](https://github.com/zaixia108/OnnxDetDLL/discussions)
- 邮件：[维护者邮箱]

## 许可证 License

本项目采用 [LICENSE NAME] 许可证 - 详见LICENSE文件。
