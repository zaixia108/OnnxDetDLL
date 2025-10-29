# Windows 构建指南 Windows Build Guide

本文档详细说明如何在Windows系统上构建和使用OnnxDetDLL Rust版本。

This document explains how to build and use the Rust version of OnnxDetDLL on Windows.

## 系统要求 System Requirements

### 必需软件 Required Software

1. **Windows 10/11** (64位 / 64-bit)
2. **Rust 1.70+**
   - 下载安装：https://rustup.rs/
   - 或使用安装程序：https://www.rust-lang.org/tools/install
3. **Python 3.7+** (用于Python绑定 / for Python bindings)
   - 下载：https://www.python.org/downloads/
4. **Visual Studio Build Tools** (Rust在Windows上需要)
   - 下载：https://visualstudio.microsoft.com/downloads/
   - 选择"Desktop development with C++"工作负载

### 可选软件 Optional Software

- **Git for Windows**: https://git-scm.com/download/win
- **PowerShell 5.1+** (Windows 10/11自带)

## 快速开始 Quick Start

### 方法1: 使用自动构建脚本 Using Automated Build Script

```cmd
# 克隆仓库 Clone repository
git clone https://github.com/zaixia108/OnnxDetDLL.git
cd OnnxDetDLL

# 运行Windows构建脚本 Run Windows build script
build.bat

# 安装Python包 Install Python package
pip install target\wheels\*.whl
```

构建脚本会自动：
- 检查Rust是否已安装
- 下载ONNX Runtime (如果需要)
- 编译Rust库
- 构建Python wheel包

The build script will automatically:
- Check if Rust is installed
- Download ONNX Runtime (if needed)
- Compile the Rust library
- Build Python wheel package

### 方法2: 手动构建 Manual Build

#### 步骤1: 安装Rust Step 1: Install Rust

在PowerShell或命令提示符中运行：

```powershell
# 下载并运行rustup安装程序
# Download and run rustup installer
curl https://win.rustup.rs/x86_64 -o rustup-init.exe
rustup-init.exe
```

按照提示完成安装，然后重启终端。

Follow the prompts to complete installation, then restart your terminal.

#### 步骤2: 下载ONNX Runtime Step 2: Download ONNX Runtime

```powershell
# 下载ONNX Runtime
$ORT_VERSION = "1.20.1"
$ORT_PACKAGE = "onnxruntime-win-x64-$ORT_VERSION"
$ORT_URL = "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_PACKAGE.zip"

# 使用PowerShell下载
Invoke-WebRequest -Uri $ORT_URL -OutFile "$ORT_PACKAGE.zip"

# 解压
Expand-Archive -Path "$ORT_PACKAGE.zip" -DestinationPath "."

# 设置环境变量
$env:ORT_LIB_LOCATION = "$PWD\$ORT_PACKAGE"
$env:PATH = "$env:ORT_LIB_LOCATION\lib;$env:PATH"
```

#### 步骤3: 构建项目 Step 3: Build Project

```cmd
# 构建Release版本
cargo build --release

# 查看生成的库文件
dir target\release\onnxdet.dll
```

#### 步骤4: 构建Python绑定 Step 4: Build Python Bindings

```cmd
# 安装maturin
pip install maturin

# 构建wheel包
maturin build --release

# 或直接安装到当前Python环境
maturin develop --release

# 查看生成的wheel包
dir target\wheels\
```

## 使用说明 Usage Instructions

### Python示例 Python Example

```python
import sys
import os

# 确保ONNX Runtime DLL在PATH中
# Ensure ONNX Runtime DLL is in PATH
onnx_runtime_path = r"C:\path\to\onnxruntime-win-x64-1.20.1\lib"
os.environ['PATH'] = onnx_runtime_path + os.pathsep + os.environ['PATH']

from onnxdet import PyOnnxDetector

# 创建检测器
detector = PyOnnxDetector(
    model_path="model.onnx",
    conf_threshold=0.3,
    iou_threshold=0.5,
    use_dml=False  # 设为True使用DirectML (仅限有GPU的Windows系统)
)

# 检测
boxes, scores, classes = detector.detect_from_file("test.jpg")
print(f"检测到 {len(boxes)} 个对象")
```

### DirectML 加速 DirectML Acceleration

在Windows上，可以使用DirectML进行GPU加速：

```python
# 启用DirectML (需要支持DirectX 12的GPU)
detector = PyOnnxDetector(
    model_path="model.onnx",
    conf_threshold=0.3,
    iou_threshold=0.5,
    use_dml=True  # 启用DirectML
)
```

**注意：** 当前版本的DirectML支持是通过ONNX Runtime的执行提供程序配置的，`use_dml`参数预留但可能需要额外的ONNX Runtime配置。

## 常见问题 Troubleshooting

### 问题1: "找不到onnxruntime.dll"

**症状：** 运行时错误，提示找不到`onnxruntime.dll`

**解决方案：**

方法A - 设置PATH环境变量：
```cmd
set PATH=C:\path\to\onnxruntime-win-x64-1.20.1\lib;%PATH%
```

方法B - 复制DLL到应用目录：
```cmd
copy C:\path\to\onnxruntime-win-x64-1.20.1\lib\onnxruntime.dll .
```

方法C - 在Python中设置：
```python
import os
os.add_dll_directory(r"C:\path\to\onnxruntime-win-x64-1.20.1\lib")
```

### 问题2: Rust编译错误 "linker 'link.exe' not found"

**症状：** Cargo构建失败，找不到链接器

**解决方案：**

安装Visual Studio Build Tools：
1. 下载：https://visualstudio.microsoft.com/downloads/
2. 选择"Desktop development with C++"工作负载
3. 重启命令提示符/PowerShell
4. 重新运行`cargo build --release`

### 问题3: Python导入错误

**症状：** `ImportError: DLL load failed`

**解决方案：**

1. 确保安装了正确的Python wheel：
```cmd
pip install --force-reinstall target\wheels\*.whl
```

2. 确保onnxruntime.dll在PATH中或应用目录中

3. 检查Python版本是否匹配（必须是64位Python）

### 问题4: 性能问题

**解决方案：**

1. 确保使用Release构建：
```cmd
cargo build --release
maturin build --release
```

2. 在Windows上启用DirectML（如果有GPU）：
```python
detector = PyOnnxDetector(..., use_dml=True)
```

## 开发环境配置 Development Setup

### Visual Studio Code推荐配置

安装以下扩展：
- rust-analyzer
- Python
- Better TOML

### 调试配置 Debug Configuration

在`.vscode/launch.json`中：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug Rust",
            "cargo": {
                "args": ["build", "--lib"],
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

## 性能优化 Performance Optimization

### Release构建优化 Release Build Optimizations

`Cargo.toml`已配置优化设置：

```toml
[profile.release]
opt-level = 3        # 最高优化级别
lto = true           # 链接时优化
```

### Windows特定优化 Windows-Specific Optimizations

1. **使用DirectML**（GPU加速）
2. **使用最新的ONNX Runtime版本**
3. **确保使用64位Python和Rust工具链**

## 打包分发 Distribution

### 创建独立应用程序 Creating Standalone Application

1. 构建Release版本
2. 复制必需的DLL：
   - `onnxruntime.dll`
   - `onnxruntime_providers_shared.dll`（如果存在）

3. 使用PyInstaller打包Python应用：
```cmd
pip install pyinstaller
pyinstaller --onefile --add-binary "onnxruntime.dll;." your_app.py
```

## 技术支持 Technical Support

如遇到其他问题：
1. 查看项目Issues: https://github.com/zaixia108/OnnxDetDLL/issues
2. 查看ONNX Runtime Windows文档
3. 检查Rust Windows安装指南

## 参考资源 References

- Rust官方Windows指南: https://forge.rust-lang.org/infra/other-installation-methods.html#windows
- ONNX Runtime发布页: https://github.com/microsoft/onnxruntime/releases
- PyO3 Windows指南: https://pyo3.rs/latest/building_and_distribution
- Maturin文档: https://www.maturin.rs/

## 版本兼容性 Version Compatibility

| 组件 | 最低版本 | 推荐版本 | 备注 |
|------|---------|---------|------|
| Windows | 10 | 11 | 64位 |
| Rust | 1.70 | 最新稳定版 | - |
| Python | 3.7 | 3.10+ | 64位 |
| ONNX Runtime | 1.20.1 | 1.20.1 | - |
| Visual Studio Build Tools | 2019 | 2022 | C++ 工作负载 |

---

**最后更新：** 2025-10-29
**测试环境：** Windows 11 Pro, Rust 1.70, Python 3.10
