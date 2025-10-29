@echo off
REM Diagnostic script for Windows build issues
REM 诊断脚本 - Windows构建问题

echo ============================================================
echo OnnxDet Windows Build Diagnostics
echo OnnxDet Windows 构建诊断工具
echo ============================================================
echo.

echo [1/7] Checking Rust installation...
where cargo >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo   ❌ FAIL: Rust not found. Install from https://rustup.rs/
    echo   ❌ 失败：未找到Rust。请从 https://rustup.rs/ 安装
) else (
    cargo --version
    rustc --version
    echo   ✓ OK: Rust is installed
    echo   ✓ 正常：Rust已安装
)
echo.

echo [2/7] Checking Python installation...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo   ⚠ WARNING: Python not found. Python bindings will be skipped.
    echo   ⚠ 警告：未找到Python。将跳过Python绑定。
) else (
    python --version
    echo   ✓ OK: Python is installed
    echo   ✓ 正常：Python已安装
)
echo.

echo [3/7] Checking Visual Studio Build Tools...
where link.exe >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo   ❌ FAIL: MSVC linker not found
    echo   ❌ 失败：未找到MSVC链接器
    echo   Install Visual Studio Build Tools with C++ workload
    echo   请安装Visual Studio Build Tools并选择C++工作负载
) else (
    link.exe /? 2>&1 | findstr /C:"Microsoft" >nul
    echo   ✓ OK: MSVC linker is available
    echo   ✓ 正常：MSVC链接器可用
)
echo.

echo [4/7] Checking ONNX Runtime location...
if "%ORT_LIB_LOCATION%"=="" (
    echo   ⚠ WARNING: ORT_LIB_LOCATION not set
    echo   ⚠ 警告：ORT_LIB_LOCATION未设置
    echo   Will attempt to download automatically
    echo   将尝试自动下载
) else (
    echo   ✓ OK: ORT_LIB_LOCATION = %ORT_LIB_LOCATION%
    if exist "%ORT_LIB_LOCATION%\lib\onnxruntime.dll" (
        echo   ✓ OK: onnxruntime.dll found
        echo   ✓ 正常：找到onnxruntime.dll
    ) else (
        echo   ❌ FAIL: onnxruntime.dll not found at specified location
        echo   ❌ 失败：在指定位置未找到onnxruntime.dll
    )
)
echo.

echo [5/7] Checking disk space...
for /f "tokens=3" %%a in ('dir /-c ^| find "bytes free"') do set FREESPACE=%%a
echo   Available disk space: %FREESPACE% bytes
echo   可用磁盘空间: %FREESPACE% 字节
echo   (Minimum recommended: 5GB / 建议最少：5GB)
echo.

echo [6/7] Checking Cargo configuration...
if exist ".cargo\config.toml" (
    echo   ✓ OK: .cargo\config.toml exists
    echo   ✓ 正常：.cargo\config.toml存在
) else (
    echo   ⚠ WARNING: .cargo\config.toml not found
    echo   ⚠ 警告：未找到.cargo\config.toml
)

if exist "Cargo.toml" (
    echo   ✓ OK: Cargo.toml exists
    echo   ✓ 正常：Cargo.toml存在
) else (
    echo   ❌ FAIL: Cargo.toml not found
    echo   ❌ 失败：未找到Cargo.toml
)
echo.

echo [7/7] Checking for previous build artifacts...
if exist "target" (
    for /f %%a in ('dir /s /b target 2^>nul ^| find /c /v ""') do set FILECOUNT=%%a
    echo   ⚠ Previous build found with %FILECOUNT% files
    echo   ⚠ 发现之前的构建，包含 %FILECOUNT% 个文件
    echo   Recommendation: Run 'cargo clean' before building
    echo   建议：构建前运行 'cargo clean'
) else (
    echo   ✓ OK: No previous build artifacts
    echo   ✓ 正常：没有之前的构建产物
)
echo.

echo ============================================================
echo Diagnostic complete. Summary:
echo 诊断完成。总结：
echo.
echo If all checks passed, try:
echo   build.bat
echo.
echo If you see errors above, address them first:
echo   - Install missing tools
echo   - Set ORT_LIB_LOCATION if needed
echo   - Run 'cargo clean' if there are old build artifacts
echo.
echo For detailed error messages during build:
echo   cargo build --release --verbose
echo.
echo See WINDOWS_BUILD.md for detailed troubleshooting.
echo 如需详细的构建错误信息，运行：
echo   cargo build --release --verbose
echo.
echo 详细故障排除请查看 WINDOWS_BUILD.md
echo ============================================================

pause
