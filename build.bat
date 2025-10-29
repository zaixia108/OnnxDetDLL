@echo off
REM Build script for OnnxDet Rust project on Windows

echo === OnnxDet Build Script for Windows ===
echo.

REM Check if Rust is installed
where cargo >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Rust is not installed. Please install from https://rustup.rs/
    exit /b 1
)

REM Check if ONNX Runtime is available
if "%ORT_LIB_LOCATION%"=="" (
    echo ORT_LIB_LOCATION not set. Attempting to download ONNX Runtime...
    
    set ORT_VERSION=1.20.1
    set ORT_PACKAGE=onnxruntime-win-x64-%ORT_VERSION%
    set ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ORT_VERSION%/%ORT_PACKAGE%.zip
    
    if not exist "%ORT_PACKAGE%" (
        echo Downloading ONNX Runtime %ORT_VERSION%...
        
        REM Try to download with PowerShell
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%ORT_URL%' -OutFile '%ORT_PACKAGE%.zip'}"
        
        if %ERRORLEVEL% neq 0 (
            echo Failed to download ONNX Runtime. Please download manually from:
            echo %ORT_URL%
            echo And extract to the current directory.
            exit /b 1
        )
        
        echo Extracting ONNX Runtime...
        powershell -Command "Expand-Archive -Path '%ORT_PACKAGE%.zip' -DestinationPath '.' -Force"
        del "%ORT_PACKAGE%.zip"
    )
    
    set ORT_LIB_LOCATION=%cd%\%ORT_PACKAGE%
    set PATH=%ORT_LIB_LOCATION%\lib;%PATH%
    echo ONNX Runtime location: %ORT_LIB_LOCATION%
)

REM Build the project
echo Building Rust library...
cargo build --release

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

REM Check if Python is available for building Python bindings
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Building Python bindings...
    
    REM Install maturin if not present
    where maturin >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo Installing maturin...
        pip install maturin
    )
    
    REM Build Python wheel
    maturin build --release
    
    echo Python wheel built successfully!
    echo Install with: pip install target\wheels\*.whl
) else (
    echo Python not found. Skipping Python bindings build.
)

echo.
echo === Build Complete ===
echo Rust library: target\release\onnxdet.dll
echo.
echo To use the library, make sure to set:
echo   set PATH=%ORT_LIB_LOCATION%\lib;%%PATH%%
echo.
echo Or copy onnxruntime.dll to your application directory.
