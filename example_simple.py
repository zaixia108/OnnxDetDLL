"""
Simple example of using OnnxDetector from Python

Before running this example:
1. Build the Python bindings: maturin develop --release
2. Make sure you have an ONNX model file
3. Install opencv-python if using the OpenCV integration: pip install opencv-python numpy
"""

from onnxdet import PyOnnxDetector

def example_detect_from_file():
    """Example: Detect objects directly from an image file"""
    print("Example 1: Detect from file")
    print("=" * 50)
    
    # Initialize detector
    # Replace 'model.onnx' with your actual model path
    detector = PyOnnxDetector(
        model_path="model.onnx",
        conf_threshold=0.3,    # Confidence threshold
        iou_threshold=0.5,     # IoU threshold for NMS
        use_dml=False          # Use DirectML (Windows only)
    )
    
    # Detect objects from file
    boxes, scores, classes = detector.detect_from_file("test_image.jpg")
    
    # Print results
    print(f"Detected {len(boxes)} objects:")
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        cls = classes[i]
        print(f"  [{i}] Class: {cls}, Confidence: {conf:.3f}, "
              f"Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")


def example_detect_with_opencv():
    """Example: Detect objects using OpenCV for image loading"""
    print("\nExample 2: Detect with OpenCV integration")
    print("=" * 50)
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return
    
    # Initialize detector
    detector = PyOnnxDetector("model.onnx", 0.3, 0.5, False)
    
    # Load image with OpenCV
    img = cv2.imread("test_image.jpg")
    if img is None:
        print("Failed to load image")
        return
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img_rgb.shape
    
    # Flatten image to 1D array (required by the detector)
    img_data = img_rgb.flatten().astype(np.uint8)
    
    # Run detection
    boxes, scores, classes = detector.detect(img_data, width, height, channels)
    
    print(f"Detected {len(boxes)} objects")
    
    # Draw results on image
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        cls = classes[i]
        
        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 255, 0), 2)
        
        # Draw label with background
        label = f"Class {cls}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10), 
                     (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
        cv2.putText(img, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Save result
    output_path = "result.jpg"
    cv2.imwrite(output_path, img)
    print(f"Result saved to {output_path}")


def main():
    print("OnnxDetector Python Examples")
    print("=" * 50)
    print()
    
    # Run examples
    try:
        example_detect_from_file()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_detect_with_opencv()
    except Exception as e:
        print(f"Example 2 failed: {e}")


if __name__ == "__main__":
    main()
