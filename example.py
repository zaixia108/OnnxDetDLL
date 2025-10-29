"""
Example usage of the OnnxDetector Python module

This demonstrates how to use the Rust-based ONNX detector from Python.
"""

import numpy as np
from onnxdet import PyOnnxDetector

def main():
    # Initialize detector
    model_path = "model.onnx"  # Replace with your model path
    detector = PyOnnxDetector(
        model_path=model_path,
        conf_threshold=0.3,
        iou_threshold=0.5,
        use_dml=False  # Set to True to use DirectML (Windows only)
    )

    # Method 1: Detect from file
    print("Method 1: Detecting from file...")
    boxes, scores, classes = detector.detect_from_file("image.jpg")
    print(f"Detected {len(boxes)} objects")
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        cls = classes[i]
        print(f"  Object {i}: class={cls}, confidence={conf:.3f}, bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    # Method 2: Detect from numpy array (for OpenCV integration)
    print("\nMethod 2: Detecting from numpy array...")
    import cv2
    img = cv2.imread("image.jpg")
    if img is not None:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img_rgb.shape
        
        # Flatten image to 1D array for passing to Rust
        img_data = img_rgb.flatten().astype(np.uint8)
        
        boxes, scores, classes = detector.detect(img_data, width, height, channels)
        print(f"Detected {len(boxes)} objects")
        
        # Draw results
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = scores[i]
            cls = classes[i]
            
            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw label
            label = f"Class {cls}: {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save result
        cv2.imwrite("result.jpg", img)
        print("Result saved to result.jpg")

if __name__ == "__main__":
    main()
