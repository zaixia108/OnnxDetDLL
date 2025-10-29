use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array2, Array3, ArrayView2, Axis};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use std::path::Path;

/// Represents a detection bounding box
#[derive(Debug, Clone)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Represents a detection result
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox,
    pub confidence: f32,
    pub class_id: i32,
}

/// ONNX-based object detector
pub struct OnnxDetector {
    session: Session,
    input_height: usize,
    input_width: usize,
    conf_threshold: f32,
    iou_threshold: f32,
}

impl OnnxDetector {
    /// Create a new detector
    pub fn new(
        model_path: impl AsRef<Path>,
        conf_threshold: f32,
        iou_threshold: f32,
        _use_dml: bool,
    ) -> Result<Self> {
        // Initialize ONNX Runtime session
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

        // Get input shape - assume NCHW format with 640x640 as default
        let input_height = 640;
        let input_width = 640;

        Ok(Self {
            session,
            input_height,
            input_width,
            conf_threshold,
            iou_threshold,
        })
    }

    /// Detect objects in an image
    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<Detection>> {
        let (img_height, img_width) = (image.height() as usize, image.width() as usize);

        // Prepare input
        let (input_tensor, ratio) = self.prepare_input(image)?;

        // Add batch dimension: convert [C, H, W] to [1, C, H, W]
        let input_with_batch = input_tensor.insert_axis(Axis(0));

        // Run inference
        // Convert to the format expected by ORT: (shape, data)
        let shape = input_with_batch.shape().to_vec();
        let (data, _offset) = input_with_batch.to_owned().into_raw_vec_and_offset();
        let input_value = Value::from_array((shape, data)).map_err(|e| anyhow::anyhow!("Failed to create input value: {}", e))?;
        let inputs = vec![("images".to_string(), input_value)];
        let outputs = self.session.run(inputs).map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

        // Process output - extract data first before calling process_output
        let output = &outputs["output0"];
        let (shape, data) = output.try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("Failed to extract tensor: {}", e))?;
        
        // Convert to ndarray for processing - assuming shape [1, n_ch, n_anchors]
        let output_array = Array2::from_shape_vec(
            (shape[1] as usize, shape[2] as usize),
            data.to_vec(),
        )?;
        
        // Drop outputs to release the mutable borrow before calling process_output
        drop(outputs);
        
        let detections = self.process_output(output_array.view(), ratio, img_width, img_height)?;

        Ok(detections)
    }

    /// Prepare input image
    fn prepare_input(&self, image: &DynamicImage) -> Result<(Array3<f32>, f32)> {
        // Convert to RGB
        let rgb_image = image.to_rgb8();

        // Resize with padding
        let (resized, ratio) = self.resize_with_padding(&rgb_image);

        // Convert to CHW format and normalize
        let mut input_tensor = Array3::<f32>::zeros((3, self.input_height, self.input_width));

        for y in 0..self.input_height {
            for x in 0..self.input_width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                input_tensor[[0, y, x]] = pixel[0] as f32 / 255.0;
                input_tensor[[1, y, x]] = pixel[1] as f32 / 255.0;
                input_tensor[[2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }

        Ok((input_tensor, ratio))
    }

    /// Resize image with padding to maintain aspect ratio
    fn resize_with_padding(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, f32) {
        let (img_width, img_height) = image.dimensions();
        
        // Calculate scale ratio
        let scale = f32::min(
            self.input_width as f32 / img_width as f32,
            self.input_height as f32 / img_height as f32,
        );

        let new_width = (img_width as f32 * scale) as u32;
        let new_height = (img_height as f32 * scale) as u32;

        // Resize image
        let resized = image::imageops::resize(
            image,
            new_width,
            new_height,
            image::imageops::FilterType::Triangle,
        );

        // Create padded image with gray background
        let mut padded = ImageBuffer::from_pixel(
            self.input_width as u32,
            self.input_height as u32,
            Rgb([114u8, 114u8, 114u8]),
        );

        // Copy resized image to padded image
        image::imageops::overlay(&mut padded, &resized, 0, 0);

        (padded, 1.0 / scale)
    }

    /// Process model output
    fn process_output(
        &self,
        output: ArrayView2<f32>,
        ratio: f32,
        _img_width: usize,
        _img_height: usize,
    ) -> Result<Vec<Detection>> {
        let shape = output.shape();
        let n_ch = shape[0];      // Number of channels (classes + 4)
        let n_anchors = shape[1]; // Number of anchors
        let num_classes = n_ch - 4;

        let mut detections = Vec::new();

        // Process each anchor
        for i in 0..n_anchors {
            let anchor_data = output.column(i);

            // Extract box coordinates
            let x = anchor_data[0];
            let y = anchor_data[1];
            let w = anchor_data[2];
            let h = anchor_data[3];

            // Find max class score
            let mut max_class_score = 0.0f32;
            let mut max_class_id = 0i32;

            for j in 0..num_classes {
                let class_score = anchor_data[4 + j];
                if class_score > max_class_score {
                    max_class_score = class_score;
                    max_class_id = j as i32;
                }
            }

            let confidence = max_class_score;

            // Filter by confidence threshold
            if confidence < self.conf_threshold {
                continue;
            }

            // Apply ratio to coordinates
            let x = x * ratio;
            let y = y * ratio;
            let w = w * ratio;
            let h = h * ratio;

            // Convert to x1, y1, x2, y2
            let x1 = x - w / 2.0;
            let y1 = y - h / 2.0;
            let x2 = x + w / 2.0;
            let y2 = y + h / 2.0;

            detections.push(Detection {
                bbox: BBox { x1, y1, x2, y2 },
                confidence,
                class_id: max_class_id,
            });
        }

        // Apply NMS
        let detections = self.non_max_suppression(detections);

        Ok(detections)
    }

    /// Non-Maximum Suppression
    fn non_max_suppression(&self, mut detections: Vec<Detection>) -> Vec<Detection> {
        // Sort by confidence (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();
        let mut suppress = vec![false; detections.len()];

        for i in 0..detections.len() {
            if suppress[i] {
                continue;
            }

            keep.push(detections[i].clone());

            for j in (i + 1)..detections.len() {
                if suppress[j] {
                    continue;
                }

                // Only suppress boxes of the same class
                if detections[i].class_id != detections[j].class_id {
                    continue;
                }

                let iou = self.calculate_iou(&detections[i].bbox, &detections[j].bbox);
                if iou > self.iou_threshold {
                    suppress[j] = true;
                }
            }
        }

        keep
    }

    /// Calculate Intersection over Union
    fn calculate_iou(&self, box1: &BBox, box2: &BBox) -> f32 {
        let x1 = box1.x1.max(box2.x1);
        let y1 = box1.y1.max(box2.y1);
        let x2 = box1.x2.min(box2.x2);
        let y2 = box1.y2.min(box2.y2);

        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        
        let area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
        let area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
        let union = area1 + area2 - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

/// Python wrapper for OnnxDetector
#[pyclass]
struct PyOnnxDetector {
    detector: OnnxDetector,
}

#[pymethods]
impl PyOnnxDetector {
    #[new]
    #[pyo3(signature = (model_path, conf_threshold=0.3, iou_threshold=0.5, use_dml=false))]
    fn new(
        model_path: String,
        conf_threshold: f32,
        iou_threshold: f32,
        use_dml: bool,
    ) -> PyResult<Self> {
        let detector = OnnxDetector::new(model_path, conf_threshold, iou_threshold, use_dml)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create detector: {}", e)))?;
        Ok(Self { detector })
    }

    /// Detect objects in an image
    /// 
    /// Args:
    ///     image_data: numpy array of shape (H, W, 3) with RGB values
    /// 
    /// Returns:
    ///     tuple of (boxes, scores, classes) where:
    ///     - boxes: numpy array of shape (N, 4) with [x1, y1, x2, y2] format
    ///     - scores: numpy array of shape (N,) with confidence scores
    ///     - classes: numpy array of shape (N,) with class IDs
    fn detect<'py>(
        &mut self,
        py: Python<'py>,
        image_data: &Bound<'py, PyArray1<u8>>,
        width: usize,
        height: usize,
        _channels: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        // Convert numpy array to image
        let data = unsafe { image_data.as_slice()? };
        
        let img_buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(
            width as u32,
            height as u32,
            data.to_vec(),
        ).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid image data"))?;

        let image = DynamicImage::ImageRgb8(img_buffer);

        // Run detection
        let detections = self.detector.detect(&image)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Detection failed: {}", e)))?;

        let n = detections.len();

        // Prepare output arrays
        let boxes = PyArray2::<f32>::zeros(py, (n, 4), false);
        let scores = PyArray1::<f32>::zeros(py, n, false);
        let classes = PyArray1::<i32>::zeros(py, n, false);

        unsafe {
            let boxes_slice = boxes.as_slice_mut()?;
            let scores_slice = scores.as_slice_mut()?;
            let classes_slice = classes.as_slice_mut()?;

            for (i, det) in detections.iter().enumerate() {
                boxes_slice[i * 4] = det.bbox.x1;
                boxes_slice[i * 4 + 1] = det.bbox.y1;
                boxes_slice[i * 4 + 2] = det.bbox.x2;
                boxes_slice[i * 4 + 3] = det.bbox.y2;
                scores_slice[i] = det.confidence;
                classes_slice[i] = det.class_id;
            }
        }

        Ok((boxes, scores, classes))
    }

    /// Detect objects from file path
    /// 
    /// Args:
    ///     image_path: path to image file
    /// 
    /// Returns:
    ///     tuple of (boxes, scores, classes)
    fn detect_from_file<'py>(
        &mut self,
        py: Python<'py>,
        image_path: String,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i32>>)> {
        // Load image
        let image = image::open(&image_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load image: {}", e)))?;

        // Run detection
        let detections = self.detector.detect(&image)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Detection failed: {}", e)))?;

        let n = detections.len();

        // Prepare output arrays
        let boxes = PyArray2::<f32>::zeros(py, (n, 4), false);
        let scores = PyArray1::<f32>::zeros(py, n, false);
        let classes = PyArray1::<i32>::zeros(py, n, false);

        unsafe {
            let boxes_slice = boxes.as_slice_mut()?;
            let scores_slice = scores.as_slice_mut()?;
            let classes_slice = classes.as_slice_mut()?;

            for (i, det) in detections.iter().enumerate() {
                boxes_slice[i * 4] = det.bbox.x1;
                boxes_slice[i * 4 + 1] = det.bbox.y1;
                boxes_slice[i * 4 + 2] = det.bbox.x2;
                boxes_slice[i * 4 + 3] = det.bbox.y2;
                scores_slice[i] = det.confidence;
                classes_slice[i] = det.class_id;
            }
        }

        Ok((boxes, scores, classes))
    }
}

/// Python module
#[pymodule]
fn onnxdet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOnnxDetector>()?;
    Ok(())
}
