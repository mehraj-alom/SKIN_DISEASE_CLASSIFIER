import cv2
import numpy as np
import os 
import yaml
from yaml.loader import SafeLoader

class YOLOConfig:
    """Configuration class for YOLO model parameters"""
    def __init__(self, 
                 confidence_threshold=0.4, 
                 prob_threshold=0.25,
                 nms_threshold=0.45,
                 input_width=640,
                 input_height=640):
        self.confidence_threshold = confidence_threshold
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.input_width = input_width
        self.input_height = input_height


class YOLODetector:
    """YOLO Object Detection class"""
    
    def __init__(self, model_path: str, yaml_path: str, config: YOLOConfig = None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to ONNX model file
            yaml_path: Path to data.yaml file containing labels
            config: YOLOConfig object with detection parameters
        """
        self.config = config if config else YOLOConfig()
        self.labels = self._load_labels(yaml_path)
        self.model = self._load_model(model_path)
        
    def _load_labels(self, file_path: str):
        """Load class labels from YAML file"""
        with open(file_path, "r") as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)
        return data_yaml["names"]
    
    def _load_model(self, file_path: str):
        """Load YOLO model from ONNX file"""
        yolo = cv2.dnn.readNetFromONNX(file_path)
        yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return yolo
    
    def _prepare_input(self, image: np.ndarray):
        """
        Prepare image for YOLO input
        
        Returns:
            blob, input_img, x_factor, y_factor
        """
        img = image.copy()
        row, col, c = img.shape
        
        #create blank square matrix for overlay
        max_rc = max(row, col)
        input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_img[0:row, 0:col] = img
        
        # Create blob
        blob = cv2.dnn.blobFromImage(
            image=input_img,
            scalefactor=1/255,
            size=(self.config.input_width, self.config.input_height),
            swapRB=True,
            crop=False
        )
        
        x_factor = input_img.shape[1] / self.config.input_width
        y_factor = input_img.shape[0] / self.config.input_height
        
        return blob, input_img, x_factor, y_factor
    
    def _apply_nms(self, detections: np.ndarray):
        """
        Apply Non-Maximum Suppression to filter detections
        
        Returns:
            boxes,confidence, classes
        """
        boxes = []
        confidences = []
        classes = []
        
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # confidence of detection on object
            
            if confidence > self.config.confidence_threshold:
                class_score = row[5:].max()  # maximum probability from objects
                class_id = row[5:].argmax()  #get the index position at which max probability occur
                
                if class_score > self.config.prob_threshold:
                    cx, cy, w, h = row[0:4]
                    #bndbox
                    # Left,top, width and  height
                    left = int(cx - 0.5*w)
                    top = int(cy - 0.5*h)
                    width = int(w)
                    height = int(h)
                    
                    box = np.array([left, top, width, height])
                    
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)
        
        return boxes, confidences, classes
    
    def detect(self, image: np.ndarray):
        """
        Perform object detection on image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            boxes, confidences, classes, x_factor, y_factor
        """



        blob, input_img, x_factor, y_factor = self._prepare_input(image)
        
        self.model.setInput(blob)
        preds = self.model.forward()
        
        if preds.ndim == 2:
            preds = np.expand_dims(preds, axis=0)
        
        detections = preds[0]
        
       #NMS filtering
        boxes, confidences, classes = self._apply_nms(detections)
        
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            self.config.prob_threshold, 
            self.config.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = [boxes[i] for i in indices]
            filtered_confidences = [confidences[i] for i in indices]
            filtered_classes = [classes[i] for i in indices]
            
            return filtered_boxes, filtered_confidences, filtered_classes, x_factor, y_factor
        else:
            return [], [], [], x_factor, y_factor


class DetectionVisualizer:
    """Class for visualizing detection results"""
    
    def __init__(self, labels: list, box_color=(0, 255, 0), text_color=(0, 255, 0)):
        """
        Initialize visualizer
        
        Args:
            labels: List of class labels
            box_color: BGR color tuple for bounding boxes
            text_color: BGR color tuple for text
        """
        self.labels = labels
        self.box_color = box_color
        self.text_color = text_color
    
    def draw_boxes(self, image: np.ndarray, boxes, confidences, classes, x_factor, y_factor):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            boxes: List of bounding boxes
            confidences: List of confidence scores
            classes: List of class IDs
            x_factor: X-axis scaling factor
            y_factor: Y-axis scaling factor
            
        Returns:
            Image with drawn bounding boxes
        """
        result_img = image.copy()
        
        for i in range(len(boxes)):
            box = boxes[i]
            left = int(box[0] * x_factor)
            top = int(box[1] * y_factor)
            width = int(box[2] * x_factor)
            height = int(box[3] * y_factor)
            
            cv2.rectangle(result_img, (left, top), (left + width, top + height), 
                         self.box_color, 2)
            label = f"{self.labels[classes[i]]}: {confidences[i]:.2f}"
            cv2.putText(result_img, label, (left, top - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,self.text_color, 2)

        return result_img
    
    def display(self, image: np.ndarray, window_name="Detection Result"):
        """Display image in window"""
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save(self, image: np.ndarray, output_path: str):
        """Save image to file"""
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")


class ImageLoader:
    """Class for loading images"""
    
    @staticmethod
    def load(file_path: str):
        """Load image from file"""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not load image from {file_path}")
        return img


class VideoProcessor:
    """Class for processing videos with object detection"""
    
    def __init__(self, detector: YOLODetector, visualizer: DetectionVisualizer):
        """
        Initialize video processor
        
        Args:
            detector: YOLODetector instance
            visualizer: DetectionVisualizer instance
        """
        self.detector = detector
        self.visualizer = visualizer
    
    def process_video(self, input_path: str, output_path: str = None, 
                     display: bool = True, fps_limit: int = None):
        """
        Process video file with object detection
        
        Args:
            input_path: Path to input video file (or 0 for webcam)
            output_path: Path to save output video (optional)
            display: Whether to display video while processing
            fps_limit: Limit processing to specific FPS (None for all frames)
        """
        if input_path == 0:
            cap = cv2.VideoCapture("http://100.115.92.22:8080/video")  # Webcam
            print("Using webcam...")
        else:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")
            print(f"Processing video: {input_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")
        if total_frames > 0:
            print(f"Total frames: {total_frames}")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                    (frame_width, frame_height))
            print(f"Saving output to: {output_path}")
        
        frame_skip = 1
        if fps_limit and fps > fps_limit:
            frame_skip = int(fps / fps_limit)
        
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % frame_skip != 0:
                    continue

                boxes, confidences, classes, x_factor, y_factor = self.detector.detect(frame)

                result_frame = self.visualizer.draw_boxes(
                    frame, boxes, confidences, classes, x_factor, y_factor
                )

                info_text = f"Frame: {frame_count} | Detections: {len(boxes)}"
                cv2.putText(result_frame, info_text, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),  2)
                
                if writer:
                    writer.write(result_frame)

                if display:
                    cv2.imshow("Video Detection", result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    elif key == ord('p'):
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)
                
                processed_count += 1
 
                if total_frames > 0 and processed_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\nProcessing complete!")
            print(f"Total frames processed: {processed_count}")
    
    def process_webcam(self, output_path: str = None, display: bool = True):
        """
        Process webcam stream with object detection
        
        Args:
            output_path: Path to save output video (optional)
            display: Whether to display video while processing
        """
        print("Starting webcam detection. Press 'q' to quit, 'p' to pause.")
        self.process_video(0, output_path=output_path, display=display)


def main():
    config = YOLOConfig(
        confidence_threshold=0.4,
        prob_threshold=0.25,
        nms_threshold=0.45,
        input_width=640,
        input_height=640
    )
    
    detector = YOLODetector(
        model_path="prediction_file/Model3-20251015T203658Z-1-001/Model3/weights/best.onnx",
        yaml_path="data.yaml",
        config=config
    )
    visualizer = DetectionVisualizer(labels=detector.labels)
    
    # Choose mode: 'image', 'video', or 'webcam'
    mode = 'video' 
    
    if mode == 'image':

        print("=== Image Detection Mode===")
        image = ImageLoader.load("2-Figure1-1.png")

        boxes, confidences, classes, x_factor, y_factor = detector.detect(image)
        if len(boxes) > 0:
            result_img = visualizer.draw_boxes(
                image, boxes, confidences, classes, x_factor, y_factor
            )
            visualizer.display(result_img)
            visualizer.save(result_img, "detection_result.png")
            
            print(f"Detected {len(boxes)} objects")
            for i, class_id in enumerate(classes):
                print(f"  - {detector.labels[class_id]}: {confidences[i]:.2f}")
        else:
            print("No objects detected")
    
    elif mode == 'video':
        print("=== Video Detection Mode  ===")
        video_processor = VideoProcessor(detector, visualizer)
        video_processor.process_video(
            input_path="videoplayback.mp4",
            output_path="output_video.avi",
            display=True,
            fps_limit=80
        )
    
    elif mode == 'webcam':
        print("==Webcam Detection Mode===")
        video_processor = VideoProcessor(detector, visualizer)
        video_processor.process_webcam(
            output_path="webcam_output.mp4",  # Set to None to not save
            display=True
        )
    
    else:
        print(f"Unknown mode: {mode}. Use 'image', 'video', or 'webcam'.")


if __name__ == "__main__":
    main()