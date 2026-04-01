import cv2
from utils.camera import Camera
from detection.object_detection import ObjectDetector
from detection.pose_detection import PoseDetector

# Initialize camera and detectors
cam = Camera()
detector = ObjectDetector()
pose_detector = PoseDetector()

# Allowed object classes (COCO index for YOLOv8n)
# Set to None to allow all classes, or list specific classes
allowed_classes = None  # e.g. [0, 1, 2, 3, 39, 67, 73, 56, 60]

# Label names (manual overrides for the map; other classes use model names)
label_map = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    39: "Bottle",
    67: "Phone",
    73: "Book",
    56: "Chair",
    60: "Table",
}

# Confidence threshold
CONF_THRESHOLD = 0.5

while True:
    frame = cam.get_frame()

    if frame is None:
        break

    # 🔥 Pose Detection (NEW)
    pose_result = pose_detector.detect(frame)
    frame = pose_detector.draw(frame, pose_result)

    # Run object detection
    results = detector.detect(frame)

    # Draw fixed header label
    cv2.putText(
        frame,
        "AI SURVEILLANCE SYSTEM",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Process detections
    model_names = results[0].names if hasattr(results[0], "names") else label_map
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        if allowed_classes is not None and cls_id not in allowed_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label_name = label_map.get(cls_id, model_names.get(cls_id, str(cls_id)))
        label_text = f"{label_name} {conf:.2f}"

        # Color coding for common objects
        color_map = {
            0: (0, 255, 0),
            1: (255, 200, 0),
            2: (0, 0, 255),
            3: (0, 165, 255),
            39: (255, 0, 255),
            67: (255, 0, 0),
            73: (0, 255, 255),
            56: (0, 165, 255),
            60: (255, 255, 0),
        }
        color = color_map.get(cls_id, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label_text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # Show frame
    cv2.imshow("AI Surveillance System", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
