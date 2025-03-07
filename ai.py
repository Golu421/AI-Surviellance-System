import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the YOLOv8n model for person detection and the custom model for gun detection
model_person = YOLO("yolov8n.pt")  # Detects only persons
model_gun = YOLO("best.pt")  # Detects only guns

# Define Classes
class_names_person = {0: "Person"}  # YOLOv8n detects only persons
class_names_gun = {0: "Gun"}  # Custom model detects only guns

# Variables for Polygon Zone
drawing = False
polygon_zone = []
first_frame_selected = False
polygon_alert = False
running_alert = False
loitering_alert = False
weapon_alert = False
person_positions = {}
entry_times = {}
LOITERING_THRESHOLD = 5  # Seconds before flagging loitering
RUNNING_SPEED_THRESHOLD = 50  # Pixels per second (approx)
frame_rate = 30  # Video frame rate

def draw_polygon(event, x, y, flags, param):
    global drawing, polygon_zone, first_frame_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_zone.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(polygon_zone) > 2:
        first_frame_selected = True
        print("Polygon selection complete.")

def is_inside_polygon(x1, y1, x2, y2):
    if not polygon_zone:
        return False
    object_points = np.array([
        [x1, y1], [x2, y1],
        [x2, y2], [x1, y2],
        [(x1 + x2) // 2, (y1 + y2) // 2]
    ], dtype=np.int32)
    poly = np.array(polygon_zone, np.int32).astype(np.float32)
    for point in object_points:
        if cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0:
            return True
    return False

def detect_objects(frame, confidence_threshold=0.2):
    global polygon_alert, running_alert, loitering_alert, weapon_alert
    polygon_alert = False
    running_alert = False
    loitering_alert = False
    weapon_alert = False
    
    current_time = time.time()
    detections = []

    # Detect persons
    results_person = model_person(frame)
    for result in results_person:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            class_name = class_names_person.get(int(cls), "Unknown")
            if conf > confidence_threshold and class_name == "Person":
                is_inside = is_inside_polygon(x1, y1, x2, y2)
                if is_inside:
                    polygon_alert = True
                    detections.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], conf, class_name))
                    obj_id = f"{int(x1)}-{int(y1)}"
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    if obj_id in person_positions:
                        prev_x, prev_y, prev_time = person_positions[obj_id]
                        time_diff = current_time - prev_time
                        if time_diff > 0:
                            distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
                            speed = distance / time_diff
                            if speed > RUNNING_SPEED_THRESHOLD:
                                running_alert = True
                    person_positions[obj_id] = (center_x, center_y, current_time)
                    if obj_id in entry_times:
                        if current_time - entry_times[obj_id] > LOITERING_THRESHOLD:
                            loitering_alert = True
                    else:
                        entry_times[obj_id] = current_time

    # Detect guns
    results_gun = model_gun(frame)
    for result in results_gun:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            class_name = class_names_gun.get(int(cls), "Unknown")
            if conf > confidence_threshold and class_name == "Gun":
                is_inside = is_inside_polygon(x1, y1, x2, y2)
                if is_inside:
                    polygon_alert = True
                    detections.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], conf, class_name))
                    weapon_alert = True

    return frame, detections

def main(video_path):
    global first_frame_selected
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Unable to read video")
        return
    cv2.namedWindow("Surveillance Output")
    cv2.setMouseCallback("Surveillance Output", draw_polygon)
    while not first_frame_selected:
        temp_frame = first_frame.copy()
        if len(polygon_zone) > 1:
            cv2.polylines(temp_frame, [np.array(polygon_zone)], False, (0, 255, 255), 2)
        cv2.imshow("Surveillance Output", temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    polygon_np = np.array(polygon_zone)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, detections = detect_objects(frame, confidence_threshold=0.2)
        polygon_color = (0, 0, 255) if polygon_alert else (0, 255, 255)
        cv2.polylines(frame, [polygon_np], True, polygon_color, 2)
        cv2.putText(frame, "Restricted Zone", (polygon_zone[0][0], polygon_zone[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, polygon_color, 2)
        if weapon_alert:
            cv2.putText(frame, "WEAPON DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if running_alert:
            cv2.putText(frame, "RUNNING DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if loitering_alert:
            cv2.putText(frame, "LOITERING DETECTED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        for det in detections:
            bbox, conf, class_name = det
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            color = (0, 0, 255) if class_name == "Gun" else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Surveillance Output", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "d.mp4"
    main(video_path)
