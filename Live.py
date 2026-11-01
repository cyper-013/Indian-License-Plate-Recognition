import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import easyocr
import time
import csv
import datetime
import os

class LPR:
    def __init__(self, output_csv="detected_plates.csv"):
        # Initialize models
        self.vehicle_model = YOLO('yolov8n.pt')
        self.plate_model = YOLO('license_plate_detector.pt')
        self.tracker = Sort()
        self.reader = easyocr.Reader(['en'])
        
        # Vehicle classes to track
        self.vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # License plate storage
        self.plate_data = {}
        
        # CSV logging setup
        self.output_csv = output_csv
        self._init_csv_file()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

    def _init_csv_file(self):
        #Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 
                    'License Plate', 
                    'Confidence',
                    'Vehicle BBox', 
                    'Plate BBox',
                    'Frame Number'
                ])
    
    def _log_to_csv(self, plate_data):
        #Log detected plate information to CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        with open(self.output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                plate_data['text'],
                plate_data['score'],
                str(plate_data['car_bbox']),
                str(plate_data['bbox']),
                self.frame_count
            ])

    def validate_indian_plate(self, text):
        """Validate if text matches Indian plate format"""
        text = text.upper().replace(" ", "")
        if len(text) != 10:
            return False
        if not text[:2].isalpha():  # State code
            return False
        if not text[2:4].isdigit():  # District number
            return False
        if not text[4:6].isalpha():  # Series
            return False
        if not text[6:].isdigit():  # Number
            return False
        return True

    def format_indian_plate(self, text):
        #Format detected text to standard Indian plate format
        text = text.upper().replace(" ", "")
        if len(text) >= 4:
            text = f"{text[:2]} {text[2:4]} {text[4:6]} {text[6:10]}"
        return text.strip()

    def get_car(self, license_plate, vehicle_track_ids):
        #Assign license plate to specific vehicle
        x1, y1, x2, y2, score, _ = license_plate
        for vehicle in vehicle_track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = vehicle
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                return xcar1, ycar1, xcar2, ycar2, car_id
        return -1, -1, -1, -1, -1

    def process_frame(self, frame):
        self.frame_count += 1
        height, width = frame.shape[:2]

        # Vehicle detection
        vehicles = []
        detections = self.vehicle_model(frame)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles and score > 0.3:
                vehicles.append([x1, y1, x2, y2, score])

        # Vehicle tracking
        track_ids = self.tracker.update(np.array(vehicles)) if vehicles else np.empty((0, 5))

        # License plate detection
        lp_detections = self.plate_model(frame)[0]
        for lp in lp_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = lp
            if score < 0.25:  # Skip low-confidence detections
                continue

            # Find associated vehicle
            xcar1, ycar1, xcar2, ycar2, car_id = self.get_car(lp, track_ids)
            if car_id == -1:
                continue

            # Crop and preprocess license plate
            lp_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if lp_crop.size == 0:
                continue

            # Enhanced preprocessing
            lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
            lp_blur = cv2.GaussianBlur(lp_gray, (3, 3), 0)
            _, lp_thresh = cv2.threshold(lp_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # OCR reading
            ocr_result = self.reader.readtext(lp_thresh, detail=0)
            if ocr_result:
                raw_text = "".join(ocr_result).upper()
                plate_text = self.format_indian_plate(raw_text)
                
                if self.validate_indian_plate(plate_text.replace(" ", "")):
                    # Store best reading per vehicle
                    current_best = self.plate_data.get(car_id, {'score': 0})
                    if score > current_best['score']:
                        plate_info = {
                            'text': plate_text,
                            'score': score,
                            'bbox': [x1, y1, x2, y2],
                            'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                            'last_seen': self.frame_count
                        }
                        self.plate_data[car_id] = plate_info
                        print(f"Valid Indian Plate: {plate_text}")
                        
                        # Log to CSV
                        self._log_to_csv(plate_info)

        # Visualization
        self.draw_results(frame)
        
        # Display FPS
        fps = self.frame_count / (time.time() - self.start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

    def draw_results(self, frame):
        # Remove stale entries
        stale_ids = [cid for cid, data in self.plate_data.items() 
                    if self.frame_count - data['last_seen'] > 30]
        for cid in stale_ids:
            del self.plate_data[cid]

        # Draw current detections
        for car_id, data in self.plate_data.items():
            # Vehicle box (green)
            x1, y1, x2, y2 = map(int, data['car_bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # License plate box (red)
            lx1, ly1, lx2, ly2 = map(int, data['bbox'])
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
            
            # License plate text
            text = data['text']
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(frame, (x1, y1 - text_h - 30), (x1 + text_w + 20, y1 - 10), (255, 144, 30), -1)
            cv2.putText(frame, text, (x1 + 10, y1 - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

def main():
    lpr = LPR(output_csv="indian_plates_log.csv")  
    cap = cv2.VideoCapture(0)  
    
    # resolution for Indian plates
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("Starting Indian License Plate Recognition. Press 'q' to quit.")
    print(f"Logging detected plates to: {lpr.output_csv}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = lpr.process_frame(frame)
        cv2.imshow('Indian LPR', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection completed. Results saved to {lpr.output_csv}")

if __name__ == "__main__":
    main()