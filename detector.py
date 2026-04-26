import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.spatial import distance as dist
import os

class EBDSDetector:
    def __init__(self):
        # Initialize Face Landmarker
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Run download_model.py first.")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # EAR constants
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 15
        self.COUNTER = 0
        
        # Attendance tracking
        self.attendance_counter = 0
        self.attendance_threshold = 30 # frames of steady face
        self.attendance_logged = False
        
        # Landmark indices (Standard MediaPipe Mesh indices)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.NOSE_TIP = 1
        
        # HCI Settings
        self.screen_w, self.screen_h = 1920, 1080
        try:
            import pyautogui
            self.screen_w, self.screen_h = pyautogui.size()
        except:
            pass

    def calculate_ear(self, eye_landmarks):
        # eye_landmarks is a list of (x, y) tuples
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def process_frame(self, frame, mode='drowsiness'):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process frame
        detection_result = self.detector.detect(mp_image)
        
        status = "Normal"
        ear = 0
        
        if detection_result.face_landmarks:
            h, w, _ = frame.shape
            face_landmarks = detection_result.face_landmarks[0] # Take first face
            
            # Extract coordinates for eyes
            left_eye_coords = []
            for i in self.LEFT_EYE:
                lm = face_landmarks[i]
                left_eye_coords.append((lm.x * w, lm.y * h))
                
            right_eye_coords = []
            for i in self.RIGHT_EYE:
                lm = face_landmarks[i]
                right_eye_coords.append((lm.x * w, lm.y * h))
            
            # Calculate EAR
            left_ear = self.calculate_ear(left_eye_coords)
            right_ear = self.calculate_ear(right_eye_coords)
            ear = (left_ear + right_ear) / 2.0
            
            # Drowsiness logic
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    status = "Drowsy"
            else:
                self.COUNTER = 0
            
            # HCI logic (Nose tracking)
            if mode == 'hci':
                nose = face_landmarks[self.NOSE_TIP]
                # Map nose position to screen (mirrored)
                cursor_x = np.interp(nose.x, (0.3, 0.7), (self.screen_w, 0))
                cursor_y = np.interp(nose.y, (0.3, 0.7), (0, self.screen_h))
                try:
                    import pyautogui
                    pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
                except:
                    pass
                cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 5, (0, 0, 255), -1)

            # Attendance logic
            if mode == 'attendance':
                if not self.attendance_logged:
                    self.attendance_counter += 1
                    cv2.putText(frame, f"Logging: {int((self.attendance_counter/self.attendance_threshold)*100)}%", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    if self.attendance_counter >= self.attendance_threshold:
                        self.attendance_logged = True
                        status = "Attendance Logged"
                else:
                    status = "Attendance Logged"

            # Draw Mesh (simplified)
            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    
        return frame, status, ear
