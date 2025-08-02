# app/services/recognition_service.py

import tensorflow as tf
import numpy as np
import cv2
import time
import mediapipe as mp

CLASS_NAMES = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u','v','unknown', 'w', 'x', 'y', 'z'
]

class SignRecognizer:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.last_success_time = 0
        self.COOLDOWN_SEC = 0.5              # Thời gian nghỉ giữa các ký tự
        self.READINESS_DELAY_SEC = 0.7       # Thời gian chờ trước khi chấp nhận ký tự tiếp theo
        self.CONFIDENCE_THRESHOLD = 0.9

    def predict(self, frame: np.ndarray) -> (str, np.ndarray):
        predicted_char = ""
        current_time = time.time()
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        frame_with_landmarks = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(frame_with_landmarks, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # --- LOGIC thời gian nghỉ và chờ ---
            if (current_time - self.last_success_time) > (self.COOLDOWN_SEC + self.READINESS_DELAY_SEC):
                landmarks_row = [lm for lm in hand_landmarks.landmark]
                wrist = landmarks_row[0]
                relative_landmarks = []
                for lm in landmarks_row:
                    relative_landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

                input_data = np.array([relative_landmarks], dtype=np.float32)
                predictions = self.model.predict(input_data, verbose=0)[0]
                score = np.max(predictions)
                
                if score >= self.CONFIDENCE_THRESHOLD:
                    class_index = np.argmax(predictions)
                    predicted_char = CLASS_NAMES[class_index]
                    self.last_success_time = current_time

        return predicted_char, frame_with_landmarks


class SentenceBuilder:
    def __init__(self, space_threshold_sec=1.5, backspace_char='unknown'):  # Đặt khoảng cách từ là 1.0 giây
        self.sentence = []
        self.last_char_time = time.time()
        self.SPACE_THRESHOLD_SEC = space_threshold_sec
        self.last_added_char = None
        self.BACKSPACE_CHAR = backspace_char

    def add_char(self, new_char: str):
        current_time = time.time()
        char_to_add = None

        is_valid_char = new_char and new_char != self.BACKSPACE_CHAR
        
        if is_valid_char and new_char != self.last_added_char:
            time_since_last_char = current_time - self.last_char_time
            if self.sentence and self.sentence[-1] != ' ' and time_since_last_char > self.SPACE_THRESHOLD_SEC:
                self.sentence.append(' ')
            
            char_to_add = new_char.upper()
            self.sentence.append(char_to_add)
            self.last_char_time = current_time
            self.last_added_char = new_char

        elif not is_valid_char:
            self.last_added_char = None
        
        return char_to_add is not None

    def get_sentence(self) -> str:
        return "".join(self.sentence)
