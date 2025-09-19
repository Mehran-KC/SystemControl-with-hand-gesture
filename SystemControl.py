import sys
import os
import time
import math
import json
import subprocess
import numpy as np

import cv2
import mediapipe as mp

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QSlider, QCheckBox, QMessageBox, QFrame, QLineEdit, QDialog,
    QGraphicsColorizeEffect,
    QDialogButtonBox, QComboBox, QProgressBar, QTabWidget, QSpacerItem, QSizePolicy,
    QTableWidget, QTableWidgetItem
)
from PyQt6.QtGui import (QColor, QImage, QPixmap)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from collections import deque, Counter

import pyautogui


GESTURE_FILE = "custom_gestures.json"




try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    joblib = None

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

try:
    import screen_brightness_control as sbc
    BRIGHTNESS_AVAILABLE = True
except Exception:
    BRIGHTNESS_AVAILABLE = False

# -----------------------
# Gesture ML Recognizer
# -----------------------
class GestureMLRecognizer:
    def __init__(self, dataset_path="gesture_dataset.json", model_path="gesture_model.pkl", label_map_path="gesture_labels.json", conf_threshold= 0.7):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.conf_threshold = float(conf_threshold)
        self.available = SKLEARN_AVAILABLE

        self.model = None
        self.samples = [] 
        self.label_to_program = {}

        self.pred_history = deque(maxlen=10)

        self._load_all()
        





    def _load_all(self):
        if os.path.exists(self.dataset_path):
            try:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    self.samples = json.load(f)
            except Exception:
                self.samples = []
        if os.path.exists(self.label_map_path):
            try:
                with open(self.label_map_path, "r", encoding="utf-8") as f:
                    self.label_to_program = json.load(f)
            except Exception:
                self.label_to_program = {}
        # model
        if SKLEARN_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def _save_dataset(self):
        try:
            with open(self.dataset_path, "w", encoding="utf-8") as f:
                json.dump(self.samples, f, ensure_ascii=False)
        except Exception:
            pass

    def _save_label_map(self):
        try:
            with open(self.label_map_path, "w", encoding="utf-8") as f:
                json.dump(self.label_to_program, f, ensure_ascii=False)
        except Exception:
            pass

    def _save_model(self):
        if SKLEARN_AVAILABLE and self.model is not None:
            try:
                joblib.dump(self.model, self.model_path)
            except Exception:
                pass

    def _extract_features_frame(self, hand_landmarks):
        """ویژگی 63بعدی (x,y,z برای 21 نقطه) نرمال‌شده نسبت به WRIST و اندازه دست."""
        if not hand_landmarks:
            return None
        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        wx, wy, wz = wrist.x, wrist.y, wrist.z

        max_d = 1e-6
        for lm in hand_landmarks.landmark:
            dx, dy = (lm.x - wx), (lm.y - wy)
            d = math.sqrt(dx*dx + dy*dy)
            if d > max_d:
                max_d = d

        feats = []
        for lm in hand_landmarks.landmark:
            feats.extend([
                (lm.x - wx) / max_d,
                (lm.y - wy) / max_d,
                (lm.z - wz) / (abs(wz) + 1e-6)
            ])
        return feats

    def _sequence_to_feature(self, landmarks_sequence):
        """میانگین ویژگی‌های فریم‌ها برای یک نمونهٔ آموزشی پایدارتر."""
        if not landmarks_sequence:
            return None
        arr = np.array(landmarks_sequence, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 63:
            mean_vec = np.mean(arr[:, :63], axis=0)
            return mean_vec.tolist()
        flat = arr.flatten()
        return flat[:63].tolist() if flat.size >= 63 else None

    def add_sequence(self, sequence, label, program=None):
        if not self.available:
            return False, "کتابخانه‌های ML نصب نیستند"

        try:
            if not sequence or not label:
                return False, "داده یا برچسب معتبر نیست"

            stable_frames = sequence
            print(f"[ML-DEBUG] total frames={len(sequence)}, stable={len(stable_frames)}")

            if not stable_frames or len(stable_frames) < 5:
                return False, "فریم‌های پایدار کافی نیست"

            feat = np.mean(stable_frames, axis=0).tolist()
            self.samples.append({"x": feat, "y": label})

            if label != "none" and program:
                self.label_to_program[label] = program

            self._save_dataset()
            self._save_label_map()

            return True, None
        except Exception as e:
            print("[ML] error adding sequence:", e)
            return False, str(e)



    def train(self):
        if not self.available:
            return False, "کتابخانه‌های ML نصب نیستند (scikit-learn, joblib)"
        try:
            X = np.array([s["x"] for s in self.samples if "y" in s], dtype=float)
            y = np.array([s["y"] for s in self.samples if "y" in s], dtype=object)

            labels = list(set(y))
            if len(labels) < 2:
                return False, f"کلاس‌های موجود: {labels} (حداقل ۲ کلاس لازم است)"

            clf = make_pipeline(
                StandardScaler(with_mean=False),
                SVC(kernel='rbf', probability=True, C=4.0, gamma='scale')
            )
            clf.fit(X, y)
            self.model = clf
            self._save_model()
            return True, None
        except Exception as e:
            print("[ML] error training:", e)
            return False, str(e)


    def is_ready(self):
        return self.available and (self.model is not None)

    def set_conf_threshold(self, v):
        try:
            self.conf_threshold = float(v)
        except Exception:
            pass

    def predict(self, hand_landmarks, threshold=None):
        if not self.model or not hand_landmarks:
            return None

        if threshold is None:
            threshold = self.conf_threshold

        feats = self._extract_features_frame(hand_landmarks)
        if feats is None:
            return None

        try:
            proba = self.model.predict_proba([feats])[0]
            results = {cls: float(p) for cls, p in zip(self.model.classes_, proba)}

            sorted_probs = sorted(results.items(), key=lambda x: x[1], reverse=True)
            best_label, best_p = sorted_probs[0]
            second_label, second_p = sorted_probs[1] if len(sorted_probs) > 1 else ("", 0.0)

            if (
                best_label == "none"
                or best_p < threshold
                or (best_p - second_p) < 0.15
            ):
                return None

            self.pred_history.append(best_label)

            if len(self.pred_history) == self.pred_history.maxlen:
                most_common, count = Counter(self.pred_history).most_common(1)[0]
                if count >= int(self.pred_history.maxlen * 0.7):
                    self.pred_history.clear()
                    return most_common, results

        except Exception as e:
            print("[ML] error predicting:", e)
            return None

        return None



    def remove_label(self, label):
        self.samples = [s for s in self.samples if s.get("y") != label]

        if label in self.label_to_program:
            del self.label_to_program[label]

        self._save_dataset()
        self._save_label_map()

        if len(set(s["y"] for s in self.samples)) >= 2:
            self.train()
        else:
            self.model = None

# -----------------------
# Custom statistical trainer (rule-based)
# -----------------------
class CustomGestureTrainer:
    def __init__(self):
        self.gesture_data = []
        self.recording = False
        self.gesture_name = ""
        self.target_program = ""
        self.landmarks_sequence = []

    def start_recording(self, gesture_name, target_program):
        self.gesture_name = gesture_name
        self.target_program = target_program
        self.landmarks_sequence = []
        self.recording = True

    def stop_recording(self):
        self.recording = False
        if len(self.landmarks_sequence) > 6:
            gesture_pattern = self.calculate_gesture_pattern()
            return {
                'name': self.gesture_name,
                'program': self.target_program,
                'pattern': gesture_pattern,
                'confidence_threshold': 0.75   
            }
        return None



    def add_landmarks(self, hand_landmarks):
        """اضافه کردن لندمارک‌ها با نرمال‌سازی نسبت به WRIST و اندازهٔ دست"""
        if self.recording and hand_landmarks:
            wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
            wx, wy, wz = wrist.x, wrist.y, wrist.z

            max_d = 1e-6
            for lm in hand_landmarks.landmark:
                dx, dy = lm.x - wx, lm.y - wy
                d = math.hypot(dx, dy)
                if d > max_d:
                    max_d = d

            landmarks_array = []
            for lm in hand_landmarks.landmark:
                landmarks_array.extend([
                    (lm.x - wx) / max_d,
                    (lm.y - wy) / max_d,
                    (lm.z - wz) / (abs(wz) + 1e-6)
                ])
            self.landmarks_sequence.append(landmarks_array)

    def calculate_gesture_pattern(self):
        if not self.landmarks_sequence:
            return None
        sequence_array = np.array(self.landmarks_sequence)
        mean_pattern = np.mean(sequence_array, axis=0)
        std_pattern = np.std(sequence_array, axis=0)
        return {
            'mean': mean_pattern.tolist(),
            'std': std_pattern.tolist(),
            'length': len(self.landmarks_sequence)
        }

    def recognize_gesture(self, hand_landmarks, custom_gestures):
        """تشخیص ژست با نرمال‌سازی + برگرداندن (gesture, confidence) یا None"""
        if not hand_landmarks or not custom_gestures:
            return None

        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        wx, wy, wz = wrist.x, wrist.y, wrist.z

        max_d = 1e-6
        for lm in hand_landmarks.landmark:
            dx, dy = lm.x - wx, lm.y - wy
            d = math.hypot(dx, dy)
            if d > max_d:
                max_d = d

        current_landmarks = []
        for lm in hand_landmarks.landmark:
            current_landmarks.extend([
                (lm.x - wx) / max_d,
                (lm.y - wy) / max_d,
                (lm.z - wz) / (abs(wz) + 1e-6)
            ])
        current_array = np.array(current_landmarks)

        best_match = None
        best_confidence = 0.0

        for gesture in custom_gestures:
            if 'pattern' not in gesture or not gesture['pattern']:
                continue
            pattern = gesture['pattern']
            mean_pattern = np.array(pattern['mean'])
            std_pattern = np.array(pattern['std'])

            diff = np.abs(current_array - mean_pattern)
            normalized_diff = diff / (std_pattern + 0.001)
            confidence = 1.0 - (np.mean(normalized_diff) / 10.0)
            confidence = max(0.0, min(1.0, confidence))

            if confidence > gesture.get('confidence_threshold', 0.75) and confidence > best_confidence:
                best_confidence = confidence
                best_match = gesture

        return (best_match, best_confidence) if best_match else None


# -----------------------
# CustomGestureDialog (PyQt)
# -----------------------
class CustomGestureDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("آموزش ژست سفارشی")
        self.setGeometry(200, 200, 520, 420)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("نام ژست:"))
        self.gesture_name_edit = QLineEdit()
        self.gesture_name_edit.setPlaceholderText("مثال: باز کردن CMD")
        layout.addWidget(self.gesture_name_edit)

        layout.addWidget(QLabel("برنامه مورد نظر:"))
        self.program_combo = QComboBox()
        self.program_combo.setEditable(True)
        default_programs = ["none", "cmd", "notepad", "calc", "mspaint", "explorer", "chrome", "firefox", "code", "spotify"]
        self.program_combo.addItems(default_programs)
        self.program_combo.setCurrentText("none")
        layout.addWidget(self.program_combo)

        layout.addWidget(QLabel("یا مسیر کامل برنامه:"))
        self.custom_path_edit = QLineEdit()
        self.custom_path_edit.setPlaceholderText("C:\\Path\\To\\Your\\Program.exe")
        layout.addWidget(self.custom_path_edit)

        instruction_label = QLabel(
            "دستورالعمل:\n"
            "1. نام ژست و برنامه را وارد کنید\n"
            "2. روی 'شروع ضبط' کلیک کنید\n"
            "3. منتظر شمارش معکوس 3 ثانیه‌ای بمانید\n"
            "4. ژست را چند بار در محدوده دوربین تکرار کنید\n"
            "5. روی 'توقف ضبط' کلیک کنید"
        )
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)

        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.countdown_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        self.record_btn = QPushButton("شروع ضبط")
        self.record_btn.clicked.connect(self.start_recording)
        button_layout.addWidget(self.record_btn)

        self.stop_btn = QPushButton("توقف ضبط")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        self.trainer = CustomGestureTrainer()
        self.recording = False

        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_value = 0

    def start_recording(self):
        gesture_name = self.gesture_name_edit.text().strip()
        program = self.custom_path_edit.text().strip() or self.program_combo.currentText()
        if not gesture_name or not program:
            QMessageBox.warning(self, "خطا", "لطفاً نام ژست و برنامه را وارد کنید.")
            return
        self.countdown_value = 3
        self.countdown_label.setText(str(self.countdown_value))
        self.countdown_timer.start(1000)
        self.record_btn.setEnabled(False)
        self.gesture_name = gesture_name
        self.program_name = program

    def update_countdown(self):
        self.countdown_value -= 1
        if self.countdown_value > 0:
            self.countdown_label.setText(str(self.countdown_value))
        else:
            self.countdown_timer.stop()
            self.countdown_label.setText("ضبط شروع شد...")
            self.trainer.start_recording(self.gesture_name, self.program_name)
            self.recording = True
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            if hasattr(self.parent(), 'processor'):
                self.parent().processor.custom_trainer = self.trainer
            QTimer.singleShot(2000, lambda: self.countdown_label.setText("در حال ضبط..."))

    def stop_recording(self):
        gesture_data = self.trainer.stop_recording()
        self.recording = False
        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if gesture_data:
            self.gesture_data = gesture_data
            gname = gesture_data['name']
            if gname == "none":
                QMessageBox.information(self, "موفقیت", "حالت عادی (none) ذخیره شد.")
            else:
                QMessageBox.information(self, "موفقیت", f"ژست '{gname}' با موفقیت ضبط شد.")
            try:
                proc = self.parent().processor if hasattr(self.parent(), 'processor') else None
                if proc and proc.ml_recognizer and proc.ml_recognizer.available:
                    ok, msg = proc.ml_recognizer.add_sequence(
                        self.trainer.landmarks_sequence, gname, self.program_name
                    )
                    if ok:
                        proc.ml_recognizer.train()
            except Exception as e:
                print("[ML] error adding sequence:", e)


    def get_gesture_data(self):
        return getattr(self, 'gesture_data', None)

# -----------------------
# HandGestureProcessor (core)
# -----------------------
class HandGestureProcessor(QObject):
    frame_processed = pyqtSignal(QImage)
    gesture_recognized = pyqtSignal(str)
    ml_probs_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.screen_width, self.screen_height = pyautogui.size()

        self.custom_recognition_counters = {}
        self.custom_recognition_required_frames = 3

        try:
            if AUDIO_AVAILABLE:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume_control = cast(interface, POINTER(IAudioEndpointVolume))
            else:
                self.volume_control = None
        except Exception:
            self.volume_control = None

        self.last_action_time = 0
        self.action_cooldown = 0.3
        self.is_playing = False
        self.brightness_mode = False
        self.current_brightness = 50
        self.mouse_smoothing = []
        self.is_clicking = False
        self.fixed_mouse_pos = None
        self.last_scroll_y = None
        self.scroll_sensitivity = 2

        self.settings = {
            "gestures": {
                "index_move": True,
                "left_click": True,
                "right_click": True,
                "scroll": True,
                "volume_brightness": True,
            },
            "pinch_threshold": 0.03,
            "custom_gestures_enabled": True,
            "ml_gestures_enabled": True,
            "ml_conf_threshold": 0.80,
        }

        self.custom_gestures = []
        self.custom_trainer = None
        self.gesture_trainer = CustomGestureTrainer()
        self.last_custom_gesture_time = 0
        self.custom_gesture_cooldown = 1.5

        self.ml_recognizer = GestureMLRecognizer(conf_threshold=self.settings["ml_conf_threshold"]) if SKLEARN_AVAILABLE else None

    def update_settings(self, settings):
        self.settings = settings
        if self.ml_recognizer:
            self.ml_recognizer.set_conf_threshold(settings.get("ml_conf_threshold", 0.80))

    def add_custom_gesture(self, gesture_data):
        self.custom_gestures.append(gesture_data)
        self.custom_recognition_counters[gesture_data['name']] = 0

    def remove_custom_gesture(self, gesture_name):
        self.custom_gestures = [g for g in self.custom_gestures if g['name'] != gesture_name]
        if gesture_name in self.custom_recognition_counters:
            del self.custom_recognition_counters[gesture_name]

    def save_custom_gestures(self):
        """ذخیره ژست‌های سفارشی در فایل JSON"""
        try:
            with open(GESTURE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.custom_gestures, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("خطا در ذخیره ژست‌ها:", e)

    def load_custom_gestures(self):
        """بارگذاری ژست‌های سفارشی از فایل JSON"""
        if os.path.exists(GESTURE_FILE):
            try:
                with open(GESTURE_FILE, "r", encoding="utf-8") as f:
                    self.custom_gestures = json.load(f)
                for g in self.custom_gestures:
                    self.custom_recognition_counters[g['name']] = 0
                print(f"{len(self.custom_gestures)} ژست سفارشی بارگذاری شد.")
            except Exception as e:
                print("خطا در بارگذاری ژست‌ها:", e)

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def smooth_mouse(self, x, y, factor=5):
        self.mouse_smoothing.append((x, y))
        if len(self.mouse_smoothing) > factor:
            self.mouse_smoothing.pop(0)
        sx = sum(p[0] for p in self.mouse_smoothing) / len(self.mouse_smoothing)
        sy = sum(p[1] for p in self.mouse_smoothing) / len(self.mouse_smoothing)
        return sx, sy

    def execute_program(self, program_path):
        try:
            if program_path == "none":
                return False
            if not ('/' in program_path or '\\' in program_path):
                subprocess.Popen(program_path, shell=True)
            else:
                subprocess.Popen(program_path)
            return True
        except Exception as e:
            print(f"Error launching {program_path}: {e}")
            return False

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        shape = frame.shape  # h, w, ch

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label 
                if self.custom_trainer and self.custom_trainer.recording:
                    self.custom_trainer.add_landmarks(hand_landmarks)

                now = time.time()
                # ML recognition
                if (self.settings.get("ml_gestures_enabled", True) and self.ml_recognizer and self.ml_recognizer.is_ready()
                        and now - self.last_custom_gesture_time > self.custom_gesture_cooldown):
                    pred = self.ml_recognizer.predict(hand_landmarks, threshold=self.settings.get("ml_conf_threshold", 0.8))
                    if pred is not None:
                        lbl, probs = pred
                        if lbl != "none":
                            program = self.ml_recognizer.label_to_program.get(lbl)
                            if program:
                                if self.execute_program(program):
                                    confidence = probs.get(lbl, 0.0)
                                    self.gesture_recognized.emit(f"اجرای ML: {lbl} ({int(confidence*100)}%)")
                                    self.last_custom_gesture_time = now

                        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                        self.ml_probs_updated.emit(dict(top3))
                    else:
                        self.ml_probs_updated.emit({})


                # statistical custom gestures
                if self.settings.get("custom_gestures_enabled", True) and self.custom_gestures:
                    res = self.gesture_trainer.recognize_gesture(hand_landmarks, self.custom_gestures)
                    if res:
                        gesture, conf = res
                        gname = gesture['name']
                        self.custom_recognition_counters[gname] = self.custom_recognition_counters.get(gname, 0) + 1
                        for g in self.custom_gestures:
                            if g['name'] != gname:
                                self.custom_recognition_counters[g['name']] = 0

                        if (self.custom_recognition_counters[gname] >= self.custom_recognition_required_frames
                                and time.time() - self.last_custom_gesture_time > self.custom_gesture_cooldown):
                            if self.execute_program(gesture['program']):
                                self.gesture_recognized.emit(f"اجرای: {gesture['name']} (conf={conf:.2f})")
                                self.last_custom_gesture_time = time.time()
                            self.custom_recognition_counters[gname] = 0
                    else:
                        for g in self.custom_gestures:
                            self.custom_recognition_counters[g['name']] = 0

                if label == "Right" and self.settings["gestures"]["volume_brightness"]:
                    self.handle_right_hand(hand_landmarks, shape, frame)
                elif label == "Left":
                    self.handle_left_hand(hand_landmarks, shape, frame)

                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        status_lines = [
            f"{'Playing' if self.is_playing else 'Paused'} | " + f"{'Brightness' if self.brightness_mode else 'Volume'}: {self.current_brightness}%"
        ]
        if self.ml_recognizer:
            cls_count = len(set([s.get('y') for s in self.ml_recognizer.samples])) if self.ml_recognizer.samples else 0
            status_lines.append(f"ML: {'Ready' if self.ml_recognizer.is_ready() else 'Not ready'} | samples={len(self.ml_recognizer.samples)}, classes={cls_count}")

        if self.custom_gestures:
            status_lines.append(f"Custom Gestures: {len(self.custom_gestures)} active")

        for i, line in enumerate(status_lines):
            cv2.putText(frame, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self.frame_processed.emit(qt_image)

    def handle_right_hand(self, hand_landmarks, shape, frame):
        threshold = self.settings.get("pinch_threshold", 0.03)
        mp_hands = self.mp_hands

        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        thumb_index_dist = math.hypot(thumb.x - index.x, thumb.y - index.y)
        ring_pinky_dist = math.hypot(ring.x - pinky.x, ring.y - pinky.y)

        now = time.time()
        if thumb_index_dist < threshold and now - self.last_action_time > self.action_cooldown:
            pyautogui.press("playpause")
            self.is_playing = not self.is_playing
            self.last_action_time = now

        if ring_pinky_dist < threshold and now - self.last_action_time > self.action_cooldown:
            self.brightness_mode = not self.brightness_mode
            self.last_action_time = now

        level_norm = 1 - wrist.y
        level_norm = min(max(level_norm, 0), 1)
        if self.brightness_mode:
            self.current_brightness = int(level_norm * 100)
            if BRIGHTNESS_AVAILABLE:
                try:
                    sbc.set_brightness(self.current_brightness)
                except Exception:
                    pass
        else:
            if self.volume_control:
                try:
                    self.volume_control.SetMasterVolumeLevelScalar(level_norm, None)
                    self.current_brightness = int(level_norm * 100)
                except Exception:
                    pass

    def handle_left_hand(self, hand_landmarks, shape, frame):
        threshold = self.settings.get("pinch_threshold", 0.03)
        mp_hands = self.mp_hands

        index_lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_lm = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        middle_lm = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_lm = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

        screen_x = int(index_lm.x * self.screen_width)
        screen_y = int(index_lm.y * self.screen_height)

        if self.settings["gestures"]["index_move"]:
            if self.is_clicking and self.fixed_mouse_pos:
                pyautogui.moveTo(*self.fixed_mouse_pos)
            else:
                sx, sy = self.smooth_mouse(screen_x, screen_y)
                pyautogui.moveTo(sx, sy)

        now = time.time()
        if self.settings["gestures"]["scroll"]:
            middle_ring_dist = math.hypot(middle_lm.x - ring_lm.x, middle_lm.y - ring_lm.y)
            if middle_ring_dist < threshold:
                if self.last_scroll_y is None:
                    self.last_scroll_y = index_lm.y
                else:
                    scroll_amount = int((self.last_scroll_y - index_lm.y) * self.scroll_sensitivity * 10)
                    if abs(scroll_amount) > 0:
                        pyautogui.scroll(scroll_amount)
                    self.last_scroll_y = index_lm.y
            else:
                self.last_scroll_y = None

        if self.settings["gestures"]["left_click"]:
            pinch_distance = math.hypot(thumb_lm.x - index_lm.x, thumb_lm.y - index_lm.y)
            if pinch_distance < threshold:
                if not self.is_clicking:
                    pyautogui.click()
                    self.is_clicking = True
                    self.fixed_mouse_pos = (screen_x, screen_y)
            else:
                self.is_clicking = False
                self.fixed_mouse_pos = None

        if self.settings["gestures"]["right_click"]:
            if math.hypot(index_lm.x - middle_lm.x, index_lm.y - middle_lm.y) < threshold and now - self.last_action_time > self.action_cooldown:
                pyautogui.rightClick()
                self.last_action_time = now

# -----------------------
# MainWindow (PyQt UI)
# -----------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("کنترل ژست دست - MediaPipe + ML")
        self.setGeometry(100, 100, 1100, 700)
        self.dark_mode = True


        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                color: #e0e0e0;
                font-family: Segoe UI, sans-serif;
                font-size: 11pt;
            }

            QLabel {
                color: #ffffff;
                font-size: 10pt;
            }

            /* ==== Card Style (QFrame) ==== */
            QFrame {
                background-color: #2a2a3b;
                border: 1px solid #444;
                border-radius: 12px;
                padding: 10px;
                margin: 8px;
            }

            /* ==== Buttons ==== */
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2c5f91;
            }

            /* ==== LineEdit, ComboBox, List, Slider ==== */
            QLineEdit, QComboBox, QListWidget, QTableWidget, QSpinBox {
                background-color: #2a2a3b;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 4px;
            }

            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #333;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: none;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }

            /* ==== Checkboxes ==== */
            QCheckBox {
                spacing: 8px;
                font-size: 10pt;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #888;
                border-radius: 4px;
                background: transparent;
            }
            QCheckBox::indicator:checked {
                background: #4a90e2;
                border: 2px solid #4a90e2;
                border-radius: 4px;
            }

            /* ==== Tabs ==== */
            QTabWidget::pane {
                border: 1px solid #444;
                background: #2a2a3b;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #333;
                color: #ccc;
                padding: 6px 12px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #4a90e2;
                color: white;
            }

            /* ==== List / Table ==== */
            QListWidget, QTableWidget {
                background-color: #2a2a3b;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 6px;
            }

            /* ==== ProgressBar ==== */
            QProgressBar {
                border: 1px solid #444;
                border-radius: 6px;
                background: #2a2a3b;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
                border-radius: 6px;
            }
    """)


        self.processor = HandGestureProcessor()
        self.processor.frame_processed.connect(self.update_image)
        self.processor.gesture_recognized.connect(self.on_gesture_recognized)
        self.processor.ml_probs_updated.connect(self.update_ml_probs)


        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        self.video_label = QLabel("در حال دریافت تصویر...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        main_layout.addWidget(self.video_label)

        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel)

        self.tabs = QTabWidget()
        right_panel.addWidget(self.tabs)

        self.create_basic_settings_tab()
        self.create_custom_gestures_tab()

        bottom_layout = QHBoxLayout()
        right_panel.addLayout(bottom_layout)

        self.status_label = QLabel("وضعیت: آماده")
        bottom_layout.addWidget(self.status_label)

        self.ml_status_frame = QFrame()
        ml_layout = QVBoxLayout(self.ml_status_frame)

        self.ml_ready_label = QLabel("وضعیت ML: Not Ready")
        ml_layout.addWidget(self.ml_ready_label)

        self.prob_bars = []
        for i in range(3):
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setFormat("")
            self.prob_bars.append(bar)
            ml_layout.addWidget(bar)

        right_panel.addWidget(self.ml_status_frame)


        bottom_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.start_btn = QPushButton("شروع/توقف دوربین")
        self.start_btn.setCheckable(True)
        self.start_btn.setChecked(True)
        self.start_btn.clicked.connect(self.toggle_camera)
        bottom_layout.addWidget(self.start_btn)

        self.timer = QTimer()
        self.timer.timeout.connect(self.processor.process_frame)
        self.timer.start(30)

        self.load_settings()
        self.update_ml_status()
        self.processor.load_custom_gestures()

        self.toggle_theme_btn = QPushButton("🌙 Dark/Light")
        self.toggle_theme_btn.clicked.connect(self.toggle_theme)
        right_panel.addWidget(self.toggle_theme_btn)

        self.gesture_table.clear()
        for g in self.processor.custom_gestures:
            self.add_gesture_row(g['name'], g['program'])


    def update_ml_probs(self, probs):
        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, bar in enumerate(self.prob_bars):
            if i < len(top3):
                label, val = top3[i]
                bar.setValue(int(val * 100))
                bar.setFormat(f"{label} ({int(val*100)}%)")
            else:
                bar.setValue(0)
                bar.setFormat("")


    def toggle_theme(self):
        if self.dark_mode:
            self.setStyleSheet("QWidget { background: #ffffff; color: #000000; }")
        else:
            self.setStyleSheet("QWidget { background: #1e1e2f; color: #e0e0e0; }")
        self.dark_mode = not self.dark_mode

    def create_basic_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        layout.addWidget(QLabel("ژست‌های پایه:"))
        self.checkbox_map = {}
        gestures = {
            "index_move": "حرکت موس (انگشت اشاره)",
            "left_click": "کلیک چپ (Pinch شست و اشاره)",
            "right_click": "کلیک راست (Pinch اشاره و وسطی)",
            "scroll": "اسکرول (Pinch میانی و حلقه)",
            "volume_brightness": "کنترل صدا/روشنایی (دست راست)"
        }
        for key, text in gestures.items():
            cb = QCheckBox(text)
            cb.setChecked(True)
            self.checkbox_map[key] = cb
            layout.addWidget(cb)

        layout.addSpacing(8)
        layout.addWidget(QLabel("آستانه Pinch (0.01..0.2):"))
        self.pinch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pinch_slider.setMinimum(1)
        self.pinch_slider.setMaximum(200)
        self.pinch_slider.setValue(int(self.processor.settings.get("pinch_threshold", 0.03)*1000))
        layout.addWidget(self.pinch_slider)
        self.pinch_slider.valueChanged.connect(self.on_pinch_changed)

        layout.addSpacing(8)
        self.custom_gestures_cb = QCheckBox("فعال‌سازی ژست‌های سفارشی (statistical)")
        self.custom_gestures_cb.setChecked(True)
        layout.addWidget(self.custom_gestures_cb)

        self.ml_gestures_cb = QCheckBox("فعال‌سازی ژست‌های ML (SVM)")
        self.ml_gestures_cb.setChecked(SKLEARN_AVAILABLE)
        layout.addWidget(self.ml_gestures_cb)

        layout.addWidget(QLabel("آستانه اطمینان ML (0.0 - 1.0):"))
        self.ml_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.ml_threshold_slider.setMinimum(50)
        self.ml_threshold_slider.setMaximum(100)
        self.ml_threshold_slider.setValue(int(self.processor.settings.get("ml_conf_threshold", 0.80)*100))
        layout.addWidget(self.ml_threshold_slider)

        save_btn = QPushButton("ذخیره تنظیمات")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)

        layout.addStretch()
        self.tabs.addTab(tab, "تنظیمات پایه")

    def highlight_row(self, row):
        """هایلایت کردن یک ردیف در جدول با تغییر رنگ پس‌زمینه برای چند لحظه"""
        if row < 0 or row >= self.gesture_table.rowCount():
            return

        for col in range(self.gesture_table.columnCount()):
            item = self.gesture_table.item(row, col)
            if item:
                item.setBackground(QColor("yellow"))

        def reset_color():
            for col in range(self.gesture_table.columnCount()):
                item = self.gesture_table.item(row, col)
                if item:
                    item.setBackground(QColor("black"))

        QTimer.singleShot(1000, reset_color)

    def create_custom_gestures_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        tab.setLayout(layout)

        btns_v = QVBoxLayout()

        add_btn = QPushButton("➕ افزودن ژست")
        add_btn.clicked.connect(self.add_gesture)
        btns_v.addWidget(add_btn)

        remove_btn = QPushButton("❌ حذف ژست انتخابی")
        remove_btn.clicked.connect(self.remove_selected_gesture)
        btns_v.addWidget(remove_btn)

        train_btn = QPushButton("🎯 آموزش دستی ML")
        train_btn.clicked.connect(self.train_ml)
        btns_v.addWidget(train_btn)

        self.ml_status_label = QLabel("ML: نامشخص")
        btns_v.addWidget(self.ml_status_label)

        btns_v.addStretch()

        right_panel = QVBoxLayout()

        self.gesture_table = QTableWidget()
        self.gesture_table.setColumnCount(3)
        self.gesture_table.setHorizontalHeaderLabels(["ژست", "برنامه", "حذف"])
        self.gesture_table.horizontalHeader().setStretchLastSection(True)
        self.gesture_table.setStyleSheet("QTableWidget { border: none; }")

        gestures_frame = QFrame()
        gestures_layout = QVBoxLayout(gestures_frame)
        gestures_layout.addWidget(QLabel("ژست‌های سفارشی"))
        gestures_layout.addWidget(self.gesture_table)

        right_panel.addWidget(gestures_frame)

        layout.addLayout(btns_v, 1)
        layout.addLayout(right_panel, 3)

        self.tabs.addTab(tab, "ژست‌های سفارشی")


    def add_gesture(self):
        dlg = CustomGestureDialog(self)
        if dlg.exec():
            gesture_data = dlg.get_gesture_data()
            if gesture_data:
                self.processor.add_custom_gesture(gesture_data)
                self.processor.save_custom_gestures()

                self.add_gesture_row(gesture_data["name"], gesture_data["program"])

                self.update_ml_status()
                self.highlight_row(self.gesture_table.rowCount() - 1)
    
    def add_gesture_row(self, gesture_name, program):
        row = self.gesture_table.rowCount()
        self.gesture_table.insertRow(row)
        self.gesture_table.setItem(row, 0, QTableWidgetItem(gesture_name))
        self.gesture_table.setItem(row, 1, QTableWidgetItem(program))

        btn = QPushButton("❌")
        btn.clicked.connect(lambda _, r=row: self.remove_gesture_row(r))
        self.gesture_table.setCellWidget(row, 2, btn)



    def remove_selected_gesture(self):
        row = self.gesture_table.currentRow()
        if row < 0:
            return
        self.remove_gesture_row(row)


    def remove_gesture_row(self, row):
        item = self.gesture_table.item(row, 0)
        if not item:
            return

        gesture_name = item.text()

        try:
            self.processor.remove_custom_gesture(gesture_name)
            self.processor.save_custom_gestures()
        except Exception as e:
            print("[UI] error removing from custom gestures:", e)

        try:
            if self.processor.ml_recognizer:
                self.processor.ml_recognizer.remove_label(gesture_name)
        except Exception as e:
            print("[UI] error removing from ML:", e)

        self.gesture_table.removeRow(row)

        self.update_ml_status()




    def train_ml(self):
        if not SKLEARN_AVAILABLE:
            QMessageBox.warning(self, "ML", "scikit-learn نصب نشده است. برای فعال کردن ML نصب کنید: pip install scikit-learn joblib")
            return
        if not self.processor.ml_recognizer:
            QMessageBox.warning(self, "ML", "شناسه ML در دسترس نیست.")
            return
        ok, msg = self.processor.ml_recognizer.train()
        if ok:
            QMessageBox.information(self, "ML", "آموزش با موفقیت انجام شد.")
        else:
            QMessageBox.warning(self, "ML", f"آموزش انجام نشد: {msg}")
        self.update_ml_status()

    def update_ml_status(self):
        ml = self.processor.ml_recognizer
        if ml and ml.samples:
            samples = len(ml.samples)
            classes = len(set(s["y"] for s in ml.samples if "y" in s))
        else:
            samples, classes = 0, 0

        ready = ml.is_ready()
        self.ml_status_label.setText(
            f"ML: {'Ready' if ready else 'Not ready'} | samples={samples} | classes={classes}"
        )


    def on_pinch_changed(self, val):
        v = max(1, val) / 1000.0
        self.processor.settings['pinch_threshold'] = v

    def save_settings(self):
        s = {
            "gestures": {k: cb.isChecked() for k, cb in self.checkbox_map.items()},
            "pinch_threshold": self.processor.settings.get('pinch_threshold', 0.03),
            "custom_gestures_enabled": self.custom_gestures_cb.isChecked(),
            "ml_gestures_enabled": self.ml_gestures_cb.isChecked(),
            "ml_conf_threshold": self.ml_threshold_slider.value() / 100.0
        }
        self.processor.update_settings(s)
        try:
            with open("sc_settings.json", "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False)
            QMessageBox.information(self, "تنظیمات", "تنظیمات ذخیره شد.")
        except Exception as e:
            QMessageBox.warning(self, "خطا", f"ذخیره تنظیمات ممکن نشد: {e}")
        self.update_ml_status()

    def load_settings(self):
        if os.path.exists("sc_settings.json"):
            try:
                with open("sc_settings.json", "r", encoding="utf-8") as f:
                    s = json.load(f)
                for k, cb in self.checkbox_map.items():
                    cb.setChecked(s.get("gestures", {}).get(k, True))
                self.custom_gestures_cb.setChecked(s.get("custom_gestures_enabled", True))
                self.ml_gestures_cb.setChecked(s.get("ml_gestures_enabled", SKLEARN_AVAILABLE))
                self.processor.update_settings(s)
                self.pinch_slider.setValue(int(s.get("pinch_threshold", 0.03)*1000))
                self.ml_threshold_slider.setValue(int(s.get("ml_conf_threshold", 0.80)*100))
            except Exception:
                pass

    def toggle_camera(self):
        if self.start_btn.isChecked():
            self.timer.start(30)
            self.start_btn.setText("توقف دوربین")
        else:
            self.timer.stop()
            self.start_btn.setText("شروع دوربین")

    def update_image(self, qimage):
        pix = QPixmap.fromImage(qimage).scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def on_gesture_recognized(self, text):
        self.status_label.setText(text)

    def closeEvent(self, event):
        try:
            self.timer.stop()
            if self.processor and self.processor.cap:
                self.processor.cap.release()
        except Exception:
            pass
        event.accept()

# -----------------------
# main
# -----------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
