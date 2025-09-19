
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

# Hand Gesture Control System

This project is a **hand gesture recognition and control system** built with **MediaPipe**, **PyQt6**, and optional **machine learning (SVM)**.  
It allows you to control mouse, keyboard, volume, brightness, and even launch programs using hand gestures detected by your webcam.

---

## 🚀 Features
- **Basic gestures:**
  - Mouse movement (index finger)
  - Left click (thumb + index pinch)
  - Right click (index + middle pinch)
  - Scroll (middle + ring pinch)
  - Volume/Brightness control (right hand)
- **Custom gestures:** Train and save your own gestures.
- **Machine Learning mode (SVM):** Train a model to recognize gestures and launch applications.

[![image2.jpg](https://i.postimg.cc/0Q3ZNrdW/image2.jpg)](https://postimg.cc/Fdy0Wr63)
---

## 📦 Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run
Run the application with:


---

## ⚙️ Notes
- A webcam is required.
- On Windows, additional permissions may be required for volume/brightness control.
- If you don’t need ML features, you can run without installing `scikit-learn` and `joblib`.

---

## 📂 Files
- **SystemControl.py** → Main application
- **requirements.txt** → Python dependencies
- **custom_gestures.json** → Saved custom gestures
- **sc_settings.json** → Saved settings
- **gesture_dataset.json / gesture_model.pkl / gesture_labels.json** → Used for ML training (created after training)

---

## 🖥️ Usage
1. Start the program with your webcam.
2. Enable/disable gestures from the **Basic Settings** tab.
3. Create custom gestures from the **Custom Gestures** tab.
4. (Optional) Train ML-based gestures for advanced recognition.

---

[![image1.png](https://i.postimg.cc/h4kpWzGh/image1.png)](https://postimg.cc/9zBGyfZj)

## 📌 Tested On
- Python 3.9+
- Windows 10/11

Linux and macOS should work, but brightness/volume controls may require adjustments.
