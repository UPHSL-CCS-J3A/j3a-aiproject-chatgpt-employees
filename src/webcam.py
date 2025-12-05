import cv2 as cv
import numpy as np
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

capture_requested = False
force_update = False   # NEW: forces age/gender refresh

def draw_info_box(frame, text):
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    lines = text.split("\n")
    width = 0
    height = 0
    for line in lines:
        (w, h), _ = cv.getTextSize(line, font, scale, thickness)
        width = max(width, w)
        height += h + 5
    x2 = frame.shape[1] - 10
    y2 = frame.shape[0] - 10
    x1 = x2 - width - 20
    y1 = y2 - height - 20
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    y = y1 + 25
    for line in lines:
        cv.putText(frame, line, (x1 + 10, y), font, scale, (255, 255, 255), 1)
        y += 22

def draw_button(frame, text="Capture", pos=(10, 10)):
    x, y = pos
    w, h = 120, 40
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), -1)
    cv.putText(frame, text, (x + 10, y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return (x, y, x + w, y + h)

def on_mouse(event, x, y, flags, param):
    global capture_requested
    if event == cv.EVENT_LBUTTONDOWN:
        button_x1, button_y1, button_x2, button_y2 = param['button_coords']
        if button_x1 <= x <= button_x2 and button_y1 <= y <= button_y2:
            capture_requested = True

def main():
    global capture_requested, force_update

    # ------------------------------------------------------
    # CREATE UPDATE BUTTON (WINDOW FLOATING ON TOP RIGHT)
    # ------------------------------------------------------
    root = tk.Tk()
    root.title("Controls")
    root.geometry("200x80+1500+10")  # move window to top-right
    root.attributes('-topmost', True)

    def update_now():
        """Force update on next frame."""
        global force_update
        force_update = True

    btn = ttk.Button(root, text="UPDATE", command=update_now)
    btn.pack(pady=20)

    # ------------------------------------------------------

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "FaceAge")
    os.makedirs(desktop_path, exist_ok=True)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        root.withdraw()
        messagebox.showerror("Error", "Webcam not detected!")
        return

    ret, frame = cap.read()
    if not ret:
        root.withdraw()
        messagebox.showerror("Error", "Cannot read from webcam!")
        return

    h, w = frame.shape[:2]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

    # Face detector
    MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    face_detector = cv.FaceDetectorYN.create(
        model=MODEL_PATH,
        config="",
        input_size=(w, h),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )

    # Gender and Age models
    AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")
    AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
    GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")
    GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = [
        'Newborn (0-2)',
        'Toddler (3-6)',
        'Child (7-12)',
        'Teen (13-18)',
        'Young Adult (19-25)',
        'Adult (26-35)',
        'Middle Age (36-45)',
        'Mature (46-59)',
        'Senior (60-100)'
    ]
    genderList = ['Male', 'Female']

    age_net = cv.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

    padding = 20
    prev_detection_time = 0
    last_results = {}

    cv.namedWindow("FaceAge Webcam")

    while True:
        root.update()  # keep button responsive

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        face_detector.setInputSize((w, h))
        faces = face_detector.detect(frame)

        # limit to one face
        if faces[1] is not None and len(faces[1]) > 0:
            faces = (faces[0], faces[1][:1])

        current_time = time.time()

        if faces[1] is not None:
            for i, face in enumerate(faces[1]):
                x, y, w_box, h_box = map(int, face[:4])

                # crop
                face_crop = frame[max(0, y - padding):min(y + h_box + padding, frame.shape[0] - 1),
                                  max(0, x - padding):min(x + w_box + padding, frame.shape[1] - 1)]

                # ✔ Update when 1 sec passed OR UPDATE button pressed
                if (current_time - prev_detection_time >= 1) or force_update:
                    force_update = False

                    blob = cv.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                    # Gender
                    gender_net.setInput(blob)
                    genderPreds = gender_net.forward()
                    gender_idx = int(genderPreds[0].argmax())
                    gender = genderList[gender_idx]
                    gender_conf = float(genderPreds[0][gender_idx] * 100)

                    # Age
                    age_net.setInput(blob)
                    agePreds = age_net.forward()
                    probs = agePreds[0]

                    midpoints = [(0+2)/2, (3+5)/2, (6+12)/2, (13+19)/2, (20+25)/2, (26+35)/2,
                                 (36+45)/2, (46+59)/2, (60+100)/2]
                    estimated_age = sum(p*m for p, m in zip(probs, midpoints))

                    age_idx = int(np.argmax(probs))
                    age_label = ageList[age_idx]
                    age_conf = float(probs[age_idx] * 100)

                    last_results[i] = {
                        'gender': gender,
                        'gender_conf': gender_conf,
                        'age_label': age_label,
                        'age_conf': age_conf,
                        'estimated_age': estimated_age
                    }

                    prev_detection_time = current_time

        # draw face + info box
        if faces[1] is not None:
            for i, face in enumerate(faces[1]):
                x, y, w_box, h_box = map(int, face[:4])
                cv.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

                if i in last_results:
                    res = last_results[i]
                    label = f"{res['gender']}, {res['age_label']}"
                    cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    text = f"Gender Confidence: {res['gender_conf']:.2f}%\n" \
                           f"Age Confidence: {res['age_conf']:.2f}%\n" \
                           f"Exact Age: {res['estimated_age']:.2f} yrs"
                    draw_info_box(frame, text)

        # capture button
        button_coords = draw_button(frame)
        cv.setMouseCallback("FaceAge Webcam", on_mouse, {'button_coords': button_coords})

        if capture_requested:
            filename = os.path.join(desktop_path, f"capture_{int(time.time())}.png")
            cv.imwrite(filename, frame)
            print(f"Photo saved: {filename}")
            capture_requested = False

        cv.imshow("FaceAge Webcam", frame)
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
