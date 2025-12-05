import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

class GenderAgeDetectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceAge")
        self.root.geometry("900x620")
        self.root.configure(bg="#1e1e1e")  # Dark background

        self.style = ttk.Style()
        self.configure_style()

        # Load models
        self.load_models()

        # Create UI
        self.create_widgets()



    def configure_style(self):
        self.style.theme_use("clam")

        # Buttons
        self.style.configure("TButton",
                             font=("Segoe UI", 11, "bold"),
                             padding=10,
                             relief="flat",
                             background="#3b82f6",
                             foreground="white"
                             )
        self.style.map("TButton",
                       background=[("active", "#2563eb")],
                       foreground=[("active", "white")]
                       )

        # Labels
        self.style.configure("TLabel",
                             background="#1e1e1e",
                             foreground="white",
                             font=("Segoe UI", 11)
                             )

        # Frames
        self.style.configure("Modern.TFrame",
                             background="#1e1e1e"
                             )

    def save_results(self):
        try:
            # Ask for a file path
            file_path = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")]
            )
            if not file_path:
                return  # User cancelled

            # Collect results from the labels
            age_text = self.results_label.cget("text")
            skin_text = self.info_label.cget("text")

            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("=== Age Detection Results ===\n")
                f.write(age_text + "\n\n")
                f.write("=== Skin Condition Results ===\n")
                f.write(skin_text + "\n")

            messagebox.showinfo("Saved", f"Results saved to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def load_models(self):
        try:
            faceProto = "../models/opencv_face_detector.pbtxt"
            faceModel = "../models/opencv_face_detector_uint8.pb"
            ageProto = "../models/age_deploy.prototxt"
            ageModel = "../models/age_net.caffemodel"
            genderProto = "../models/gender_deploy.prototxt"
            genderModel = "../models/gender_net.caffemodel"

            self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

            self.ageList = [
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
            self.genderList = ['Male', 'Female']

            self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
            self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
            self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")

    def create_widgets(self):
        # Title -------------------------------------
        title = ttk.Label(
            self.root,
            text="FaceAge: Age & Skin Condition Detector",
            font=("Segoe UI", 24, "bold"),
            foreground="#60a5fa",
            background="#1e1e1e"
        )
        title.pack(pady=20)

        def save_results(self):
            try:
                # Ask for a file path
                file_path = filedialog.asksaveasfilename(
                    title="Save Results",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt")]
                )
                if not file_path:
                    return  # User cancelled

                # Collect results from the labels
                age_text = self.results_label.cget("text")
                skin_text = self.info_label.cget("text")

                # Write to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("=== Age Detection Results ===\n")
                    f.write(age_text + "\n\n")
                    f.write("=== Skin Condition Results ===\n")
                    f.write(skin_text + "\n")

                messagebox.showinfo("Saved", f"Results saved to {file_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

        # Button Bar --------------------------------
        button_frame = ttk.Frame(self.root, style="Modern.TFrame")
        button_frame.pack(pady=5)

        ttk.Button(button_frame, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Open Webcam", command=self.open_webcam).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=10)

        def save_results(self):
            try:
                # Ask for a file path
                file_path = filedialog.asksaveasfilename(
                    title="Save Results",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt")]
                )
                if not file_path:
                    return  # User cancelled

                # Collect results from the labels
                age_text = self.results_label.cget("text")
                skin_text = self.info_label.cget("text")

                # Write to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("=== Age Detection Results ===\n")
                    f.write(age_text + "\n\n")
                    f.write("=== Skin Condition Results ===\n")
                    f.write(skin_text + "\n")

                messagebox.showinfo("Saved", f"Results saved to {file_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

        # Main content container
        main_content = ttk.Frame(self.root, style="Modern.TFrame")
        main_content.pack(padx=10, pady=15, fill=tk.BOTH, expand=True)

        # Image Display Frame
        self.image_frame = tk.Frame(main_content, bg="#252525", bd=3, relief=tk.RIDGE, width=400, height=500)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            self.image_frame,
            text="No Image Loaded",
            font=("Segoe UI", 14),
            fg="white",
            bg="#252525"
        )
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Results & Info Frame (beside the image)
        results_container = ttk.Frame(main_content, style="Modern.TFrame")
        results_container.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        self.results_label = tk.Label(
            results_container,
            text="Age Detection will display here",
            bg="#111827",
            fg="#e5e7eb",
            font=("Segoe UI", 12),
            relief=tk.FLAT,
            height=5,
            width=40,
            anchor="nw",
            justify="left",
            padx=5,
            pady=5
        )
        self.results_label.pack(pady=(0, 10), fill=tk.X)

        self.info_label = tk.Label(
            results_container,
            text="Skin Condition Detection will display here",
            bg="#1f2937",
            fg="#f3f4f6",
            font=("Segoe UI", 12),
            relief=tk.FLAT,
            height=5,
            width=40,
            anchor="nw",
            justify="left",
            padx=5,
            pady=5
        )
        self.info_label.pack(fill=tk.X)

        # Sample Images -------------------------------
        sample_frame = ttk.Frame(self.root, style="Modern.TFrame")
        sample_frame.pack(pady=5)

        ttk.Label(sample_frame, text="Sample Images:", font=("Segoe UI", 11, "bold")).pack()

        btn_row = ttk.Frame(sample_frame, style="Modern.TFrame")
        btn_row.pack(pady=5)

        samples = [
            '../sample_images/girl1.jpg',
            '../sample_images/man1.jpg',
            '../sample_images/kid1.jpg',
            '../sample_images/woman1.jpg'
        ]

        for sample in samples:
            if os.path.exists(sample):
                ttk.Button(btn_row, text=os.path.basename(sample),
                           command=lambda s=sample: self.load_sample(s)).pack(side=tk.LEFT, padx=5)

    def highlightFace(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return frameOpencvDnn, faceBoxes

    def detect_age_gender(self, image_path):
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return None, "Could not read image"

            h, w = frame.shape[:2]

            # Load Face Detector (YuNet)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "..", "models")
            MODEL_PATH = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")

            face_detector = cv2.FaceDetectorYN.create(
                model=MODEL_PATH,
                config="",
                input_size=(w, h),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )

            faces = face_detector.detect(frame)

            if faces[1] is None:
                return frame, "No face detected"

            results = []
            age_ranges = [(0, 2), (3, 5), (6, 12), (13, 19), (20, 25),
                          (26, 35), (36, 45), (46, 59), (60, 100)]
            midpoints = [(a + b) / 2 for (a, b) in age_ranges]

            for face in faces[1]:
                x, y, w_box, h_box = map(int, face[:4])
                padding = 20
                face_crop = frame[
                            max(0, y - padding):min(y + h_box + padding, frame.shape[0] - 1),
                            max(0, x - padding):min(x + w_box + padding, frame.shape[1] - 1)
                            ]

                blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227),
                                             self.MODEL_MEAN_VALUES, swapRB=False)

                # Gender
                self.genderNet.setInput(blob)
                genderPreds = self.genderNet.forward()
                gender_idx = int(genderPreds[0].argmax())
                gender = self.genderList[gender_idx]
                gender_conf = float(genderPreds[0][gender_idx] * 100)

                # Age
                self.ageNet.setInput(blob)
                agePreds = self.ageNet.forward()
                probs = agePreds[0]
                estimated_age = sum(p * m for p, m in zip(probs, midpoints))
                age_idx = int(np.argmax(probs))
                age_label = self.ageList[age_idx]
                age_conf = float(probs[age_idx] * 100)

                # Draw on image
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(frame, f"{gender}, {age_label}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                results.append(
                    f"Estimated Age: {estimated_age:.2f} years\n"
                    f"Age Group: {gender} {age_label}\n"
                    f"Confidence Level: {age_conf:.2f}%\n"
                )

            return frame, results

        except Exception as e:
            return None, f"Error: {str(e)}"

    def open_webcam(self):
        import subprocess
        import sys

        try:
            # Close main window
            self.root.destroy()

            # Run webcam.py
            base_dir = os.path.dirname(os.path.abspath(__file__))
            webcam_path = os.path.join(base_dir, "webcam.py")
            subprocess.call([sys.executable, webcam_path])

            # After webcam closes → reopen main window
            new_root = tk.Tk()
            app = GenderAgeDetectorUI(new_root)
            new_root.mainloop()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open webcam: {str(e)}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if file_path:
            self.process_image(file_path)

    def load_sample(self, sample_name):
        if os.path.exists(sample_name):
            self.process_image(sample_name)
        else:
            messagebox.showerror("Error", f"Sample image {sample_name} not found")

    def process_image(self, image_path):
        try:
            # Detect age and gender
            result_img, results = self.detect_age_gender(image_path)

            if result_img is not None:
                # Convert BGR to RGB for display
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                # Resize image for display
                height, width = result_img_rgb.shape[:2]
                max_height = 300
                if height > max_height:
                    ratio = max_height / height
                    new_width = int(width * ratio)
                    result_img_rgb = cv2.resize(result_img_rgb, (new_width, max_height))

                # Convert to PIL and display
                pil_image = Image.fromarray(result_img_rgb)
                photo = ImageTk.PhotoImage(pil_image)

                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo

                # Display results
                if isinstance(results, list):
                    result_text = "\n".join(results)
                else:
                    result_text = results

                self.results_label.configure(text=result_text)
            else:
                self.results_label.configure(text=results)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderAgeDetectorUI(root)
    root.mainloop()
