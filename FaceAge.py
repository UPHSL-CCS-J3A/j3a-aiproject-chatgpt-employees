"""
FaceAge: Enhanced Facial Age Estimation using Image Processing
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import time
import threading

class FaceAgeDetectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceAge")
        self.root.geometry("1000x850")
        self.root.configure(bg="#f8f9ff")
        
        self.webcam_active = False
        self.cap = None
        self.webcam_thread = None
        self.show_diversity = False

        self.style = ttk.Style()
        self.configure_style()
        self.load_models()
        self.create_widgets()

    def configure_style(self):
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=10, relief="flat", background="#a7c7e7", foreground="#2c3e50")
        self.style.map("TButton", background=[("active", "#87ceeb")], foreground=[("active", "#2c3e50")])
        self.style.configure("TLabel", background="#f8f9ff", foreground="#2c3e50", font=("Segoe UI", 11))
        self.style.configure("Modern.TFrame", background="#f8f9ff")

    # Load the AI models for gender and age prediction
    def load_models(self):
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "models")

            # Normalize image input before prediction
            self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            self.ageList = ['Newborn (0-2)', 'Toddler (3-6)', 'Child (7-12)', 'Teen (13-18)', 'Young Adult (19-25)', 'Adult (26-35)', 'Middle Age (36-45)', 'Mature (46-59)', 'Senior (60-100)']
            self.genderList = ['Male', 'Female']

            age_proto = os.path.join(models_dir, "age_deploy.prototxt")
            age_model = os.path.join(models_dir, "age_net.caffemodel")
            gender_proto = os.path.join(models_dir, "gender_deploy.prototxt")
            gender_model = os.path.join(models_dir, "gender_net.caffemodel")
            # Load the models into memory
            self.ageNet = cv2.dnn.readNet(age_model, age_proto)
            self.genderNet = cv2.dnn.readNet(gender_model, gender_proto)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")

    # Creates elements for the UI
    def create_widgets(self):
        # Title
        title = ttk.Label(self.root, text="FaceAge: Age & Skin Condition Detector", font=("Segoe UI", 24, "bold"), foreground="#6b73ff", background="#f8f9ff")
        title.pack(pady=20)

        # Button Bar
        button_frame = ttk.Frame(self.root, style="Modern.TFrame")
        button_frame.pack(pady=5)

        ttk.Button(button_frame, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Start Webcam", command=self.toggle_webcam).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=10)

        # Main content
        main_content = ttk.Frame(self.root, style="Modern.TFrame")
        main_content.pack(padx=10, pady=15, fill=tk.BOTH, expand=True)

        # Image Display
        self.image_frame = tk.Frame(main_content, bg="#e8f4f8", bd=3, relief=tk.RIDGE, width=500, height=400)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, text="No Image Loaded", font=("Segoe UI", 14), fg="#5a6c7d", bg="#e8f4f8")
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Results Panel
        results_container = ttk.Frame(main_content, style="Modern.TFrame")
        results_container.pack(side=tk.LEFT, padx=10, fill=tk.BOTH)

        self.results_label = tk.Label(results_container, text="Age Detection will display here", bg="#ffeaa7", fg="#2c3e50", font=("Segoe UI", 12), relief=tk.FLAT, height=8, width=40, anchor="nw", justify="left", padx=5, pady=5)
        self.results_label.pack(pady=(0, 10), fill=tk.BOTH, expand=True)

        # Info panel with button
        info_panel = tk.Frame(results_container, bg="#dda0dd")
        info_panel.pack(fill=tk.BOTH, expand=True)

        # Diversity analysis button in the pink panel
        self.diversity_btn = tk.Button(info_panel, text="Show Diversity Analysis", font=("Segoe UI", 10, "bold"),
                                      bg="#c48cc4", fg="#2c3e50", relief=tk.FLAT,
                                      command=self.toggle_diversity_display)
        self.diversity_btn.pack(pady=5)

        self.info_label = tk.Label(info_panel, text="", bg="#dda0dd", fg="#2c3e50", font=("Segoe UI", 12), relief=tk.FLAT, height=6, width=40, anchor="nw", justify="left", padx=5, pady=5)
        self.info_label.pack(fill=tk.BOTH, expand=True)

        # Sample Images
        sample_frame = ttk.Frame(self.root, style="Modern.TFrame")
        sample_frame.pack(pady=5)

        ttk.Label(sample_frame, text="Sample Images:", font=("Segoe UI", 11, "bold")).pack()

        btn_row = ttk.Frame(sample_frame, style="Modern.TFrame")
        btn_row.pack(pady=5)

        samples = ['sample_images/girl1.jpg', 'sample_images/man1.jpg', 'sample_images/kid1.jpg', 'sample_images/woman1.jpg']

        for sample in samples:
            if os.path.exists(sample):
                clean_name = os.path.basename(sample).replace('.jpg', '').replace('1', '').capitalize()
                ttk.Button(btn_row, text=clean_name, command=lambda s=sample: self.load_sample(s)).pack(side=tk.LEFT, padx=5)

    # Improves the inputs for different skin tones, ethnicities, and lighting conditions
    def preprocess_for_diversity(self, frame):
        # Helps normalize lighting so it c
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Helps normalize lighting so it sees the facial features
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)

        # Merge back the image and convert to BGR so model can understand
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Gamma correction for different lighting conditions
        gamma = self.estimate_gamma(frame) # If image is too dark or bright
        enhanced = self.adjust_gamma(enhanced, gamma) # Brightens or darkens the image

        return enhanced

    # How bright the image is
    def estimate_gamma(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to B&W
        mean_brightness = np.mean(gray)

        if mean_brightness < 80:  # Dark image
            return 0.7
        elif mean_brightness > 180:  # Bright image
            return 1.3
        else:  # Normal lighting
            return 1.0

    # Adjust the brightness based on the gamma value
    def adjust_gamma(self, frame, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)

    def analyze_diversity_features(self, frame, face_crop):
        # Estimate ethnicity based on facial features and skin tone
        ethnicity = self.estimate_ethnicity(face_crop)

        # Analyze skin tone using ITA (Individual Typology Angle)
        skin_tone = self.analyze_skin_tone(face_crop)

        # Detect skin condition
        skin_condition = self.detect_skin_condition(face_crop)

        # Assess lighting conditions
        lighting = self.assess_lighting(frame)

        return {
            'ethnicity': ethnicity,
            'skin_tone': skin_tone,
            'skin_condition': skin_condition,
            'lighting': lighting
        }

    def estimate_ethnicity(self, face_crop):
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)

        # Analyze skin tone in face region
        h, s, v = cv2.split(hsv)
        l, a, b = cv2.split(lab)

        # Calculate average values
        avg_hue = np.mean(h)
        avg_lightness = np.mean(l)
        avg_a = np.mean(a)
        avg_b = np.mean(b)

        # Simple ethnicity estimation based on skin tone characteristics
        if avg_lightness > 160 and avg_a < 135:
            return "Caucasian"
        elif avg_lightness < 100 and avg_a > 130:
            return "African"
        elif avg_hue > 10 and avg_hue < 25 and avg_lightness > 120:
            return "Asian"
        elif avg_b > 135 and avg_lightness > 110:
            return "Hispanic/Latino"
        elif avg_lightness > 130 and avg_a > 125:
            return "Middle Eastern"
        else:
            return "Mixed/Other"

    def analyze_skin_tone(self, face_crop):
        # skin tone using small cheek pixel samples
        h, w = face_crop.shape[:2]

        # Very specific small cheek areas (10x10 pixel patches)
        patch_size = 10

        # Left cheek - small patch in middle of left cheek area
        left_y = int(h * 0.5)  # Middle height
        left_x = int(w * 0.25)  # Left side
        left_cheek = face_crop[left_y:left_y+patch_size, left_x:left_x+patch_size]

        # Right cheek - small patch in middle of right cheek area
        right_y = int(h * 0.5)  # Middle height
        right_x = int(w * 0.75)  # Right side
        right_cheek = face_crop[right_y:right_y+patch_size, right_x:right_x+patch_size]

        # Get average RGB from both small cheek patches
        left_avg = np.mean(left_cheek.reshape(-1, 3), axis=0)
        right_avg = np.mean(right_cheek.reshape(-1, 3), axis=0)

        # Average both cheeks
        avg_b, avg_g, avg_r = ((left_avg + right_avg) / 2).astype(int)

        # Convert hex to single value for comparison
        hex_value = (avg_r << 16) + (avg_g << 8) + avg_b

        # Balanced skin tone classification
        avg_total = (avg_r + avg_g + avg_b) / 3

        # White skin: Light tones
        if avg_total > 170 and avg_r > 150 and avg_g > 140:
            return "White"
        # Black skin: Dark tones
        elif avg_total < 100 and avg_r < 110:
            return "Black"
        # Brown skin: Medium tones with warm characteristics
        elif 100 <= avg_total <= 170:
            return "Brown"
        # Edge cases based on brightness
        elif avg_total > 170:
            return "White"
        else:
            return "Black"

    def assess_lighting(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate brightness statistics
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)

        # Assess lighting quality
        if mean_brightness < 60:
            lighting_level = "Low Light"
        elif mean_brightness > 200:
            lighting_level = "Bright Light"
        else:
            lighting_level = "Normal Light"

        # Assess uniformity
        if brightness_std > 60:
            uniformity = "Uneven"
        elif brightness_std > 40:
            uniformity = "Moderate"
        else:
            uniformity = "Even"

        return f"{lighting_level}, {uniformity}"

    def detect_skin_condition(self, face_crop):
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # Calculate texture variance (higher = more texture/roughness)
        texture_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))

        # Analyze color uniformity
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        s_std = np.std(hsv[:,:,1])  # Saturation standard deviation

        # Simple condition estimation
        if texture_var > 800 and s_std > 30:
            return "Textured/Acne-prone"
        elif texture_var > 500:
            return "Slightly textured"
        elif s_std < 15:
            return "Smooth/Clear"
        else:
            return "Normal"

    def detect_age_gender(self, frame):
        try:
            # Apply adaptive preprocessing
            processed_frame = self.preprocess_for_diversity(frame)

            h, w = frame.shape[:2]
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            face_model = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
            face_proto = os.path.join(models_dir, "opencv_face_detector.pbtxt")

            face_net = cv2.dnn.readNet(face_model, face_proto)

            # Use processed frame for better detection
            blob = cv2.dnn.blobFromImage(processed_frame, 1.0, (300, 300), [104, 117, 123], True, False)
            face_net.setInput(blob)
            detections = face_net.forward()

            results = []
            age_ranges = [(0, 2), (3, 5), (6, 12), (13, 19), (20, 25), (26, 35), (36, 45), (46, 59), (60, 100)]
            midpoints = [(a + b) / 2 for (a, b) in age_ranges]

            # Different colors for multiple faces
            face_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            face_count = 0

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)

                    # Make box square
                    box_w = x2 - x1
                    box_h = y2 - y1
                    size = max(box_w, box_h)

                    # Center the square box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    x1 = center_x - size // 2
                    y1 = center_y - size // 2
                    x2 = x1 + size
                    y2 = y1 + size

                    # Ensure square stays within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    padding = 20
                    face_crop = frame[max(0, y1-padding):min(y2+padding, h-1), max(0, x1-padding):min(x2+padding, w-1)]

                    blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

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

                    # Modern styled face box with different colors for multiple faces
                    box_color = face_colors[face_count % len(face_colors)]
                    face_count += 1
                    box_thickness = max(2, int(min(w, h) / 200))  # Scale box thickness
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

                    # Corner accents for modern look
                    corner_len = max(15, int(min(w, h) / 30))
                    corner_thickness = max(3, int(min(w, h) / 150))
                    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, corner_thickness)
                    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, corner_thickness)
                    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, corner_thickness)
                    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, corner_thickness)
                    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, corner_thickness)
                    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, corner_thickness)
                    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, corner_thickness)
                    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, corner_thickness)

                    # Fixed consistent text sizing based on image dimensions
                    base_size = min(w, h)
                    if base_size < 400:
                        text_scale = 0.4
                        thickness = 1
                    elif base_size < 800:
                        text_scale = 0.6
                        thickness = 2
                    else:
                        text_scale = 0.8
                        thickness = 2

                    # Prepare text with face number
                    text = f"Face#{face_count}: {gender}, {age_label}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, text_scale, thickness)

                    # Smart label positioning - below the box with padding
                    padding = max(10, int(base_size / 50))
                    label_x = x1
                    label_y = y2 + text_h + padding

                    # Ensure label stays within bounds
                    if label_y + padding > h:
                        label_y = y1 - padding
                    if label_x + text_w + (padding * 2) > w:
                        label_x = w - text_w - (padding * 2)
                    label_x = max(padding, label_x)

                    # Modern label background with proper padding
                    bg_padding = max(6, int(base_size / 80))
                    bg_color = (50, 50, 50)  # Dark gray
                    cv2.rectangle(frame, (label_x - bg_padding, label_y - text_h - bg_padding),
                                (label_x + text_w + bg_padding, label_y + bg_padding), bg_color, -1)
                    cv2.rectangle(frame, (label_x - bg_padding, label_y - text_h - bg_padding),
                                (label_x + text_w + bg_padding, label_y + bg_padding), box_color, max(1, box_thickness - 1))

                    # Clean white text with anti-aliasing
                    cv2.putText(frame, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                    # Analyze diversity features
                    diversity_info = self.analyze_diversity_features(frame, face_crop)
                    self.diversity_results = diversity_info  # Store for info display

                    results.append(f"Face #{face_count}:\nEstimated Age: {estimated_age:.2f} years\nAge Group: {gender} {age_label}\nConfidence Level: {age_conf:.2f}%\n")

            return frame, results
        except Exception as e:
            return frame, [f"Error: {str(e)}"]

    # --- WEBCAM FUNCTIONS ---
    def toggle_webcam(self):
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Webcam not detected!")
                return

            self.webcam_active = True
            self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
            self.webcam_thread.start()

            # Update button text
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Button) and "Webcam" in child.cget("text"):
                            child.configure(text="Stop Webcam")
                            break
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")

    def stop_webcam(self):
        self.webcam_active = False
        if self.cap:
            self.cap.release()

        # Update button text
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button) and "Stop" in child.cget("text"):
                        child.configure(text="Start Webcam")
                        break

    def webcam_loop(self):
        last_detection = 0
        last_face_data = None  # Store face detection data

        while self.webcam_active:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Flip horizontally
            current_time = time.time()

            # Detect every 3 seconds to reduce lag
            if current_time - last_detection >= 3:
                result_frame, results = self.detect_age_gender(frame.copy())

                if results:
                    result_text = "\n".join(results)
                    self.results_label.configure(text=result_text)

                    # Keep info panel empty unless diversity button is clicked
                    if not self.show_diversity:
                        self.info_label.configure(text="")

                    # Store face detection data for persistent display
                    last_face_data = self.extract_face_data(frame.copy())

                last_detection = current_time

            # Apply stored face data to current frame
            if last_face_data:
                frame = self.apply_face_overlay(frame, last_face_data)

            # Resize before conversion for better performance
            height, width = frame.shape[:2]
            if height > 400:
                ratio = 400 / height
                new_width = int(width * ratio)
                frame = cv2.resize(frame, (new_width, 400))

            # Convert and display
            result_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(result_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo

            time.sleep(0.033)  # ~30 FPS

    def upload_image(self):
        if self.webcam_active:
            self.stop_webcam()
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            self.process_image(file_path)

    def load_sample(self, sample_name):
        if self.webcam_active:
            self.stop_webcam()
        if os.path.exists(sample_name):
            self.process_image(sample_name)
        else:
            messagebox.showerror("Error", f"Sample image {sample_name} not found")

    def process_image(self, image_path):
        try:
            # Load the image
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("Error", "Could not read image")
                return
            # Run age and gender prediction
            result_img, results = self.detect_age_gender(frame)

            # Convert and display photo
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            # Resize image
            height, width = result_rgb.shape[:2]
            if height > 400:
                ratio = 400 / height
                new_width = int(width * ratio)
                result_rgb = cv2.resize(result_rgb, (new_width, 400))
            # Conver to PhotoImage format
            pil_image = Image.fromarray(result_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            # Display the results
            if results:
                result_text = "\n".join(results)
                self.results_label.configure(text=result_text)
                self.show_diversity = not self.show_diversity
                self.diversity_btn.configure(text="Show Diversity Analysis")

                # Keep info panel empty unless diversity button is clicked
                if not self.show_diversity:
                    self.info_label.configure(text="")


        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def save_results(self):
        try:
            file_path = filedialog.asksaveasfilename(title="Save Results", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if not file_path:
                return

            # Save image if available
            if hasattr(self.image_label, 'image') and self.image_label.image:
                # Get current image from display
                pil_img = ImageTk.getimage(self.image_label.image)

                # Add text overlay with results
                from PIL import ImageDraw, ImageFont

                # Create larger canvas to fit text
                age_text = self.results_label.cget("text")
                info_text = self.info_label.cget("text")
                text_lines = [line for line in (age_text.split('\n') + [''] + info_text.split('\n')) if line.strip()]

                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default()

                # Calculate text area needed
                line_height = 18
                text_height = len(text_lines) * line_height + 20

                # Create new image with extra space for text
                new_height = pil_img.height + text_height
                new_img = Image.new('RGB', (pil_img.width, new_height), color='black')
                new_img.paste(pil_img, (0, 0))

                draw = ImageDraw.Draw(new_img)

                # Add text below image
                y_offset = pil_img.height + 10
                for line in text_lines:
                    draw.text((10, y_offset), line, fill="yellow", font=font)
                    y_offset += line_height

                pil_img = new_img

                pil_img.save(file_path)
                messagebox.showinfo("Saved", f"Image with results saved to {file_path}")
            else:
                messagebox.showwarning("Warning", "No image to save")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    # For webcam face detection
    def extract_face_data(self, frame):
        try:
            h, w = frame.shape[:2]
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            face_model = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
            face_proto = os.path.join(models_dir, "opencv_face_detector.pbtxt")

            face_net = cv2.dnn.readNet(face_model, face_proto)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
            face_net.setInput(blob)
            detections = face_net.forward()

            face_data = []
            age_ranges = [(0, 2), (3, 5), (6, 12), (13, 19), (20, 25), (26, 35), (36, 45), (46, 59), (60, 100)]

            # Different colors for multiple faces
            face_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            face_count = 0

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                # Only continue if face detection confidence is 70%
                if confidence > 0.7:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)

                    # Make box square
                    box_w = x2 - x1
                    box_h = y2 - y1
                    size = max(box_w, box_h)

                    # Center the square box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    x1 = center_x - size // 2
                    y1 = center_y - size // 2
                    x2 = x1 + size
                    y2 = y1 + size

                    # Ensure square stays within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    padding = 20
                    face_crop = frame[max(0, y1-padding):min(y2+padding, h-1), max(0, x1-padding):min(x2+padding, w-1)]

                    blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

                    # Gender
                    self.genderNet.setInput(blob)
                    genderPreds = self.genderNet.forward()
                    gender_idx = int(genderPreds[0].argmax())
                    gender = self.genderList[gender_idx]

                    # Age
                    self.ageNet.setInput(blob)
                    agePreds = self.ageNet.forward()
                    probs = agePreds[0]
                    age_idx = int(np.argmax(probs))
                    age_label = self.ageList[age_idx]

                    face_count += 1
                    face_data.append({
                        'box': (x1, y1, x2, y2),
                        'text': f"Face#{face_count}: {gender}, {age_label}",
                        'color': face_colors[(face_count-1) % len(face_colors)]
                    })

            return face_data
        except:
            return None

    def apply_face_overlay(self, frame, face_data):
        h, w = frame.shape[:2]

        for face in face_data:
            x1, y1, x2, y2 = face['box']
            text = face['text']
            box_color = face['color']

            # Modern styled face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

            # Corner accents
            corner_len = 20
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 5)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 5)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 5)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 5)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 5)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 5)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 5)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 5)

            # Text styling
            text_scale = min(w, h) / 1000.0
            text_scale = max(0.5, min(text_scale, 0.8))
            thickness = max(1, int(text_scale * 2))

            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, text_scale, thickness)

            # Label positioning
            label_x = x1
            label_y = y2 + text_h + 15

            if label_y + 10 > h:
                label_y = y1 - 10
            if label_x + text_w + 20 > w:
                label_x = w - text_w - 20

            # Label background
            bg_color = (50, 50, 50)
            cv2.rectangle(frame, (label_x - 8, label_y - text_h - 8), (label_x + text_w + 8, label_y + 8), bg_color, -1)
            cv2.rectangle(frame, (label_x - 8, label_y - text_h - 8), (label_x + text_w + 8, label_y + 8), box_color, 2)

            # Text with anti-aliasing
            cv2.putText(frame, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    def toggle_diversity_display(self):
        self.show_diversity = not self.show_diversity

        # Update button text
        if self.show_diversity:
            self.diversity_btn.configure(text="Hide Diversity Analysis")
            # Show diversity if data exists
            if hasattr(self, 'diversity_results'):
                diversity_text = f"Ethnicity: {self.diversity_results['ethnicity']}\nSkin Tone: {self.diversity_results['skin_tone']}\nSkin Condition: {self.diversity_results['skin_condition']}\nLighting: {self.diversity_results['lighting']}"
                self.info_label.configure(text=diversity_text)
            else:
                self.info_label.configure(text="No diversity data available")
        else:
            self.diversity_btn.configure(text="Show Diversity Analysis")
            self.info_label.configure(text="")

    def __del__(self):
        if self.webcam_active:
            self.stop_webcam()

if __name__ == "__main__":
    print("Starting FaceAge Detection UI...")
    root = tk.Tk()
    app = FaceAgeDetectorUI(root)
    root.mainloop()
