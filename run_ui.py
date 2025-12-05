"""
FaceAge: Enhanced Facial Age Estimation using Image Processing
Integrated UI with webcam functionality
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

    def load_models(self):
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            
            self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            self.ageList = ['Newborn (0-2)', 'Toddler (3-6)', 'Child (7-12)', 'Teen (13-18)', 'Young Adult (19-25)', 'Adult (26-35)', 'Middle Age (36-45)', 'Mature (46-59)', 'Senior (60-100)']
            self.genderList = ['Male', 'Female']
            
            age_proto = os.path.join(models_dir, "age_deploy.prototxt")
            age_model = os.path.join(models_dir, "age_net.caffemodel")
            gender_proto = os.path.join(models_dir, "gender_deploy.prototxt")
            gender_model = os.path.join(models_dir, "gender_net.caffemodel")
            
            self.ageNet = cv2.dnn.readNet(age_model, age_proto)
            self.genderNet = cv2.dnn.readNet(gender_model, gender_proto)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")

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
        
        self.info_label = tk.Label(results_container, text="Skin Condition Detection will display here", bg="#dda0dd", fg="#2c3e50", font=("Segoe UI", 12), relief=tk.FLAT, height=8, width=40, anchor="nw", justify="left", padx=5, pady=5)
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

    def detect_age_gender(self, frame):
        try:
            h, w = frame.shape[:2]
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            face_model = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
            face_proto = os.path.join(models_dir, "opencv_face_detector.pbtxt")
            
            face_net = cv2.dnn.readNet(face_model, face_proto)
            
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
            face_net.setInput(blob)
            detections = face_net.forward()
            
            results = []
            age_ranges = [(0, 2), (3, 5), (6, 12), (13, 19), (20, 25), (26, 35), (36, 45), (46, 59), (60, 100)]
            midpoints = [(a + b) / 2 for (a, b) in age_ranges]
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
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
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Consistent text sizing based on image dimensions
                    text_scale = min(w, h) / 800.0  # Scale based on image size
                    text_scale = max(0.4, min(text_scale, 0.7))  # Limit scale between 0.4 and 0.7
                    thickness = max(1, int(text_scale * 2))
                    
                    # Prepare text with proper sizing
                    text = f"{gender}, {age_label}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
                    
                    # Position text above box, ensure it stays within image bounds
                    text_x = max(5, min(x1, w - text_w - 5))
                    text_y = max(text_h + 5, y1 - 5)
                    
                    # Draw text background for better readability
                    cv2.rectangle(frame, (text_x - 2, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), (0, 0, 0), -1)
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), thickness)
                    
                    results.append(f"Estimated Age: {estimated_age:.2f} years\nAge Group: {gender} {age_label}\nConfidence Level: {age_conf:.2f}%\n")
            
            return frame, results
        except Exception as e:
            return frame, [f"Error: {str(e)}"]

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
        display_frame = None
        
        while self.webcam_active:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Flip horizontally
            current_time = time.time()
            
            # Always show live feed, detect every 2 seconds
            if current_time - last_detection >= 2:
                display_frame, results = self.detect_age_gender(frame.copy())
                
                if results:
                    result_text = "\n".join(results)
                    self.results_label.configure(text=result_text)
                    self.info_label.configure(text="Real-time webcam analysis active")
                
                last_detection = current_time
            else:
                # Show live feed without detection boxes
                display_frame = frame
            
            # Convert and display current frame
            result_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            height, width = result_rgb.shape[:2]
            if height > 400:
                ratio = 400 / height
                new_width = int(width * ratio)
                result_rgb = cv2.resize(result_rgb, (new_width, 400))
            
            pil_image = Image.fromarray(result_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            time.sleep(0.05)  # ~20 FPS for smoother display

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
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("Error", "Could not read image")
                return
            
            result_img, results = self.detect_age_gender(frame)
            
            # Convert and display
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            height, width = result_rgb.shape[:2]
            if height > 400:
                ratio = 400 / height
                new_width = int(width * ratio)
                result_rgb = cv2.resize(result_rgb, (new_width, 400))
            
            pil_image = Image.fromarray(result_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            if results:
                result_text = "\n".join(results)
                self.results_label.configure(text=result_text)
                self.info_label.configure(text="Static image analysis complete")
            
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

    def __del__(self):
        if self.webcam_active:
            self.stop_webcam()

if __name__ == "__main__":
    print("Starting FaceAge Detection UI...")
    root = tk.Tk()
    app = FaceAgeDetectorUI(root)
    root.mainloop()