# **FaceAge: Enhanced Facial Age Estimation using Image Processing**

### Members:
- Follante, Adrian Paolo S.
- Manalo, Ram Andrei M.
- Ramos, Renzo Emmanuel V.
- Unido, Jem Arden D.

## **Problem Description**

Accurately determining a person’s age is important in various fields, such as identity verification, security protocols, healthcare assessments, and personalized digital services. However, current age estimation practices often rely on manual visual inspection of facial features, which can be time-consuming, subjective, and inconsistent. This can lead to inefficient decision-making and unreliable age estimation systems.

## **Proposed Solution Overview**

To address this issue, we propose an intelligent application that estimates the age of human faces through image analysis. The system will utilize a Convolutional Neural Network (CNN) to automatically extract and analyze visual features such as wrinkles, skin texture, bone structure, and facial geometry. By applying deep learning techniques, the application aims to deliver accurate, fast, and consistent age estimations, reducing human error and enhancing productivity in various operations. As an extra feature, we also added a skin condition detection feature, which could help in predicting possible skin diseases that a person may have.

## **PEAS Model**
### Performance Measure
- Accuracy of predicted age compared to actual age
- Classify the human face into their predicted age category: Newborn (0 to 2 years old), Toddler (3-6), Child (7-12), Teen (13-18), Young Adult (19-25), Adult (26-35), Middle Age (36-45), Mature (46-59), and Senior (60 and above)
- Fast processing time of image analyzation using webcam and uploaded images
- Ability to process different ethnicities, skin conditions, skin tones, and lighting conditions
- High confidence score for each prediction

### Environment
- Virtual environment running on a computer or laptop
- Accepts two input environments:
- Real-time webcam feed
- Uploaded facial image


### Actuators
- Displays the estimated age of the human
- Shows age group classification
- Displays confidence level of the prediction
- Displays possible skin condition
- Optionally saves the results to a file

### Sensors
- Webcam stream for real-time face capture
- Image uploaded manually by the user
- Face detection system (Haar Cascade, HOG, CNN-based detectors)
- Feature extraction tools (landmarks, embeddings, texture detectors)

## **AI Concepts Used**

The AI concepts implemented in FaceAge: Enhanced Facial Age Estimation Using Image Processing focus on using deep learning models for facial analysis to estimate age and predict skin-related features. The system leverages pre-trained Convolutional Neural Networks (CNNs) and Deep Neural Networks (DNNs) to detect faces, estimate age, and classify gender from input images.

Face detection is performed using the YuNet face detector (via an ONNX model), which accurately identifies facial regions in images. Once a face is detected, the system extracts the facial area and feeds it into the age and gender networks, which are trained to predict age groups and gender probabilities. Age estimation is calculated using a combination of classification outputs and probabilistic weighting to produce a more precise age estimate.

The workflow demonstrates principles of supervised learning, as the pre-trained models were trained on labeled datasets of faces with corresponding age and gender information. Image preprocessing steps, such as resizing, normalization, and mean subtraction, are applied to ensure consistency with the training data.

Additionally, the system is designed for real-time applications, supporting both image uploads and webcam input. While it does not perform model training within the application, it applies inference techniques, feature extraction, and pattern recognition to make accurate predictions on unseen images. The architecture demonstrates the integration of AI concepts, computer vision, and image processing to provide fast, reliable, and automated facial analysis for applications in digital services, security, and user profiling.

_AI Concepts Used Summary:_

- **Learning Agent** - The system uses pre-trained deep neural networks to analyze facial images and estimate age by recognizing patterns related to skin texture, facial geometry, and age-related features. Although the app does not train the models itself, it applies learned representations for inference.
- **Optimization Strategy**  - ImThe underlying models were trained using optimization techniques (e.g., gradient descent) on labeled facial datasets to minimize prediction errors and improve age and gender estimation accuracy.
- **Decision Component** - The system detects faces, extracts facial features, and predicts the age group and gender. It computes an estimated age using probabilistic weighting, providing reliable, fast, and automated predictions suitable for real-time applications.

## **System Architecture Flowchart**

<img width="189" height="605" alt="image" src="https://github.com/user-attachments/assets/37bda7b3-c12e-488f-81f8-b87f03d05bac" />

### User Input
- User Uploads Image or Webcam Feed

### Image Preprocessing
- Resize / Normalize
- Face Detection (e.g., Haar, MTCNN)

### Feature Extraction (CNN)
A Convolutional Neural Network analyzes the image to detect and extract key features such as:
- Detect facial features (wrinkles, skin texture, geometry)
- Extract relevant visual patterns for age estimation

### Model Prediction
- Convolutional Neural Network (Supervised Learning)
- Predicts age (exact or age group)

### Decision & Classification
- Classifies into an exact age or age range
- Calculates confidence level 

### Output Generation
- Displays estimated age and confidence level
- Shows age range or exact age
- Displays possible skin conditions

