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

The AI concepts implemented in FaceAge: Enhanced Facial Age Estimation Using Image Processing revolve around enabling the system to learn from facial images to accurately estimate a person's age. By utilizing supervised learning techniques, the system is trained on labeled facial data to detect and interpret age-related features, such as skin texture, wrinkles, facial geometry, and other visual cues. The core of the system is a Convolutional Neural Network (CNN), which learns patterns in the facial features that correspond to different age groups.

During training, the system applies optimization strategies like gradient descent to minimize errors and improve the model's predictive accuracy. Once trained, the system processes an input facial image, extracts relevant features, and either predicts the person's exact age or classifies them into an appropriate age range. This process ensures fast, consistent, and reliable results, reducing human bias and increasing efficiency across applications in areas such as security, healthcare, and digital services.

_AI Concepts Used Summary:_

- **Learning Agent** - The system learns from facial images to estimate age by analyzing patterns related to skin texture, wrinkles, and facial structure.
- **Optimization Strategy**  - Implements optimization techniques (e.g., gradient descent) to minimize prediction errors and enhance model accuracy during training.
- **Decision Component** - The system classifies individuals into age groups or predicts an exact age, offering reliable and fast age estimation for real-world applications.

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

