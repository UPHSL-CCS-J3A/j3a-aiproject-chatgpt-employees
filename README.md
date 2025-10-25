# **ChickAge: Estimating the Age of Chickens Using Image Processing**

### Members:
- Follante, Adrian Paolo S.
- Manalo, Ram Andrei M.
- Ramos, Renzo Emmanuel V.
- Unido, Jem Arden 

## **Problem Description**

In the poultry industry, determining the accurate age of chickens is essential for managing feeding schedules, monitoring growth, and maintaining overall flock health. However, current methods often rely on manual observation, which is time-consuming, inconsistent, and dependent on human expertise. This lack of precision can lead to inefficiencies in farm management and affect production quality.

## **Proposed Solution Overview**

To address this issue, we propose an intelligent application that estimates the age of chickens through image analysis. The system will utilize a Convolutional Neural Network (CNN) to automatically extract and analyze visual features such as feather patterns, body size, and color changes from chicken images. By applying deep learning techniques, the application aims to deliver accurate, fast, and consistent age estimations, reducing human error and enhancing productivity in poultry operations.

## **PEAS Model**
### Performance Measure
- Accuracy of predicted chicken age compared to actual age
- Classify the chicken into their predicted age category: Chick, Juvenile, or Adult
- Fast processing time of image analyzation
- Ability to process different chicken breeds, lighting conditions, and camera angles

### Environment
- Virtual environment running on a computer
- User uploads a chicken image as input

### Actuators
- Displays estimated age and age category of the chicken
- Shows confidence level of the prediction
- Provides feeding or care instructions based on age
- Saves the results to a file

### Sensors
- Image attached by the user
- Extracts visual features such as feathers, body size, comb growth, etc.

## **AI Concepts Used**

The AI concept implemented in “ChickAge: Estimating the Age of Chickens Using Image Processing”  is primarily a learning agent, as the system is capable of learning from chicken images to estimate their age with precision. It employs supervised learning through a Convolutional Neural Network, enabling it to identify patterns such as feather development, body size, and color variations. During the training process, the system utilizes optimization strategies to enhance prediction accuracy by minimizing errors. Upon analyzing an image, the system makes a decision by classifying the chicken into an appropriate age category and providing relevant recommendations for feeding or care. This methodology ensures that ChickAge operates efficiently, consistently, and reliably, thereby reducing human error and supporting more effective poultry management.


_AI Concepts Used Summary:_

- **Learning Agen**t - The system learns from chicken images to estimate age accurately. Uses a Convolutional Neural Network to recognize visual patterns like feathers, size, and color.
- **Optimization Strategy**  - Minimizes prediction errors during training to improve accuracy.
- **Decision Component** - Classifies chickens into age categories and provides care recommendations.



_System Architecture Flowchart – ChickAge AI Process:_

- User uploads a chicken image through the application interface.

The system prepares the image for analysis by:
          - Resizing and normalizing
          -  Removing noise
          - Enhancing color and clarity

_Feature Extraction (CNN):_

A Convolutional Neural Network analyzes the image to detect and extract key features such as:

- Feather patterns
- Comb size
- Body proportions
- Color variations

_Model Prediction:_
The CNN model uses supervised learning to estimate the chicken’s age based on extracted features.
Decision & Classification
The system classifies the chicken into one of three categories:
 🐣 Chick 
🐥 Juvenile 
🐔 Adult

 It also calculates the confidence level of the prediction.

_Output Generation:_
- Results are displayed to the user including:
- Estimated age and category
- Confidence level
- Feeding or care recommendations
- Option to save results to a file
