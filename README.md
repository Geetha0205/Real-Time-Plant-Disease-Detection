# 🌱 Real-Time Plant Disease Detection Using AI

An AI-powered plant disease detection application built using a custom-trained YOLOv11 model and Streamlit. The system identifies healthy and diseased plant leaves through image uploads and real-time webcam detection.

## 🚀 Project Overview

This project uses deep learning and computer vision to detect plant diseases from leaf images. A custom YOLOv11 model was trained from scratch using labeled plant leaf images to identify healthy and diseased leaves.

The application provides an interactive interface using Streamlit, allowing users to upload images or use a webcam for real-time detection.

## ✨ Features

- **AI-Based Detection:** Uses a custom-trained YOLOv11 model.
- **Image Upload:** Detects plant diseases from uploaded leaf images.
- **Real-Time Detection:** Supports webcam-based plant leaf detection.
- **Visual Results:** Displays detection results through the Streamlit interface.
- **Deep Learning:** Uses computer vision techniques for automated disease identification.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Model | YOLOv11 |
| Computer Vision | OpenCV |
| Web Framework | Streamlit |
| Model Format | PyTorch |
| Development Environment | Python / Jupyter Notebook |

## 📂 Project Structure

```text
Real-Time-Plant-Disease-Detection/
│
├── images/
│   └── selected_images/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Geetha0205/Real-Time-Plant-Disease-Detection.git
```

### 2. Navigate to the project directory

```bash
cd Real-Time-Plant-Disease-Detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 🧠 Model Information

- Model: YOLOv11
- Training Approach: Custom training from scratch
- Task: Plant leaf disease detection
- Input: Plant leaf images and webcam frames
- Output: Detection results displayed in the application

## 🔮 Future Enhancements

- Support for additional plant species and disease categories.
- Improve detection accuracy using larger datasets.
- Deploy the application on a cloud platform.
- Add detailed disease information and treatment recommendations.

## 👩‍💻 Developed By

**Geetha Doddala**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence and Data Science

GitHub: [Geetha0205](https://github.com/Geetha0205)

LinkedIn: [Geetha Doddala](https://www.linkedin.com/in/geethadoddala04/)