📌 ARCH TECHNOLOGIES

Machine Learning Internship – Task Submission

Intern Name: Huzaifa Baig
Domain: Machine Learning
Organization: ARCH TECHNOLOGIES

📖 Project Overview

This repository contains the complete submission for the ARCH TECHNOLOGIES Machine Learning Internship Tasks.

The project consists of two major machine learning tasks:

Email Spam Classification (Desktop Application)

Handwritten Digit Recognition using MNIST Dataset (Web Application)

Both tasks demonstrate an end-to-end Machine Learning workflow, including:

Data preprocessing

Feature extraction

Model training

Model evaluation

Model deployment through GUI/Web applications

✅ Task 1: Email Spam Classification

📧 Project Description

This task focuses on building a Machine Learning–based Email/SMS Spam Classifier using a labeled text dataset.

The system classifies messages as:

Spam (unwanted/promotional)

Not Spam (Ham) (legitimate messages)

Along with model training, a desktop GUI application is developed in Python, allowing users to test messages in real time without writing any code.

🎯 Objectives

Learn text preprocessing techniques

Convert text data into numerical features using TF-IDF

Train a Naive Bayes classification model

Evaluate model performance using standard metrics

Deploy the trained model in a desktop GUI application

📂 Dataset Used

UCI SMS Spam Collection Dataset

Each record contains:

Label: spam or ham

Message text

🛠️ Technologies & Libraries

Programming Language: Python

Machine Learning: scikit-learn

Data Handling: pandas, numpy

NLP: nltk (stopwords removal, text cleaning)

Feature Extraction: TF-IDF Vectorizer

Model: Multinomial Naive Bayes

Model Saving: pickle

GUI: Python Desktop GUI

🧠 Machine Learning Workflow

Dataset Loading

Text Preprocessing

Lowercasing

Removing symbols and numbers

Stopwords removal

Feature Extraction using TF-IDF

Train-Test Split (80% training, 20% testing)

Model Training (Naive Bayes)

Model Evaluation

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Model Saving (spam_model.pkl, vectorizer.pkl)

Real-time Prediction through GUI

🖥️ Desktop GUI Application

Features:

Text input for email/message

One-click prediction

Clear result display:

🔴 SPAM

🟢 NOT SPAM

The GUI uses the same trained model and preprocessing pipeline to ensure accurate predictions.

📁 Task 1 Project Structure

SpamClassifierGUI/
│
├── spam_gui.py
├── spam_model.pkl
├── vectorizer.pkl
├── SMSSpamCollection
├── README.md

▶️ How to Run Task 1

pip install pandas scikit-learn nltk
python spam_gui.py

✅ Task 2: MNIST Digit Recognition

✍️ Project Description

This task focuses on handwritten digit recognition using the MNIST dataset.

Multiple Machine Learning and Deep Learning models are implemented and compared to classify digits from 0 to 9.

A Streamlit web application is developed where users can upload their own handwritten digit image and instantly see predictions.

🎯 Objectives

Understand image classification using MNIST

Implement ML & DL models

Compare model performance

Build a real-world AI web application

Improve practical skills in image preprocessing and evaluation

📂 Dataset Information

Training images: 60,000

Testing images: 10,000

Image size: 28 × 28

Type: Grayscale

Classes: Digits (0–9)

🤖 Models Implemented

Logistic Regression

K-Nearest Neighbors (KNN)

Artificial Neural Network (ANN)

Convolutional Neural Network (CNN)

📌 CNN achieves the highest accuracy, as it is best suited for image-based tasks.

🖼️ Image Preprocessing Steps

Convert to grayscale

Resize to 28 × 28

Normalize pixel values (0–1)

Reshape according to model

🌐 Streamlit Web Application

Features:

Upload handwritten digit image

Select prediction model

Display uploaded image

Show predicted digit instantly

📁 Task 2 Project Structure

Task 2 Mnist Digit Recognition/
│
├── app.py
├── cnn_model.h5
├── nn_model.h5
├── logistic_model.pkl
├── knn_model.pkl
├── README.md

▶️ How to Run Task 2

pip install streamlit tensorflow scikit-learn numpy pillow opencv-python
streamlit run app.py

📊 Model Evaluation

Models are evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

The CNN model performs best among all tested models.

🏁 Conclusion

This repository successfully demonstrates the application of Machine Learning and Deep Learning to solve real-world problems:

Spam detection using NLP

Image classification using MNIST

The projects cover:

Data preprocessing

Feature engineering

Model training & evaluation

Model deployment using GUI and Web applications

This submission fulfills all requirements of the ARCH TECHNOLOGIES Machine Learning Internship.

👤 Author

Huzaifa Baig
BS Computer Science
Machine Learning Internship
ARCH TECHNOLOGIES
