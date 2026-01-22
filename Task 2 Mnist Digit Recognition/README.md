🧠 Handwritten Digit Recognition Using MNIST Dataset

📌 Project Description

This project focuses on handwritten digit recognition using the well-known MNIST dataset.
The main aim of this work is to train and evaluate different Machine Learning and Deep Learning models that are capable of correctly identifying handwritten digits ranging from 0 to 9.

To make the project more practical and interactive, a web application has also been developed using Streamlit. Through this application, users can upload their own handwritten digit image and instantly view the prediction made by the selected model.

🎯 Objectives of the Project

The key objectives of this project are:

To understand the working of image classification using the MNIST dataset

To implement and compare traditional Machine Learning models with Deep Learning models

To develop a real-world AI application with a simple and user-friendly interface

To gain hands-on experience in image preprocessing, model training, and evaluation

📊 Dataset Information

The MNIST dataset is a widely used benchmark dataset for handwritten digit recognition tasks.

Training images: 60,000

Testing images: 10,000

Image size: 28 × 28 pixels

Image type: Grayscale

Classes: Digits from 0 to 9

Each image in the dataset represents a single handwritten digit.

🤖 Models Used

The following models were trained and tested during this project:

Logistic Regression

K-Nearest Neighbors (KNN)

Artificial Neural Network (ANN)

Convolutional Neural Network (CNN)

After evaluation, it was observed that the CNN model achieved the highest accuracy, as it is specially designed to handle image-based data effectively.

🛠️ Image Preprocessing

Before training and prediction, the images are processed using the following steps:

Conversion of images to grayscale

Resizing images to 28 × 28 pixels

Normalizing pixel values from 0–255 to 0–1

Reshaping images according to the requirements of the selected model

These preprocessing steps play an important role in improving model performance and prediction accuracy.

🌐 Web Application

A Streamlit-based web application is developed to make the project interactive and easy to use.

Key Features of the Application:

Upload handwritten digit images

Select a model (Logistic Regression, KNN, ANN, or CNN)

Display the uploaded image

Show the predicted digit instantly

This application allows users to easily test and compare different models.

📁 Project Structure
Task 2 Mnist Digit Recognition/
│
├── app.py
├── cnn_model.h5
├── nn_model.h5
├── logistic_model.pkl
├── knn_model.pkl
├── README.md

▶️ How to Run the Project
Step 1: Install Required Libraries
pip install streamlit tensorflow scikit-learn numpy pillow opencv-python

Step 2: Run the Application
streamlit run app.py


After executing the above command, the web application will automatically open in the browser.

📈 Model Evaluation

The performance of the models is evaluated using standard classification metrics, including:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Among all the models, the CNN model demonstrated the best overall performance.

💻 Tools and Technologies Used

Python

NumPy

Scikit-learn

TensorFlow / Keras

OpenCV

Streamlit

📝 Conclusion

This project demonstrates how various Machine Learning and Deep Learning techniques can be applied to solve a real-world image classification problem.
The results clearly indicate that Deep Learning models, especially Convolutional Neural Networks, perform significantly better for image recognition tasks when compared to traditional Machine Learning models.

Working on this project helped in developing practical skills related to data preprocessing, model training, performance evaluation, and application deployment.

👤 Author

Huzaifa Baig
BS Computer Science
MNIST Digit Recognition Project