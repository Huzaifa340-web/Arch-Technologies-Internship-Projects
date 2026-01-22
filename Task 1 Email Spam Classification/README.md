📧 Email Spam Classification Desktop Application

ARCH TECHNOLOGIES – Machine Learning Task Submission

📌 Project Overview

This project is a Machine Learning–based Email/SMS Spam Classifier developed as part of the ARCH TECHNOLOGIES Machine Learning tasks.
The system is capable of identifying whether a given email or message is Spam or Not Spam.

Along with the machine learning model, a desktop GUI application has also been developed using Python to make the system easy and user-friendly.

The project demonstrates complete end-to-end machine learning workflow, starting from data preprocessing to model training, evaluation, and real-time prediction using a graphical interface.

🎯 Objectives

To understand text preprocessing techniques in Machine Learning

To build a spam classification model using Naive Bayes

To convert text data into numerical features using TF-IDF

To evaluate model performance using standard metrics

To create an attractive and functional desktop GUI application

To allow users to test real emails through the GUI

📂 Dataset Used

UCI SMS Spam Collection Dataset

The dataset contains labeled messages:

spam → unwanted or promotional messages

ham → normal messages

Each record contains:

Label (spam / ham)

Message text

🛠️ Technologies & Libraries Used
🔹 Programming Language

Python

🔹 Machine Learning Libraries

scikit-learn

pandas

numpy

🔹 Natural Language Processing

nltk

Stopwords removal

Text preprocessing

🔹 Feature Extraction

TF-IDF Vectorizer

🔹 Machine Learning Model

Multinomial Naive Bayes

🔹 GUI (Desktop Application)

Python GUI Library (Desktop-based)

GUI allows:

Text input

One-click prediction

Visual spam / not spam result

🔹 Model Saving

pickle (to save trained model and vectorizer)

🧠 Machine Learning Workflow

Dataset Loading

SMS dataset loaded using pandas

Text Preprocessing

Convert text to lowercase

Remove symbols and numbers

Remove stopwords

Clean and normalize text

Feature Extraction

Convert cleaned text into numerical form using TF-IDF

Train-Test Split

80% data for training

20% data for testing

Model Training

Multinomial Naive Bayes classifier trained on text features

Model Evaluation

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Real-Time Prediction

New email/message input tested

Model predicts Spam or Not Spam

Model Saving

Trained model saved as spam_model.pkl

Vectorizer saved as vectorizer.pkl

🖥️ Desktop GUI Application

To make the project practical and user-friendly, a desktop GUI application was created.

GUI Features:

Text area to enter email/message

Predict button

Clear visual result:

🔴 SPAM

🟢 NOT SPAM

Uses the same trained ML model

Ensures same preprocessing as training (for accurate results)

This GUI allows non-technical users to easily test emails without running code manually.

📁 Project Structure
SpamClassifierGUI/
│
├── spam_gui.py          # GUI application code
├── spam_model.pkl       # Trained ML model
├── vectorizer.pkl       # TF-IDF vectorizer
├── SMSSpamCollection   # Dataset file
├── README.md            # Project documentation

▶️ How to Run the Project
1️⃣ Install Required Libraries
pip install pandas scikit-learn nltk

2️⃣ Train the Model

Run the training script to generate:

spam_model.pkl

vectorizer.pkl

3️⃣ Run GUI Application
python spam_gui.py

4️⃣ Test Emails

Enter any email/message

Click Predict

View result instantly

📊 Results

The model achieves high accuracy on test data

Successfully identifies spam patterns

GUI predictions match trained model results

Real-world email testing works correctly

✅ Conclusion

This project successfully demonstrates how Machine Learning and NLP can be used to solve real-world problems like spam detection.
By integrating a desktop GUI, the project becomes practical, interactive, and user-friendly.

It covers:

Data preprocessing

Feature engineering

Model training & evaluation

Model deployment through a GUI

This project fulfills all the requirements of ARCH TECHNOLOGIES Machine Learning Task Submission.

👤 Developed By

Huzaifa Baig
Machine Learning Task Submission
ARCH TECHNOLOGIES

