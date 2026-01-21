import sys
import re
import pickle
import nltk
from nltk.corpus import stopwords
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel,
    QTextEdit, QPushButton, QVBoxLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

nltk.download('stopwords')
stop_words = stopwords.words('english')

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


class SpamDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spam Email Detector")
        self.setGeometry(500, 200, 500, 400)
        self.setStyleSheet("background-color: #1e1e2f;")

        layout = QVBoxLayout()

        title = QLabel("Spam Email Classifier")
        title.setFont(QFont("Arial", 18))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignCenter)

        self.textbox = QTextEdit()
        self.textbox.setPlaceholderText("Enter email text here...")
        self.textbox.setStyleSheet("""
            background-color: white;
            font-size: 14px;
            padding: 10px;
            border-radius: 8px;
        """)

        button = QPushButton("Check Email")
        button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button.clicked.connect(self.check_spam)

        self.result = QLabel("")
        self.result.setFont(QFont("Arial", 16))
        self.result.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(self.textbox)
        layout.addWidget(button)
        layout.addWidget(self.result)

        self.setLayout(layout)

    def check_spam(self):
        text = self.textbox.toPlainText()

        if text.strip() == "":
            self.result.setText("Please enter some text")
            self.result.setStyleSheet("color: yellow;")
            return

        cleaned_text = clean_text(text)
        text_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vector)

        if prediction[0] == 1:
            self.result.setText("SPAM EMAIL")
            self.result.setStyleSheet("color: red;")
        else:
            self.result.setText("NOT SPAM")
            self.result.setStyleSheet("color: lightgreen;")

app = QApplication(sys.argv)
window = SpamDetector()
window.show()
sys.exit(app.exec_())
