# IMDB Movie Review Sentiment Analysis 🎬

A Deep Learning project that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative** using an LSTM (Long Short-Term Memory) neural network.

## 📌 Project Overview
This project processes text data (movie reviews), tokenizes the words, and feeds them into a Recurrent Neural Network (RNN/LSTM) built with **TensorFlow/Keras**. The model is trained to understand the context of reviews and classify them with high accuracy.

## 📂 Files Description
* **`IMDB Sentiment Analysis.py`**: The main script for:
    * Loading and cleaning the dataset (`final data set1.xlsx`).
    * Tokenizing and padding text sequences.
    * Building and training the LSTM model.
    * Saving the model (`.keras`) and tokenizer (`.pickle`).
* **`test.py`**: A user-friendly script to test the model. It allows you to input any sentence and get an instant sentiment prediction.
* **`tokenizer.pickle`**: The saved tokenizer object to ensure inputs are processed exactly like the training data.
* **`final_model.keras`**: The trained Deep Learning model file.

## 🛠️ Tech Stack
* **Python 3.x**
* **TensorFlow / Keras** (Deep Learning)
* **Pandas** (Data Manipulation)
* **NumPy** (Numerical Operations)
* **Pickle** (Object Serialization)
* **Colorama** (Colored Terminal Output)

## 🚀 How to Run

### 1. Install Dependencies
Make sure you have the required libraries installed:
```bash
pip install tensorflow pandas numpy colorama openpyxl
