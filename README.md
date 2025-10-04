# RNN for Question Answering using PyTorch
This repository contains an implementation of a Recurrent Neural Network (RNN) for question-answering tasks using PyTorch. The project demonstrates natural language processing techniques for building a simple QA system.

📋 Project Overview
The notebook implements an RNN-based model to process question-answer pairs and learn to map questions to their corresponding answers. The implementation covers the complete pipeline from data preprocessing to model architecture.

🏗️ Project Structure
The pipeline follows these key steps:

1. Data Loading & Exploration
Loads a custom QA dataset with 90 unique question-answer pairs

Displays sample data structure with questions and answers

2. Text Preprocessing
Tokenization: Converts text to lowercase and removes punctuation

Vocabulary Building: Creates a comprehensive vocabulary from both questions and answers

Text-to-Indices: Converts tokens to numerical indices for model input

3. Vocabulary Construction
Builds a vocabulary of 324 unique tokens

Includes special 'UNK' token for out-of-vocabulary words

Handles diverse topics including geography, science, history, and pop culture

4. Dataset Preparation
Custom PyTorch Dataset class for QA pairs

Converts text data to numerical representations

Prepares data for training

🎯 Key Features
Custom Tokenization: Handles punctuation and case normalization

Vocabulary Management: Dynamic vocabulary building from dataset

Text Processing: Complete pipeline from raw text to numerical indices

PyTorch Integration: Uses PyTorch's Dataset class for efficient data loading

📊 Dataset Information
The dataset contains:

90 question-answer pairs covering diverse topics

Questions: Various factual questions from different domains

Answers: Concise factual responses

Topics include: Geography, Science, History, Literature, Pop Culture

🛠️ Technical Implementation
Tokenization Function
python
def tokenize(text):
    text = text.lower()
    text = text.replace('?', '')
    text = text.replace("'", "")
    return text.split()
Vocabulary Building
Starts with 'UNK' token (index 0)

Dynamically adds new tokens from both questions and answers

Handles compound words and special characters

Text-to-Indices Conversion
python
def text_to_indices(text, vocab):
    indexed_text = []
    for token in tokenize(text):
        if token in vocab:
            indexed_text.append(vocab[token])
        else:
            indexed_text.append(vocab['UNK'])
    return indexed_text
🚀 Usage
Load and explore the QA dataset

Run the vocabulary building process

Convert text to numerical indices

Prepare the dataset for RNN training

🔧 Current Status
The notebook currently includes:

✅ Data loading and exploration

✅ Text preprocessing and tokenization

✅ Vocabulary building (324 tokens)

✅ Text-to-indices conversion

✅ Dataset class skeleton

Next Steps Required:

Complete the Dataset class implementation

Implement RNN model architecture

Add training loop and loss functions

Implement evaluation metrics

📈 Potential Applications
Simple factual question answering

Educational chatbots

Information retrieval systems

Foundation for more complex NLP tasks

🛠️ Technical Stack
Framework: PyTorch

Data Processing: pandas, custom text processing

NLP Techniques: Tokenization, vocabulary building

Architecture: RNN-based sequence processing

This project serves as a foundation for understanding RNN applications in natural language processing and question-answering systems using PyTorch.
