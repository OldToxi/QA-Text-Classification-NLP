# QA-Text-Classification-NLP
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-success.svg)

## Project Overview
This repository contains my implementation of various Machine Learning and Deep Learning architectures to classify question-answer texts into specific semantic categories. The objective is to map questions to their underlying intent using linguistic preprocessing and neural network embeddings.

## Technical Implementation

### 1. Linguistic Preprocessing
To prepare the raw text for the neural networks, I implemented a robust cleaning pipeline:
* **Sanitization:** Removal of HTML tags, URLs, and redundant punctuation.
* **Normalization:** Tokenization, stopword removal, and lemmatization to reduce the vocabulary space while preserving semantic meaning.

### 2. Feature Engineering & Embeddings
I compared traditional statistical methods against dense vector representations:
* **TF-IDF:** Implemented to establish a baseline using Logistic Regression and Naive Bayes.
* **Word2Vec (Skip-gram):** Utilized to capture dense semantic relationships for the deep learning models.

### 3. Model Architecture Progression
I architected and trained a progression of models to evaluate the impact of memory gating and bidirectionality on text sequences:
* Simple RNN
* GRU & Bidirectional GRU
* LSTM & Bidirectional LSTM

## Performance Analysis & Insights
Through rigorous evaluation using Accuracy, F1-Scores, and Confusion Matrices, the **Unidirectional LSTM emerged as the optimal architecture** (achieving ~68.62% accuracy). 

**Key Technical Takeaways:**
1. **The Role of Memory Gating:** The LSTM significantly outperformed the Simple RNN. By solving the vanishing gradient problem, the LSTM effectively retained long-term dependencies in longer question texts.
2. **Diminishing Returns of Bidirectionality:** While Bidirectional models capture backward context, they added computational complexity without a proportional increase in accuracy for this specific dataset.
3. **Semantic Overlap Challenges:** Error analysis of the confusion matrix revealed the model struggled most with conceptually adjacent categories (e.g., misclassifying "Society & Culture" as "Religion"). This highlights the limitation of fixed word embeddings, pointing toward Transformer-based models (like BERT) as the logical next step for contextual nuance.

## Repository Structure
├── QA_Text_Classification.ipynb  # Main Notebook
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation
