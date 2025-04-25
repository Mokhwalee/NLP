# 📝 Natural Language Processing (NLP)

This repository contains my study notes, code implementations, and examples for key concepts and algorithms in Natural Language Processing (NLP). The goal is to understand and apply fundamental NLP techniques, from text preprocessing to advanced deep learning models.

## 📂 Contents

### 1. **Text Preprocessing**
- `01_text_preprocessing.ipynb`
- Tokenization
- Stopword Removal
- Stemming and Lemmatization
- POS Tagging
Covers fundamental preprocessing techniques including tokenization, lowercasing, removing stopwords, stemming, lemmatization, and part-of-speech (POS) tagging using NLTK and spaCy.

### 2. **Bag of Words and TF-IDF**
- `02_bow_tfidf.ipynb`
- Bag of Words Model
- Term Frequency-Inverse Document Frequency (TF-IDF)
Demonstrates how to convert raw text into numerical feature vectors using Bag of Words and Term Frequency–Inverse Document Frequency (TF-IDF) representations for text classification tasks.

### 3. **Word Embeddings**
- `03_word_embeddings.ipynb`
- Word2Vec
- GloVe Embeddings
- FastText
Introduces distributed representations of words using pre-trained Word2Vec, GloVe, and FastText models. Shows how embeddings capture semantic relationships between words.

### 4. **Language Models**
- `04_1_language_models.ipynb` and `04_2_transformers.ipynb`
- N-gram Language Models
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM) Networks
4.1 Explores statistical and neural language models including N-gram models, RNNs, and LSTMs. Covers their use in predicting next words and generating coherent text.
4.2 Details the transformer model architecture (self-attention, positional encoding, multi-head attention). Includes basic implementation and how it's used in modern NLP models.


### 5. **Text Classification**
- `05_text_classification.ipynb`
- Sentiment Analysis Example
- Spam Detection
- Text Classification with Logistic Regression, SVM, and Neural Networks
Implements sentiment analysis and spam detection using traditional ML models (Logistic Regression, SVM) and neural networks. Covers pipeline building, feature extraction, and evaluation.

### 6. **Sequence-to-Sequence Models**
- `06_seq2seq_models.ipynb`
- Machine Translation
- Text Summarization
Presents encoder-decoder models used for tasks like machine translation and summarization. Includes implementation of Seq2Seq with attention in PyTorch or TensorFlow.

### 7. **Attention Mechanism**
- `07_attention_mechanism.ipynb`
- Attention in Seq2Seq Models
- Self-Attention and Multi-Head Attention
Explains the intuition and implementation of attention in neural networks. Covers additive and dot-product attention, along with its role in improving Seq2Seq models.

### 8. **Pretrained Models and Fine-tuning**
- `08_finetuning_bert.ipynb`
- BERT
- GPT
- Hugging Face Transformers 
Covers Hugging Face’s Transformers library, focusing on loading and fine-tuning BERT, GPT, and other foundation models for downstream tasks like classification and Q&A.

## 🛠️ Dependencies
- Python 3.x
- numpy
- pandas
- scikit-learn
- nltk
- spacy
- gensim
- torch / tensorflow
- transformers (Hugging Face)

You can install the required packages via:
```bash
pip install -r requirements.txt
```

✅ Optional: Add this line at the end of "requirements.txt" if you plan to use GPU for TensorFlow or PyTorch: 
```bash
tensorflow-gpu
```

For PyTorch GPU: Check your CUDA version and install PyTorch accordingly from:
https://pytorch.org/get-started/locally/
(e.g., torch==2.0.1+cu118 for CUDA 11.8)