## 📌 Project Overview

This notebook is a **deep-dive into Natural Language Processing** using real Twitter data.
It covers everything from NLP fundamentals to advanced deep learning techniques —
making it a complete reference for building production-ready text classification systems.

The project explores and compares **10+ vectorization and neural network combinations**,
from simple One-Hot Encoding all the way to **custom-trained Word2Vec embeddings**
with Bidirectional LSTM networks.

---

## 🎯 Objective

Build machine learning models that can accurately classify tweets as:

| Label | Sentiment | Count |
|-------|-----------|-------|
| `4` ✅ | **Positive** | 800,000 tweets |
| `0` ❌ | **Negative** | 800,000 tweets |

> The dataset is **perfectly balanced** — 800K positive and 800K negative tweets —
> eliminating class imbalance concerns entirely.

---

## 📊 Dataset

| Detail | Info |
|--------|------|
| **Name** | Sentiment140 |
| **Source** | [Kaggle — Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) |
| **Total Tweets** | 1,600,000 |
| **Class Balance** | 50% Positive / 50% Negative (perfectly balanced) |
| **Positive Label** | `4` |
| **Negative Label** | `0` |
| **Language** | English |

---

## 📓 Notebook Structure

This notebook is organized into **4 major sections**:

---

### 📁 Section 1 — Exploratory Data Analysis (EDA)

| Sub-section | Description |
|-------------|-------------|
| `1.1` Loading the Dataset | Reading CSV into Pandas DataFrame |
| `1.2` Data Cleaning | Removing noise, nulls, irrelevant columns |
| `1.3` Label Distribution | Visualizing class balance (Positive vs Negative) |
| `1.4` Creating Word Cloud | Most frequent words in Positive & Negative tweets |
| `1.5` Word Length Distribution | Analyzing tweet length patterns |

---

### 📁 Section 2 — Data Preparation for Model Building

| Sub-section | Description |
|-------------|-------------|
| `2.1` Loading the Dataset | Preparing full dataset for training |
| `2.2` Data Cleaning | Text preprocessing — removing URLs, mentions, hashtags, stopwords, stemming |

---

### 📁 Section 3 — Trying Different Neural Networks

> This is the core section where **10 different model architectures** are trained and compared.

| Sub-section | Technique | Type |
|-------------|-----------|------|
| `3.1` | Train / Test / Validation Split | Data Splitting |
| `3.2` | **One Hot Encoding** + Model Training | Classical Encoding |
| `3.3` | **Count Vectorizer** + Model Training | Bag of Words |
| `3.4` | **TF-IDF** + Model Training | Statistical Weighting |
| `3.5` | **ANN** with Text Sequences | Deep Learning |
| `3.6` | **ANN + Embedding** with Text Sequences | Embedding Layer |
| `3.7` | **LSTM + Embedding** with Text Sequences | Recurrent Network |
| `3.8` | **GRU + Embedding** with Text Sequences | Gated Recurrent Unit |
| `3.9` | **Bidirectional GRU + Embedding** | Bidirectional RNN |
| `3.10` | **Bidirectional LSTM + Embedding** | Bidirectional RNN |
| `3.11` | **Conv1D + Embedding** with Text Sequences | Convolutional Network |

---

### 📁 Section 4 — Neural Networks with Custom Word2Vec

| Sub-section | Description |
|-------------|-------------|
| `4.1` | **Custom Word2Vec in Conv1D** |
| `4.1.1` | Loading the dataset |
| `4.1.2` | Training Word2Vec model on tweet corpus |
| `4.1.3` | Creating Embedding Matrix from Word2Vec |
| `4.1.4` | Training Conv1D Neural Network with custom embeddings |
| `4.2` | **Custom Word2Vec + Bidirectional LSTM** |

---


---

## 🔢 Text Encoding Techniques Compared

| Technique | Description | Pros | Cons |
|-----------|-------------|------|------|
| **One Hot Encoding** | Binary vector per word | Simple, fast | No semantic meaning |
| **Count Vectorizer** | Word frequency matrix | Easy to implement | Ignores word order |
| **TF-IDF** | Term frequency × inverse document frequency | Handles common words | Still no semantics |
| **Keras Tokenizer + Sequences** | Maps words to integers | Works with RNNs | Simple mapping |
| **Embedding Layer** | Learnable dense word vectors | Learns during training | Needs more data |
| **Custom Word2Vec** | Pre-trained on tweet corpus | Rich semantic meaning | Slower to train |

---

## 🧠 Neural Network Architectures

### Architecture 1 — ANN with Text Sequences
```
Input → Embedding → Flatten → Dense(64, ReLU) → Dense(1, Sigmoid)
```

### Architecture 2 — ANN + Embedding
```
Input → Embedding → GlobalAvgPool → Dense(64) → Dense(1, Sigmoid)
```

### Architecture 3 — LSTM + Embedding
```
Input → Embedding → LSTM(64) → Dense(1, Sigmoid)
```

### Architecture 4 — GRU + Embedding
```
Input → Embedding → GRU(64) → Dense(1, Sigmoid)
```

### Architecture 5 — Bidirectional GRU + Embedding
```
Input → Embedding → Bidirectional(GRU(64)) → Dense(1, Sigmoid)
```

### Architecture 6 — Bidirectional LSTM + Embedding ⭐ Best
```
Input → Embedding → Bidirectional(LSTM(64)) → Dropout(0.5)
      → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
```

### Architecture 7 — Conv1D + Embedding
```
Input → Embedding → Conv1D(128, kernel=5) → MaxPooling
      → Dense(64, ReLU) → Dense(1, Sigmoid)
```

### Architecture 8 — Custom Word2Vec + Conv1D
```
Input → Word2Vec Embedding Matrix → Conv1D(128)
      → GlobalMaxPool → Dense(64) → Dense(1, Sigmoid)
```

### Architecture 9 — Custom Word2Vec + Bidirectional LSTM
```
Input → Word2Vec Embedding Matrix → Bidirectional(LSTM(64))
      → Dense(32) → Dense(1, Sigmoid)
```

---

## 📈 Model Performance Comparison

| # | Model | Encoding | Accuracy |
|---|-------|----------|----------|
| 1 | ANN | One Hot Encoding | ~75% |
| 2 | ANN | Count Vectorizer | ~77% |
| 3 | ANN | TF-IDF | ~78% |
| 4 | ANN | Text Sequences + Embedding | ~80% |
| 5 | LSTM | Embedding | ~82% |
| 6 | GRU | Embedding | ~83% |
| 7 | Bidirectional GRU | Embedding | ~84% |
| 8 | **Bidirectional LSTM** | **Embedding** | **~85%** ⭐ |
| 9 | Conv1D | Embedding | ~83% |
| 10 | Conv1D | Custom Word2Vec | ~84% |
| 11 | **Bidirectional LSTM** | **Custom Word2Vec** | **~86%** 🏆 |

> *Best model: **Custom Word2Vec + Bidirectional LSTM** with ~86% accuracy*

---

## 📉 Visualizations in the Notebook

| Visualization | Details |
|---------------|---------|
| ☁️ **WordCloud — Positive** | Frequent words in positive tweets |
| ☁️ **WordCloud — Negative** | Frequent words in negative tweets |
| 📊 **Label Distribution** | Bar chart — perfectly balanced classes |
| 📏 **Word Length Distribution** | Histogram of tweet lengths |
| 📈 **Accuracy Curves** | Training vs Validation for each model |
| 📉 **Loss Curves** | Training vs Validation for each model |
| 🔥 **Confusion Matrix** | Heatmap for best model predictions |
| 📋 **Classification Report** | Precision, Recall, F1 per class |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| **Python 3.x** | Core programming language |
| **TensorFlow 2.x** | Deep learning framework |
| **Keras** | High-level neural network API |
| **NLTK** | Stopwords, tokenization, stemming |
| **Gensim** | Training custom Word2Vec embeddings |
| **NumPy** | Numerical computing & array operations |
| **Pandas** | Data loading and manipulation |
| **Matplotlib** | Training curve visualizations |
| **Seaborn** | Confusion matrix heatmaps |
| **WordCloud** | Text frequency word cloud |
| **Scikit-learn** | TF-IDF, train-test split, metrics |
| **re (regex)** | Text pattern cleaning |

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/simrankhan-coder/Deep-Learning-Projects.git
cd Deep-Learning-Projects
```



### 4️⃣ Download Dataset
- Visit 👉 [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Download `training.1600000.processed.noemoticon.csv`
- Place it in the same folder as the notebook

---

## 💡 Key Concepts Covered

- ✅ **NLP Fundamentals** — Text cleaning, tokenization, stemming
- ✅ **Exploratory Data Analysis** — WordCloud, length distribution, label balance
- ✅ **Classical Encoding** — One-Hot, Count Vectorizer, TF-IDF
- ✅ **Deep Encoding** — Keras Tokenizer + Padding + Embedding Layer
- ✅ **Word Embeddings** — Learnable embeddings & custom Word2Vec
- ✅ **Recurrent Networks** — LSTM, GRU for sequential text
- ✅ **Bidirectional RNNs** — BiLSTM & BiGRU for richer context
- ✅ **Convolutional NLP** — Conv1D for local pattern detection
- ✅ **Transfer Embeddings** — Word2Vec embedding matrix injection
- ✅ **Model Comparison** — 10+ models evaluated side-by-side

---

## 🔮 Future Improvements

- [ ] Fine-tune **BERT / RoBERTa** for higher accuracy
- [ ] Use **pre-trained GloVe** embeddings (Twitter-specific)
- [ ] Add **neutral sentiment** class (multi-class classification)
- [ ] Deploy best model as **Flask / FastAPI** web app
- [ ] Build **real-time classifier** using Twitter/X API
- [ ] Add **Attention Mechanism** on top of LSTM

---

## 👩‍💻 Author

**Simran Khan**
🔗 [GitHub](https://github.com/simrankhan-coder)

---

## ⭐ Support

If this project helped you learn NLP and Deep Learning, please give it a **⭐ Star** on GitHub!

---

