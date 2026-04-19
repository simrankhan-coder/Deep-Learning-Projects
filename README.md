📌 Project Overview
This notebook is a deep-dive into Natural Language Processing using real Twitter data.
It covers everything from NLP fundamentals to advanced deep learning techniques —
making it a complete reference for building production-ready text classification systems.
The project explores and compares 10+ vectorization and neural network combinations,
from simple One-Hot Encoding all the way to custom-trained Word2Vec embeddings
with Bidirectional LSTM networks.

🎯 Objective
Build machine learning models that can accurately classify tweets as:
LabelSentimentCount4 ✅Positive800,000 tweets0 ❌Negative800,000 tweets

The dataset is perfectly balanced — 800K positive and 800K negative tweets —
eliminating class imbalance concerns entirely.


📊 Dataset
DetailInfoNameSentiment140SourceKaggle — Sentiment140Total Tweets1,600,000Class Balance50% Positive / 50% Negative (perfectly balanced)Positive Label4Negative Label0LanguageEnglish

📓 Notebook Structure
This notebook is organized into 4 major sections:

📁 Section 1 — Exploratory Data Analysis (EDA)
Sub-sectionDescription1.1 Loading the DatasetReading CSV into Pandas DataFrame1.2 Data CleaningRemoving noise, nulls, irrelevant columns1.3 Label DistributionVisualizing class balance (Positive vs Negative)1.4 Creating Word CloudMost frequent words in Positive & Negative tweets1.5 Word Length DistributionAnalyzing tweet length patterns

📁 Section 2 — Data Preparation for Model Building
Sub-sectionDescription2.1 Loading the DatasetPreparing full dataset for training2.2 Data CleaningText preprocessing — removing URLs, mentions, hashtags, stopwords, stemming

📁 Section 3 — Trying Different Neural Networks

This is the core section where 10 different model architectures are trained and compared.

Sub-sectionTechniqueType3.1Train / Test / Validation SplitData Splitting3.2One Hot Encoding + Model TrainingClassical Encoding3.3Count Vectorizer + Model TrainingBag of Words3.4TF-IDF + Model TrainingStatistical Weighting3.5ANN with Text SequencesDeep Learning3.6ANN + Embedding with Text SequencesEmbedding Layer3.7LSTM + Embedding with Text SequencesRecurrent Network3.8GRU + Embedding with Text SequencesGated Recurrent Unit3.9Bidirectional GRU + EmbeddingBidirectional RNN3.10Bidirectional LSTM + EmbeddingBidirectional RNN3.11Conv1D + Embedding with Text SequencesConvolutional Network

📁 Section 4 — Neural Networks with Custom Word2Vec
Sub-sectionDescription4.1Custom Word2Vec in Conv1D4.1.1Loading the dataset4.1.2Training Word2Vec model on tweet corpus4.1.3Creating Embedding Matrix from Word2Vec4.1.4Training Conv1D Neural Network with custom embeddings4.2Custom Word2Vec + Bidirectional LSTM
