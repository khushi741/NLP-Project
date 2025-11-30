# Sentiment Analysis Using Deep Learning (Without TF-IDF)

## Project Description
This project performs **sentiment analysis on Twitter data** using a deep learning approach with **LSTM networks**, without using TF-IDF or traditional bag-of-words features.  
It classifies tweets into **Positive, Neutral, and Negative** sentiments.

Key objectives include:  
- Text preprocessing (stopword removal, punctuation cleaning, stemming)  
- Tokenization and sequence padding  
- Building an LSTM-based deep learning model  
- Visualizing sentiment distribution and common words  
- Predicting sentiment categories of tweets  

---

## Dataset
- **Source:** Twitter dataset  
- **Columns:**  
  - `clean_text` – Preprocessed tweet text  
  - `category` – Sentiment label: `1 = Positive`, `0 = Neutral`, `-1 = Negative`  
- **Size:** 186,517 tweets  

---

## Technologies Used
- **Python** – Programming language  
- **Pandas & NumPy** – Data handling  
- **NLTK** – Stopword removal and stemming  
- **Matplotlib & Seaborn** – Visualization  
- **WordCloud** – Visual representation of frequent words  
- **TensorFlow/Keras** – LSTM-based deep learning model  
- **Google Colab** – Execution environment  

---

## How It Works

### 1. Data Preprocessing
- Convert all text to lowercase  
- Remove stopwords and punctuations  
- Apply Porter Stemming to reduce words to root forms  
- Map negative sentiment `-1` to `2` for classification  

### 2. Exploratory Data Analysis (EDA)
- Visualize proportion of positive, neutral, and negative tweets using **pie charts**  
- Generate **word clouds** for each sentiment category  

### 3. Tokenization and Padding
- Tokenize tweets using Keras Tokenizer  
- Convert tokens to sequences and pad them to a fixed length  

### 4. LSTM-Based Model
- **Architecture:**  
  - Input Layer → Embedding Layer → LSTM Layer (64 units) → Dense Layer (64 units) → Dropout (0.2) → Dense Output Layer (3 units, Softmax)  
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Validation Split:** 20%  
- **Training Result:** 86.34% accuracy on test data  

---

## Example Visualizations
- **Pie Chart:** Distribution of sentiment categories  
- **Word Clouds:** Most frequent words in Positive, Neutral, and Negative tweets  

---

## How to Run
1. Clone the repository  
2. Upload `Twitter_Data.csv` to your Colab or local environment  
3. Install required packages:
```bash
pip install pandas numpy nltk tensorflow matplotlib seaborn wordcloud
Run the notebook from top to bottom

Results
Training Accuracy: 88.89%

Validation Accuracy: 86.34%

Test Accuracy: 86.34%

The model effectively classifies tweets into Positive, Neutral, and Negative categories

Conclusion
This project demonstrates how LSTM-based deep learning models can perform sentiment analysis directly on text without TF-IDF or manual feature engineering.
The model achieved 86% test accuracy, making it suitable for social media sentiment classification tasks.

