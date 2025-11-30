Sentiment Analysis Using Deep Learning (Without TF-IDF)
Project Description

This project performs sentiment analysis on Twitter data using a deep learning approach with LSTM networks, without using TF-IDF or traditional bag-of-words features. It classifies tweets into Positive, Neutral, and Negative sentiments.

The project focuses on text preprocessing, tokenization, word embedding, and sequential modeling using deep learning.

Dataset

Source: Twitter dataset containing tweets and sentiment labels

Columns:

clean_text – Preprocessed tweet text

category – Sentiment label: 1 = Positive, 0 = Neutral, -1 = Negative

Size: 186,517 tweets

Key Steps
1. Data Preprocessing

Converted all text to lowercase

Removed stopwords and punctuations

Applied Porter Stemming to reduce words to their root forms

Re-mapped negative sentiment label -1 to 2 for classification

2. Exploratory Data Analysis (EDA)

Visualized proportion of positive, neutral, and negative tweets using pie charts

Created word clouds for each sentiment category to understand common words

3. Tokenization and Padding

Tokenized the tweet text using Keras Tokenizer

Converted tokens into sequences and applied padding to ensure equal input length

4. Deep Learning Model

Used LSTM-based neural network for sequential modeling:

Input Layer → Embedding → LSTM → Dense → Dropout → Output Layer (Softmax)

Model trained to classify three sentiment categories

5. Model Training

Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Validation split: 20%

Training achieved 86.34% accuracy on test data

Technologies Used

Python – Programming language

Pandas, NumPy – Data handling

NLTK – Stopword removal and stemming

Matplotlib & Seaborn – Visualization

WordCloud – Visual representation of frequent words

TensorFlow/Keras – LSTM-based deep learning model

Google Colab – Execution environment

Model Architecture
Sequential LSTM Model:
Input Layer
→ Embedding Layer (trainable)
→ LSTM Layer (64 units, tanh activation)
→ Dense Layer (64 units, tanh activation)
→ Dropout Layer (0.2)
→ Dense Output Layer (3 units, softmax)


Total Parameters: ~17.78 million

Handles sequential input without TF-IDF or vectorization

Results

Training Accuracy: 88.89%

Validation Accuracy: 86.34%

Test Accuracy: 86.34%

The model effectively classifies tweets into Positive, Neutral, and Negative categories

Visualizations

Pie Chart: Distribution of sentiment categories

Word Clouds: Common words in Positive, Neutral, and Negative tweets

How to Run

Clone the repository

Upload Twitter_Data.csv in your Colab or local environment

Install required packages:

pip install pandas numpy nltk tensorflow matplotlib seaborn wordcloud


Run the notebook from top to bottom

Conclusion

This project demonstrates how LSTM-based deep learning models can perform sentiment analysis directly on text without using TF-IDF or other feature engineering techniques. The model achieved 86% test accuracy, making it suitable for social media sentiment classification tasks.
