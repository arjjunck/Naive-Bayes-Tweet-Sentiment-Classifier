# Tweet Sentiment Classification using Multinomial Naive Bayes

## Project Overview

This project implements a **Multinomial Naive Bayes classifier from scratch** in Python to classify tweets as either **positive** or **negative**.

The model is trained using separate positive and negative tweet datasets. It builds a vocabulary from the training data, calculates word frequencies, applies Laplace smoothing, and classifies unseen tweets using log-probabilities.

This project demonstrates the core mathematical logic behind text classification without relying on a pre-built machine learning model from Scikit-learn.

## Objective

The main objective of this project is to understand and implement the **Multinomial Naive Bayes learning algorithm** for tweet sentiment classification.

The project focuses on:

- Loading positive and negative tweet datasets
- Cleaning and preprocessing tweet text
- Building a vocabulary from training data
- Calculating word frequencies for each class
- Applying Laplace smoothing
- Computing class prior probabilities
- Classifying tweets using log-probabilities
- Evaluating positive, negative and average accuracy

## Problem Type

This is a **binary text classification problem**.

The two target classes are:

| Label | Meaning |
|---|---|
| `pos` | Positive tweet |
| `neg` | Negative tweet |

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Regular Expressions
- Math library

## Techniques Applied

- Text preprocessing
- Tokenisation
- Vocabulary creation
- Word frequency counting
- Multinomial Naive Bayes
- Laplace smoothing
- Prior probability calculation
- Log-probability scoring
- Sentiment classification
- Accuracy evaluation

## Project Workflow

```text
Load training and test tweet files
        |
        v
Preprocess tweets
- Convert to lowercase
- Remove non-alphabetic characters
- Split into words
        |
        v
Build vocabulary
        |
        v
Count word frequencies
- Positive word counts
- Negative word counts
        |
        v
Calculate probabilities
- Word likelihoods
- Class priors
- Laplace smoothing
        |
        v
Classify test tweets
        |
        v
Evaluate model performance
```

## Data Loading

The notebook loads four text files:

```text
trainPos.txt
trainNeg.txt
testPos.txt
testNeg.txt
```

These files represent:

- Positive training tweets
- Negative training tweets
- Positive test tweets
- Negative test tweets

Each file is loaded into a Pandas DataFrame with one tweet per row.

## Text Preprocessing

Each tweet is preprocessed using the following steps:

1. Convert text to lowercase
2. Remove non-alphabetic characters
3. Split the tweet into individual words

Example preprocessing logic:

```python
def preprocess(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^a-z\s]', '', tweet)
    words = tweet.split()
    return words
```

This helps standardise the text before building the vocabulary and calculating word frequencies.

## Vocabulary and Word Frequency

The model creates a vocabulary containing all unique words found in the training tweets.

For each word, the algorithm stores:

- Frequency in positive tweets
- Frequency in negative tweets

If a word appears in one class but not the other, its frequency is set to zero for the missing class.

## Multinomial Naive Bayes

The classifier uses the Multinomial Naive Bayes algorithm.

For each class, the probability of a tweet is calculated using:

```text
P(class | tweet) ∝ P(class) × P(word1 | class) × P(word2 | class) × ...
```

Because multiplying many small probabilities can cause numerical underflow, the model uses log-probabilities:

```text
log P(class | tweet) = log P(class) + Σ log P(word | class)
```

The tweet is assigned to the class with the higher log-probability.

## Laplace Smoothing

Laplace smoothing is used to handle unseen words and avoid zero probabilities.

The formula used is:

```text
P(word | class) = (word_count + 1) / (total_words_in_class + vocabulary_size)
```

This ensures that every word has a small probability, even if it did not appear in a particular class during training.

## Model Evaluation

The model was evaluated separately on positive and negative test tweets.

### Results

| Metric | Score |
|---|---:|
| Vocabulary size | 490,317 |
| Total positive words | 5,012,555 |
| Total negative words | 5,368,482 |
| Positive accuracy | 73.90% |
| Negative accuracy | 82.00% |
| Average accuracy | 77.95% |

The model performed better on negative tweets than positive tweets in the current test run.

## Key Findings

- A from-scratch Multinomial Naive Bayes classifier can achieve reasonable sentiment classification performance.
- The final average accuracy was approximately **77.95%**.
- Laplace smoothing is important because test tweets may contain words not seen in one of the training classes.
- Log-probabilities are necessary to avoid numerical underflow when multiplying many small word probabilities.
- The model is simple, interpretable and useful for understanding the foundations of NLP classification.

## Suggested Repository Structure

```text
tweet-sentiment-naive-bayes/
│
├── README.md
├── requirements.txt
├── .gitignore
├── tweet_sentiment_naive_bayes.ipynb
│
└── data/
    ├── train/
    │   ├── trainPos.txt
    │   └── trainNeg.txt
    │
    └── test/
        ├── testPos.txt
        └── testNeg.txt
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tweet-sentiment-naive-bayes.git
cd tweet-sentiment-naive-bayes
```

### 2. Install dependencies

```bash
pip install pandas numpy jupyter
```

### 3. Add the dataset

Place the tweet text files inside the `data/` folder:

```text
data/train/trainPos.txt
data/train/trainNeg.txt
data/test/testPos.txt
data/test/testNeg.txt
```

### 4. Update file paths

Inside the notebook, update the file paths:

```python
train_pos = 'data/train/trainPos.txt'
train_neg = 'data/train/trainNeg.txt'
test_pos = 'data/test/testPos.txt'
test_neg = 'data/test/testNeg.txt'
```

### 5. Run the notebook

```bash
jupyter notebook
```

Open and run:

```text
tweet_sentiment_naive_bayes.ipynb
```

## Requirements

Create a `requirements.txt` file with:

```text
pandas
numpy
jupyter
```

## Files to Remove Before Publishing

Remove unnecessary system or temporary files if they appear:

```text
.DS_Store
.ipynb_checkpoints/
__pycache__/
```

Add this `.gitignore` file:

```text
.DS_Store
.ipynb_checkpoints/
__pycache__/
*.pyc
.env
```

## Limitations

- The preprocessing is basic and removes punctuation, numbers and special characters.
- Stopwords are not removed.
- Stemming or lemmatisation is not applied.
- The model does not handle negation very well, for example “not good”.
- Word order is ignored because Naive Bayes treats words independently.
- The classifier is based on word frequency only and does not use advanced embeddings.
- The dataset paths are currently local and should be updated before publishing.

## Future Improvements

- Add stopword removal.
- Add stemming or lemmatisation.
- Compare with Scikit-learn’s `MultinomialNB`.
- Add TF-IDF vectorisation.
- Add confusion matrix and classification report.
- Compare with Logistic Regression and SVM.
- Save predictions to a CSV file.
- Build a simple Streamlit app for live tweet sentiment prediction.
- Add support for custom user input.

## Project Status

Completed as a basic NLP and machine learning classification project.

## Author

**Arjun Krishna Krishnakumar**

Aspiring AI/ML and software developer with an interest in machine learning, natural language processing, text classification and practical AI applications.

## License

This project is for educational and portfolio purposes.
