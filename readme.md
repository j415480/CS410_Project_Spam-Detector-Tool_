# Overview
**JS -- CS410 -- Fall 2025 UIUC MS CS**

Hello! Welcome to my Spam Detector Tool.

My tool loads a Kaggle CSV dataset containing spam or legitimate (ham) messages, 
and attempts to predict each message's legitimacy.
My app uses Naive Bayes, Logistic Regression, and KNN (K=5) models.
During each run, you may choose a random seed to control data splitting.
You will also be presented with statistics and results.
You may also search messages using BM25 information-retrieval (search query, see BM25 score).
Thanks for checking out my tool!

# SMS Spam Detection and Information Retrieval Project

This project  builds a simple but complete pipeline for SMS spam detection using the **SMS Spam Collection** dataset (Kaggle / UCI)

The SMS Spam Collection dataset CSV file includes 5573 unique entries, where 87% are labeled ham, 13% labeled spam.

Main components:
- CSV data loading into project
- Text cleaning and preprocessing (same case letters, grammar/punctuation cleanup)
- Feature extraction with TF-IDF
- Models used: Multinomial Naive Bayes, Logistic Regression, K-Nearest Neighbors at K=5
- Results include model accuracy, precision, recall, F1 score, and a matrix
- BM25 showing model ranking of messages given a user-defined query
- And of course, there are attempts to mitigate edge cases (blank entries, non-numbers...)

## Setup

1. Open project folder ```cd ~/CS410_Project_Spam-Detector-Tool_-main``` (if ZIP downloaded)
2. In project terminal: ```pip install -r requirements.txt```
3. In project terminal: ```python main.py```
4. Interact with the terminal as requested by app

## Project Tree
```
CS410_Project_Spam-Detector-Tool_-main/
├─ data/
│  └─ kaggle-dataset.csv  # SMS Spam Collection dataset from Kaggle
├─ src/
│  ├─ bm25_searcher.py    # BM25 search to rank texts by query relevance
│  ├─ data_loader.py      # loads CSV into Pandas DF, build targets and labels
│  ├─ models.py           # TF-IDF implementation, model trainer and evaluator, results
│  └─ text_preprocess.py  # clean raw CSV text
├─ main.py                # implements pipeline - load, preprocess, split, train, show results
├─ README.md              # overview
└─ requirements.txt       # dependencies needed for project
```


## Packages used:
```
pandas       #dataframes and data loading
numpy        #math operations
scikit-learn #model processing, TF-IDF extraction, train/test data split
rank-bm25    #bm25 retrieval
kagglehub    #connecting to kaggle if local dataset not found
tqdm         #neat little progress bar during data cleaning
```

## Credits
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

