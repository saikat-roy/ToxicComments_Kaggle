# ToxicComments_Kaggle
Solutions to the Toxic Comments on Kaggle. (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Dependencies
1) Python 3.7
2) NumPy
3) SciPy
4) Pandas
5) scikit-learn

## Data Requirements
Requires the "train.csv" and "test.csv" from the Kaggle Toxic Comment Classification Challenge Page.

## Basic Methodology
1) Perform basic Cleaning by removing the Reference Text from the Comment Body and non-alpha numeric characters.
2) Vectorization using BOW model, or Tfidf vectorizer (scikit-learn)
3) Trains individual models for each comment class (toxic, severe_toxic, obscene, threat, insult, identity_hate). Class balancing is handled during training weighted updates instead of manual sample balancing.
4) Predictions are handled for each class separately similar to training.

## Algorithms Implemented with Scores on Kaggle
1) Logistic Regression - CountVectorizer - 0.8917
2) Logistic Regression - TfidfVectorizer - 0.8964
