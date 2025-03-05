# Amazon Alexa Reviews Sentiment Analysis

## Overview
This project analyzes Amazon Alexa reviews to understand customer sentiment, preferences, and pain points. The analysis includes data preprocessing, exploratory data analysis, sentiment analysis, and machine learning modeling.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn

## Analysis
Data Preprocessing
- Loaded the Amazon Alexa reviews dataset.
- Performed data cleaning and preprocessing, including removing non-alphabetic characters, converting to lowercase, and stemming.

## Exploratory Data Analysis
- Analyzed the distribution of ratings, feedback, and variations.
- Created bar charts and histograms to visualize the distributions.
- Calculated descriptive statistics, such as mean, median, and standard deviation.

## Sentiment Analysis
- Performed sentiment analysis on the text reviews using NLTK's VADER sentiment analyzer.
- Created word clouds to visualize the sentiment of positive and negative reviews.

## Machine Learning Modeling
- Split the data into training and testing sets.
- Trained multiple machine learning models, including logistic regression, naive Bayes, support vector machine, random forest, and gradient boosting.
- Evaluated the performance of each model using metrics such as precision, recall, F1-score, and AUC-ROC.
- Identified the best-performing model as random forest.

## Hyperparameter Tuning
- Performed hyperparameter tuning on the random forest model using RandomizedSearchCV.
- Identified the optimal hyperparameters and re-trained the model.

## Results
- The random forest model achieved a precision of 0.95, recall of 0.98, F1-score of 0.97, and AUC-ROC of 0.85.
- The sentiment analysis revealed that the majority of reviews were positive, with a sentiment score of 0.75.

## Conclusion
This project demonstrated the use of natural language processing and machine learning techniques to analyze Amazon Alexa reviews. The results provide insights into customer sentiment, preferences, and pain points, which can be used to improve the product and customer experience.
