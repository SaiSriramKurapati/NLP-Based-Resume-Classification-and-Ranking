
# NLP-based Resume Classification and Ranking System

## Authors:
- Sai Sriram Kurapati
- Krishna Vamsi Gujjula
- College of Engineering and Computer Science, University of Central Florida, Orlando, Florida 32816

## Overview:
This project aims to address the challenge of managing large volumes of resumes in the job market. Our system uses Natural Language Processing (NLP) techniques to parse, categorize resumes, and match them to job descriptions, thus aiding in the identification of the most qualified candidates. By leveraging advanced NLP techniques, the system can reduce the time and effort required in the hiring process, enhance decision-making quality, and ensure a fair and objective evaluation of candidates.

## Installation:
1. Clone the repository to your local machine.
2. Ensure Python 3.x is installed.
3. Install necessary libraries: pip install -r requirements.txt.

## Usage
Prepare a dataset of resumes and job descriptions.
Run the main script: python resume_classification.py
The system will output classified and ranked resumes based on their relevance to the job descriptions.

## Dataset
The dataset used in this project is acquired from Kaggle and consists of 2208 resumes in .pdf or .docx format, along with a .csv file containing metadata for each resume​​.

## Techniques and Models Used
Pre-processing: Tokenization, stop word removal, stemming, and lemmatization.
Vectorization: Using CountVectorizer from scikit-learn.
Classification Models: Support Vector Classifier, Naive Bayes, Random Forest Classifier, Multi-Layer Perceptron, Gradient Boosting Classifier​​.
Evaluation: The models were evaluated using cross-validation techniques, with Gradient Boosting performing best in terms of accuracy, precision, recall, and F1 score​​.

## Evaluation and Results
The resume ranking algorithm's accuracy was evaluated to be approximately 92.5% based on the top five resumes for each category​​.
Gradient Boosting was found to be the most effective model for resume classification​​.

## Limitations and Future Work
Lack of available labeled data for the ranking task.
Reliance on keyword matching, which may not fully capture the context and nuances of candidates' abilities.
Future work includes expanding the dataset, improving data quality, and exploring deep learning techniques like neural networks for improved accuracy​​.
