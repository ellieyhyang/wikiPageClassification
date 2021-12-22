# wikiPageClassification  

## Description

The goal of this project is to use supervised machine learning models to automatically identify a text document as Court Case or Wikipedia Page. The two classifiers implemented are logistic regression and support vector machines (SVM). 

## Dataset

The training and testing data are files in .txt format, within which each line is a single document in a pseuda XML format. 

## Model Training

Both classifiers will be trained on the Term Frequency matrix that is built using the top 20k-most-frequent words from the entire training corpus. 
(Click [here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to learn more about Term Frequency)
