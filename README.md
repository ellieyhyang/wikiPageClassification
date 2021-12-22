# wikiPageClassification  

## Description

The goal of this project is to use supervised machine learning models to automatically identify a text document as Court Case or Wikipedia Page. The two classifiers implemented are logistic regression and support vector machines (SVM). 

## Dataset

The training and testing data are files in .txt format, within which each line is a single document in a pseuda XML format. 

## Model Training

Both classifiers will be trained on the Term Frequency matrix of the training set built using the top 20k-most-frequent words from the training corpus. 
(Click [here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to learn more about Term Frequency)

## Model Testing

Both modles will be tested using the Term Frequency matrix of the testing set build using the same bag of 20k words. 


## Folders

* Classification-Models: PySpark implementations of the classification models on the 20k-dimension term frequency matrix
* Classification-Models-Reduced-Feature-Matrix: PySpark implementations of the classifications models on the reduced term frequency matrix
* data: contains a small dataset for testing the model implementations and links to the full datasets
* notebooks: contains Jupyter Notebooks that demostrate examples of running the codes
