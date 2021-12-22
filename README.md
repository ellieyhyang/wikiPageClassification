# wikiPageClassification  

## Description

The goal of this project is to use supervised machine learning models to automatically identify a text document as Court Case or Wikipedia Page. The two classifiers implemented are logistic regression and support vector machines (SVM). 

## Dataset

The training and testing data are files in *.txt* format, within which each line is a single document in a pseud0 XML format. 

## Model Training

1. Both classifiers will be trained on the Term Frequency (TF) matrix of the training set built using the top 20k-most-frequent words from the training corpus. 
(Click [here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to learn more about Term Frequency)

2. The TF matrix is reduced to 10k-dimension based on feature variance. The same models are re-implemented. 

(see the README.md file in the model folders for details)

## Model Testing

Both models will be tested using the Term Frequency matrix of the testing set build using the same bag of words from the training set. 

(see the README.md file in the model folders for details)


## Folders

* __Classification-Models__: contains PySpark implementations of the classification models on the 20k-dimension term frequency matrix

* __Classification-Models-Reduced-Feature-Matrix__: contains PySpark implementations of the classifications models on the reduced term frequency matrix

* __data__: contains a small dataset for testing the model implementations and links to the full datasets

* __notebooks__: contains Jupyter Notebooks that show examples of running the codes

