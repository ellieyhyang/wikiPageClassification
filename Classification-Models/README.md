# Classifications on 20k-Dimension TF matrix


## Description

## Scripts

__1. logistic_regression_with_gradient_descent.py__: 
* Generate the TF frequency
* Train a Logistic Regression model using Gradient Descent algorithm
* Test and evaluate the model



__2. logistic_regression_with_MLlib.py__: 
* Generate the TF frequency
* Use 'LinearSVC' module from the spark ML library to train a Support Vector Machine classifier
* Test and evaluate the model


## Obtaining the Dataset

Training Dataset (consists of ~ 170,000 text documents)

Testing Dataset (consists of ~ 18,700 text documents)


## Python Scripts


## Instructions to run

Make sure that you have Download and configured Apache Spark on your machine. 

### Running locally on an IDE (e.g Pycharm) - Not recommended
Download the dataset to your local disk. Clone the script and paste it to your IDE, substitute all arugument fields (sys.argv[?]) with corresponding path.

### Running on a Cloud Service (e.g Google Cloud or Amzon AWS)
Upload the script to your Cloud Drive. When submitting a job, supply the internal path of the script, pass in the corresponding interal URLs for the required arguments. 
