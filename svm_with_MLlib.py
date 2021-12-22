"""
Description: This script contains a pySpark program that used spark ml
             library to train a Support Vector Machine classifier on 
             data(Wiki pages) based on term frequency of the top 20k
             words of the corpus. 
"""

from __future__ import print_function
import re
import sys
import numpy as np
from operator import add
import psutil
import time
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LinearSVC


# ========================== User-Defined Functions  ===========================


# Build an array of size 20,000 with each position tells how many occurance of 
# the word in that position of the top-20-frequent-words dictionary
def buildArray(listOfIndices):

    returnVal = np.zeros(20000)
    
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    
    mysum = np.sum(returnVal)
    
    returnVal = np.divide(returnVal, mysum)
    
    return returnVal

# ============================= End of Functions  =============================


if __name__ == "__main__":

    # Log the program time
    start_time = time.time()

    # System Arguments: 
    # [1]: Training Data
    # [2]: Testing Data

    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)

    # ============================= Data Preprocessing  =============================

    # Log the starting time for reading and preparing training and testing datatsets
    start_data_read = time.time()

    # use a regular expression here to check for non-letter characters
    regex = re.compile('[^a-zA-Z]')

    '''------------------- Training Set -------------------'''

    # - STEP 1: Build the top 20k-words dictionary

    # Load file into an RDD
    d_corpus = sc.textFile(sys.argv[1], 1)

    # Transform both into a set of (docID, text) pairs
    d_keyAndText = d_corpus\
                .map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))


    # remove all non letter characters
    # Split the text in each (docID, text) pair into a list of words
    # Resulting RDD is a dataset with (docID, ["word1", "word2", "word3", ...])
    d_keyAndListOfWords = d_keyAndText\
                .map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))


    # Build Top-20 Word Dictionary from the training set
    # map (docID, ["word1", "word2", "word3", ...])
    # to ("word1", 1) ("word2", 1)...
    allWords = d_keyAndListOfWords\
                .map(lambda x: (x[1]))\
                .flatMap(lambda x: [w for w in x])\
                .map(lambda x: (x, 1))
    allWords.cache()

    # count all of the words, giving --> ("word1", 1433), ("word2", 3423423), etc.
    allCounts = allWords.reduceByKey(lambda x, y: x+y)

    # Get the top 20,000 words in a local array in a sorted format based on frequency
    topWords = allCounts.top(20000, key=lambda x: x[1])
    topWords = np.array(topWords)

    # Create a RDD that has a set of (word, dictNum) pairs
    # start by creating an RDD that has the number 0 through 19999
    # 20000 is the number of words that will be in our dictionary
    topWordsK = sc.parallelize(range(20000))

    # Then, transform (0), (1), (2), ... to ("MostCommonWord", 1) ("NextMostCommon", 2), ...
    # the number will be the spot in the dictionary used to tell us where the word is located
    dictionary = topWordsK.map(lambda x: (topWords[x][0], x))

    # Will be using Map-Side Join Operation
    # Collect the small RDD as Map (a dict in python)
    dictionaryAsMap = dictionary.collectAsMap()

    # broad cast this to all worker nodes. 
    sc.broadcast(dictionaryAsMap)


    # - STEP 2: Build TF matrix for the Training Set 

    # Create a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...
    allWordsWithDocID = d_keyAndListOfWords\
                        .flatMap(lambda x: ((j, x[0]) for j in x[1]))


    # Then do a simple map on it to get a set of (word, (dictionaryPos, docID)) pairs
    allDictionaryWords = allWordsWithDocID\
                            .map(lambda x: (x[0], (x[1], dictionaryAsMap.get(x[0]))) 
                             if x[0] in dictionaryAsMap.keys() else None)\
                            .filter(lambda x: x!=None)\
                            .map(lambda x: (x[0], (x[1][1], x[1][0])))
    allDictionaryWords.cache()

    # Drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

    # Create a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

    # Converts the dictionary positions to a bag-of-words numpy array...
    # use the buildArray function to build the feature array
    # this gives a set of (docID, featureArray)
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc\
                            .map(lambda x: (x[0], buildArray(x[1])))

    # - STEP 3: Label Training Data

    # Create an RDD that has (label, TFArray) pairs for all documents
    # labeling: 1 if docID starts with 'AU' (Australia court case) and 0 if others
    # Then map it to RDD of (label, Vector.dense) pairs
    docLabelAndTFArray = allDocsAsNumpyArrays\
                        .map(lambda x: (1, x[1]) if x[0][:2] == 'AU' else (0, x[1]))\
                        .map(lambda x: (x[0], Vectors.dense(x[1])))

    # Convert it into dataframe
    labelTFVectorDF = sqlContext.createDataFrame(docLabelAndTFArray, ['label', 'features'])
    # labelTFVectorDF.show(10)

    # cache this Dataframe (will be used to learn the model)
    labelTFVectorDF.cache()


    '''------------------- Testing Set ------------------- '''

    # - STEP 1: Build TF feature matrix for the Training Set 
    # (using the same Dictionary as trarining set)

    # Load file into an RDD
    test_corpus = sc.textFile(sys.argv[2], 1)

    # Transform it into a set of (docID, text) pairs
    test_keyAndText = test_corpus\
                .map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))

    # remove all non letter characters
    # Split the text in each (docID, text) pair into a list of words
    # Resulting RDD is a dataset with (docID, ["word1", "word2", "word3", ...])
    test_keyAndListOfWords = test_keyAndText\
                .map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))


    # Create a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...
    test_allWordsWithDocID = test_keyAndListOfWords\
                                .flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Then do a simple map on it to get a set of (word, (dictionaryPos, docID)) pairs
    test_allDictionaryWords = test_allWordsWithDocID\
                                .map(lambda x: (x[0], (x[1], dictionaryAsMap.get(x[0]))) 
                                 if x[0] in dictionaryAsMap.keys() else None)\
                                .filter(lambda x: x!=None)\
                                .map(lambda x: (x[0], (x[1][1], x[1][0])))

    # Drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    test_justDocAndPos = test_allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

    # Create a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    test_allDictionaryWordsInEachDoc = test_justDocAndPos.groupByKey()

    # Converts the dictionary positions to a bag-of-words numpy array...
    # use the buildArray function to build the feature array
    # this gives a set of (docID, featureArray)
    test_allDocsAsNumpyArrays = test_allDictionaryWordsInEachDoc\
                                    .map(lambda x: (x[0], buildArray(x[1])))


    # Create an RDD that has (label, TFArray) pairs for all documents
    # labeling: 1 if docID starts with 'AU' (Australia court case) and 0 if others
    # Then map it to RDD of (label, Vector.dense) pairs
    test_docLabelAndTFArray = test_allDocsAsNumpyArrays\
                                .map(lambda x: (1, x[1]) if x[0][:2] == 'AU' else (0, x[1]))\
                                .map(lambda x: (x[0], Vectors.dense(x[1])))

    # Convert test RDD into dataframe
    test_labelTFVectorDF = sqlContext.createDataFrame(test_docLabelAndTFArray, ['label', 'features'])
    # test_labelTFVectorDF.show(10)
    test_labelTFVectorDF.cache()

    # Log the end time for reading and preparing training and testing datatsets
    end_data_read = time.time()

    print("\nTime(sec) taken to read and preprocess both training and testing datasets:", end_data_read - start_data_read)

    # ========================== End of Data Preprocessing  ==========================

    
    # ==================== Implementing SVM Model  ====================

    '''------------------- Model Training ------------------- '''

    # Log the start time
    start_model_train = time.time()

    # Train the model
    svm_model = LinearSVC(maxIter=100).fit(labelTFVectorDF)

    # log the end time
    end_model_train = time.time()
    print("\nTime(sec) taken to train the SVM model:", end_model_train - start_model_train)


    '''------------------- Model Testing ------------------- '''

    # Log the start time
    start_model_test = time.time()

    # Make predictions for the test sets
    # Resulted RDD contains (pred, label) pairs
    test_predAndLabel =  svm_model.transform(test_labelTFVectorDF)\
                           .select(['prediction', 'label'])\
                           .rdd\
                           .map(lambda x: (float(x[0]), float(x[1])))


    # Store the performance metrics
    test_metrics = MulticlassMetrics(test_predAndLabel)

    # Print the results
    print("\nModel testing results:")
    print("Confusion Matrix (0, 1): \n", test_metrics.confusionMatrix().toArray())
    print("Accuracy:", test_metrics.accuracy)
    print("Precision:", test_metrics.precision(1.0))
    print("Recall:", test_metrics.recall(1.0))
    print("F-score", test_metrics.fMeasure(1.0))

    # Log the end time
    end_model_test = time.time()
    print("\nTime(sec) taken to test the model:", end_model_test - start_model_test)


    sc.stop()

    # Log the program end time
    end_time = time.time()
    print("Time(sec) taken for the entire process:", end_time - start_time)
