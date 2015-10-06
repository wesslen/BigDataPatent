# BigDataPatent
Text Mining Patents for Big Data Course Project


# Introduction
This readme outlines the steps in Python to text mine Patents for 3M.

There are four sections (so far) of the code:
1. Modules & Working Directory
2. Exploratory Analysis of a Sample of Abstracts
3. Data Wrangling (Tokenize, Clean, TF-IDF)
4. Topic Modeling (K-Means, LDA, Topic Word Cloud)
5. 

# 1. Modules & Set Working Directory

## For Python, you must call in modules that you will use later on.


```python

import re
import os

import nltk
from nltk.stem.porter import PorterStemmer  #Stemmer
from nltk.stem import WordNetLemmatizer # Lemmatization
#nltk.download() # download stopwords datasets when prompted (in Corpora Tab)
from nltk.corpus import stopwords # Import the stop word list
#print stopwords.words("english") 

import matplotlib.pyplot as plt
import pandas as pd

import gensim
from gensim import corpora, models, similarities

```

## Set home directory and load csv file

```python

#Set to your directory
os.chdir("/home/ryanceros/Dropbox/Project - Big Data Analytics/WordCloud")

#CSV File Name
dataset = '3M onlyRyan.csv'

#Load CSV File
exampleData = pd.read_csv(dataset, header=None)

#Read in Header Name
exampleData.columns = ["PatentNumber","CompanyName","PatentAssignee",
"YearGranted","YearApplied","PatentClass","PatentTitle","PatentAbstract"]

#Check shape and column names
exampleData.shape
#(1559, 8)

exampleData.columns.values
#array(['PatentNumber', 'CompanyName', 'PatentAssignee', 'YearGranted',
#'YearApplied', 'PatentClass', 'PatentTitle', 'PatentAbstract'], dtype=object)
```


# 2. Exploratory Analysis of Abstracts

## Four Examples of Abstracts
A normal, non-normal, missing and duplicated abstract  

```python

#Normal: Example of a Visual/Imaging Patent
print exampleData["PatentAbstract"][22]
#Systems and methods for improving visual attention models use effectiveness 
#assessment from an environment as feedback to improve visual attention models.
#The effectiveness assessment uses data indicative of a particular behavior

#Non-Normal: Example of an Abbreviated / Techincal Abstract
print exampleData["PatentAbstract"][0]
#Azide compositions selected from i) UOCR2CR2N3 wherein U is D

#But a handful (about 10%) are missing abstracts, for example: 
print exampleData["PatentAbstract"][24]
#nan

#Remove patents with a missing Abstract
exampleData.dropna(subset=['PatentAbstract'], inplace=True)
#1,559 to 1,381 patents

#About 20 are duplicated, for example: 
print exampleData["PatentAbstract"][956:973]
#Pharmaceutical formulations and methods includ... (multiple lines...)
```

## Run StemLemma.py (Stemmer, Lemmatization functions)

[Reference Document](http://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization)

## Two examples of old and new abstracts after cleaning.
```python

#Normal, changed from 24 to 20 due to dropped NaN abstracts
 
print(patent_to_words(exampleData["PatentAbstract"][20]))
 
#Cleaned (New):
#system method improve visual attention model use effectiveness 
#assessment environment feedback improve visual attention model 
#effectiveness assessment us data indicative particular behavior 
 
#Old:
#Systems and methods for improving visual attention models use effectiveness 
#assessment from an environment as feedback to improve visual attention models.
#The effectiveness assessment uses data indicative of a particular behavior
 
#Example of Abbreviated / Technical Abstract 
 
 print(patent_to_words(exampleData["PatentAbstract"][0]))
 
#Cleaned (New):
#azide composition select uocr cr n wherein u d

#Old:
#Azide compositions selected from i) UOCR2CR2N3 wherein U is D 

```

# 3. Data Wrangling

## Clean and Tokenize patents into lists (each patent is a words array)

```python

# Get the number of reviews based on the dataframe column size
num_patents = exampleData["PatentAbstract"].size

# Initialize an empty list to hold the clean reviews
clean_abstracts = []

# Loop over each review; create an index i that goes from 0 to the length
# of the patent list 
for i in xrange( 0, num_patents ):
    # Call our function for each one, and add the result to the list of
    # clean abstracts
    patent = patent_to_words(exampleData["PatentAbstract"][i])
    array = patent.split()
    clean_abstracts.append(array)
```


## Convert tokenized document to dictionary and document-term matrix

[Reference Document](https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html) 

```python

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(clean_abstracts)
    
# convert tokenized documents into a document-term matrix (bag-of-words)
corpus = [dictionary.doc2bow(text) for text in clean_abstracts]
```

## Term Frequency and Inverse Document Frequency (TF-IDF)

```python

#TF IDF
tfidf = models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]
```

# 4. Topic Modeling

## Run KMeans.py to create KMeans function and to Determine Number of Topics

[Reference Document](http://sujitpal.blogspot.com/2014/08/topic-modeling-with-gensim-over-past.html)

## Generate LDA Model using gensim

[Reference Document for gensim module][https://github.com/piskvorky/gensim]

## Generate Word Clouds for each Topic

[Reference Document](http://sujitpal.blogspot.com/2014/08/topic-modeling-with-gensim-over-past.html)
