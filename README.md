# BigDataPatent
Text Mining Patents for Big Data Course Project


# Introduction
This readme outlines the steps in Python to text mine Patents for 3M.


# Modules
For Python, you must call in modules that you will use later on.


```
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

# Set home directory and load csv file

```
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


# Exploratory Analysis of Abstracts
Look at four examples of Abstracts: normal, non-normal, missing and duplicated.  

```
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

# Run StemLemma.py (Stemmer, Lemmatization functions)

# Two examples of old and new abstracts after cleaning.
```
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
 
 Abbreviated / Techincal Abstract 
 
 print(patent_to_words(exampleData["PatentAbstract"][0]))
 
#Cleaned (New):
#azide composition select uocr cr n wherein u d

#Old:
#Azide compositions selected from i) UOCR2CR2N3 wherein U is D 

```

# Clean and Tokenize patents into lists (each patent is a words array)

# Convert tokenized document to dictionary and document-term matrix

# Term Frequency and Inverse Document Frequency (TF-IDF)

# Run KMeans.py to create KMeans function and to Determine Number of Topics

# Generate LDA Model using gensim

# Generate Word Clouds for each Topic
