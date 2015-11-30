# Patent Project for Big Data for Competitive Advantage (DSBA 6140)
Topic Modeling Patents for Big Data Course Project


## Introduction
This readme outlines the steps in Python to text mine Patents for 3M and seven competitors.

There are five sections of the code:

1.  Modules & Working Directory
2.  Load Dataset, Set Column Names and Sample (Explore) Data
3.  Data Wrangling (Tokenize, Clean, TF-IDF)
4.  Topic Modeling (K-Means, LDA, Topic Word Cloud)
5.  K-Means Clustering on the Topic Probabilities

This code was created as a collection of several online references. Each reference is labelled with a [number] tag that will be used through this document to cite when a reference was used to create the section of code.

Somewhat technical:
*  [1] [Introduction to Bag-of-Words modeling in Python](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)
*  [2] [Applications and Challenges of Text Mining with Patents](http://ceur-ws.org/Vol-1292/ipamin2014_paper4.pdf)
*  [3] [Topic Modeling Visualizations](https://de.dariah.eu/tatom/topic_model_visualization.html)
*  [4] [Stemming & Lemmatization](http://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization)

Very technical:
*  [5] [NLTK Homepage](http://www.nltk.org/)
*  [6] [gensim Homepage](http://radimrehurek.com/gensim/index.html)
*  [7] [LDA Modeling for Python](https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html)
*  [8] [Topic Modeling (& K-Means) with Gensim](http://sujitpal.blogspot.com/2014/08/topic-modeling-with-gensim-over-past.html)
*  [9] [Constructing a broad-coverage lexicon for text mining in the patent domain](http://www.lrec-conf.org/proceedings/lrec2010/pdf/378_Paper.pdf)
*  [10] [Document Clustering with Similarity](http://brandonrose.org/clustering#Tf-idf-and-document-similarity)
*  [11] [DBSCAN](http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html)
*  [12] [Identifying Bigrams using gensim](https://radimrehurek.com/gensim/models/phrases.html)


## 1. Modules & Set Working Directory

### Import Modules
For Python, you must call in modules that you will use later on.

```python

# from nltk
from nltk.corpus import stopwords # Import the stop word list
import nltk
from nltk.stem.porter import PorterStemmer  
from nltk.stem import WordNetLemmatizer # Lemmatization
 
# modules that are used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import csv
import logging
from scipy.spatial.distance import cdist, pdist
from wordcloud import WordCloud
 
# import gensim
import gensim
from gensim import corpora, models, similarities

```

### Home Directory
Set home directory

```python

# home directory - modify to your own homedirectory
os.chdir("/home/ryanceros/Dropbox/Project - Big Data Analytics/WordCloud")
```


## 2. Load Dataset, Set Column Names and Sample Data

### Load Dataset and Set Column Names
The dataset containing nearly 33,000 patents for eight companies is named "updatedCompanies.csv".

```python
# sample dataset
dataset = 'updatedCompanies.csv'
 
# Load dataset 
rawData = pd.read_csv(dataset, header=None)
 
# Rename the columns as CSV does not contain headers 
rawData.columns = ["PatentNumber","CompanyName","Company","PatentAssignee",
"YearGranted","YearApplied","Year","PatentClass1","PatentClass2","PatentClassClean",
"ClassName","PatentTitle","PatentAbstract"]
     
# Check Shape 
exampleData.shape
# ((32405, 13)
 
exampleData.columns.values
#array(['PatentNumber', 'CompanyName', 'PatentAssignee', 'YearGranted',
#       'YearApplied', 'PatentClass1', 'PatentClass2', 'ClassName',
#       'PatentTitle', 'PatentAbstract'], dtype=object)
```

## Run StemLemma.py (Stemmer, Lemmatization functions)
[4] was modified to create the script StemLemma.py

```python
exec(open("StemLemma.py").read())
```

## Sample (Explore) Data
Using an example (num = 560), we explore fields about the patent including what the Abstract Bag-of-Words looks like.

```python
 num=560
 print "Company Name:  %s" % (exampleData["Company"][num]) 
 print("")
 print("Patent Title: " + exampleData["PatentTitle"][num]) 
 print("")
 print("Class Name: " + exampleData["ClassName"][num]) 
 print("")
 print("Class Number (Left 3): %s " % exampleData["PatentClassClean"][num])  
 print("")
 print("Abstract: " + exampleData["PatentAbstract"][num]) 
 print("")
 print("Abstract Bag of Words: " + patent_to_words(exampleData["PatentAbstract"][num])) 

```

## 3. Data Wrangling

### Clean and Tokenize patents into lists (each patent is a words array)

```python

# Get the number of reviews based on the dataframe column size
num_patents = exampleData["PatentAbstract"].size
 
# Initialize an empty list to hold the clean reviews
clean_abstracts = []
 
# Loop over each review; create an index i that goes from 0 to the length
# of the patent list 
for i in xrange( 0, num_patents ):
    # Call our function for each one, and add the result to the list of
    patent = patent_to_words(exampleData["PatentAbstract"][i])
    array = patent.split()
    clean_abstracts.append(array)

# Identify Bigrams using gensim's Phrases function
bigram = models.Phrases(clean_abstracts)
 
final_abstracts = []
 
for i in xrange(0,num_patents):
    sent = clean_abstracts[i] 
    temp_bigram = bigram[sent]
    final_abstracts.append(temp_bigram)
 
```


### Convert tokenized document to dictionary and document-term matrix

```python
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(final_abstracts)
     
# convert tokenized documents into a document-term matrix (bag-of-words)
corpus = [dictionary.doc2bow(text) for text in final_abstracts]
```

### Term Frequency and Inverse Document Frequency (TF-IDF)

```python
#TF IDF
tfidf = models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]
```

## 4. Topic Modeling

### K-Means to Determine Number of Topics
Run KMeans.py to create KMeans function and to determine the number of topics. This section used the method laid out in [8].

```python
exec(open("KMeans.py").read())
```

### Generate LDA Model using gensim
`gensim` is a text mining module. [6] is the official website of gensim and provides several introductory tutorials on using the modules. [7] and [8] were used to create this section. 

```python
# generate LDA model 
NUM_TOPICS = 5
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)
 
# Project to LDA space
 
%time ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word = dictionary, passes=100)
 
ldamodel.print_topics(NUM_TOPICS)
 
docTopicProbMat = ldamodel.get_document_topics(corpus,minimum_probability=0)
 
listDocProb = list(docTopicProbMat)
```

### Put LDA Probabilities into a Matrix and then DataFrame
This step cleans up the output of LDA (topic probabilities for each document) and converts it to a pandas dataframe to make analysis easier.

```python
probMatrix = np.zeros(shape=(num_patents,NUM_TOPICS))
for i,x in enumerate(listDocProb):      #each document i
    for t in x:     #each topic j
        probMatrix[i, t[0]] = t[1] 
         
df = pd.DataFrame(probMatrix)

```

### Generate Word Clouds for each Topic
This step creates word clouds for each of the topics. This section was created referencing [8].

```python
final_topics = ldamodel.show_topics(num_words = 20)
curr_topic = 0
 
for line in final_topics:
    line = line.strip()
    scores = [float(x.split("*")[0]) for x in line.split(" + ")]
    words = [x.split("*")[1] for x in line.split(" + ")]
    freqs = []
    for word, score in zip(words, scores):
        freqs.append((word, score))
    wordcloud = WordCloud(max_font_size=40).generate_from_frequencies(freqs)
     
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")    
    curr_topic += 1
```

### Plot Probability heatmap (Before Topic Clusters)  
This step creates a topic probability heatmap for all documents. It is unordered so the probabilities will be scattered since they are based on the order of the dataset. [3] was referenced to create this part of the code.

```python
plt.pcolor(df.transpose(), norm=None, cmap='Blues')
 
topic_labels = ['Synthetic_Material',
'Data_Information',
'Chemistry',
'Electrical',
'Energy_Turbine'
]
 
plt.yticks(np.arange(df.shape[1])+0.5, topic_labels)
 
plt.colorbar(cmap='Blues')
```

## 5. K-Means Clustering on the Topics

### K-Means to Determine Number of Topics
This section samples a range of potential number of topics (1 to 11). The within sum of squares is calculated and the "elbow" rule is used to deduce that k should be 5. The within sum of squares is exported as a csv file.

```python
k_range = range(1,11)
 
k_means_var = [KMeans(n_clusters=k).fit(df) for k in k_range]
 
centroids = [X.cluster_centers_ for X in k_means_var]
 
k_euclid = [cdist(df, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke,axis=1) for ke in k_euclid]
 
wcss = [sum(d**2) for d in dist]

dfwcss = pd.DataFrame(wcss)
dataset = 'WCSS.csv'
dfwcss.to_csv(dataset, quotechar='\"', quoting=csv.QUOTE_NONNUMERIC,delimiter=',')
 
#tss = sum(pdist(df)**2)/df.shape[0]
#bss = tss - wcss

```

### Run K-Means with 5 Clusters; Create a Total Patent Dataset

```python
k = 5
 
kmeans = KMeans(n_clusters=k).fit(df)
clusters = kmeans.labels_
dfclusters = pd.DataFrame(clusters)
 
# Append Patent Number, Company and Class Name
df.columns = ['Synthetic_Material',
'Data_Information',
'Chemistry',
'Electrical',
'Energy_Turbine']
 
df["PatentNumber"] = exampleData["PatentNumber"]
df["Company"] = exampleData["Company"]
df["ClassName"] = exampleData["ClassName"]
df["PatentTitle"] = exampleData["PatentTitle"]
df["PatentAssignee"] = exampleData["PatentAssignee"]
df["Year"] = exampleData["Year"]
df["Cluster"] = dfclusters
```

### Export Patent Topic Dataset and Rerun Probability Plot ordered by Topic
This step saves the patent dataset (with LDA topic probabilities and K-means cluster). This step also reruns the probability heat map but this time ordered by the topic cluster.

```python
# Save in a new directory
os.chdir("/home/ryanceros/Dropbox/Project - Big Data Analytics/WordCloud/CompetitorLDA")
 
dataset = 'ProbDocUpdated.csv'
df.to_csv(dataset, quotechar='\"', quoting=csv.QUOTE_NONNUMERIC,delimiter=',')
 
with open("topicProb.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(final_topics)
 
newPlot = df.sort(['Cluster'], ascending=[1])
newPlot2=newPlot[topic_labels]
plt.pcolor(newPlot2.transpose(), norm=None, cmap='Blues')
plt.yticks(np.arange(5)+0.5, topic_labels)
plt.colorbar(cmap='Blues')
```

![Topic Probabilities Output](https://cloud.githubusercontent.com/assets/7621432/11481385/33be51d2-976b-11e5-9d0c-08d4d836416e.png)
