# Part I: Modules and Working Directory

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
 
# home directory - modify to your own home directory
os.chdir("/home/ryanceros/Dropbox/Project - Big Data Analytics/WordCloud")

# Part II: Load Dataset, Set Column Names and Sample Data

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
 
 
##########################################################
# Run StemLemma to execute Lemmatization and patent_to_words functions
exec(open("StemLemma.py").read())
#########################################################

 
  
 ####################################################
 # Exploratory Choose Sample
 ####################################################
    
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
 

  
 ## Part III: Data Wrangling
 
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
 
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(final_abstracts)
     
# convert tokenized documents into a document-term matrix (bag-of-words)
corpus = [dictionary.doc2bow(text) for text in final_abstracts]
 
#TF IDF
tfidf = models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]
 

###########################
## Cosine Similarity
###########################

index_tfidf = similarities.MatrixSimilarity(tfidf[corpus])
 
#Convert documents to LSI space
#lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=5)
#corp_lsi = lsi[corpus]
#index_lsi = similarities.MatrixSimilarity(lsi[corpus], num_features=12)
 
# Project to LDA space
  
#edge = []            
#for i in range(3484):
#    sims = index_tfidf[corpus_tfidf[i]]
#    sims = sims[0:3484]
#    for j in range(3484):
#        sim = list(enumerate(sims))[j]        
#        if sim[1] > 0.1 and i < j:        
#            edge.append([i,j,sim[1]])
         
#with open("edgeCosineSimilarity.csv", "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(edge)
 
#simMatrix = np.zeros(shape=(num_patents,num_patents))
#for i in range(num_patents):       #each document i
#    sims = index_lsi[corp_lsi[i]]
#    for t in range(num_patents):     #each document j
#        sim = list(enumerate(sims))[t]
#        simMatrix[i,sim[0]] = sim[1]        
        
#simMatrix = pd.DataFrame(np.triu(simMatrix, 1))
 
#simMatrix = simMatrix.transpose()
 
# Plot HeatMap
 
#plt.pcolor(simMatrix, cmap=plt.cm.seismic, vmin=-1, vmax=1)
#plt.colorbar()
#plt.add_patch(
#    patches.Rectangle(
#        (0,num_patents),
#        ))
#plt.yticks(np.arange(0.5, len(simdf.index), 1), simdf.index)
#plt.xticks(np.arange(0.5, len(simdf.columns), 1), simdf.columns)
#plt.show()
 
 
# Part III: Topic Modeling
 
##########################################
# Run KMeans to Determine Number of Topics
exec(open("KMeans.py").read())
##########################################

 
# generate LDA model 
NUM_TOPICS = 5
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)
 
# Project to LDA space
 
%time ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word = dictionary, passes=100)
 
ldamodel.print_topics(NUM_TOPICS)
 
docTopicProbMat = ldamodel.get_document_topics(corpus,minimum_probability=0)
 
listDocProb = list(docTopicProbMat)
 
#CPU times: user 28min 19s, sys: 57.1 ms, total: 28min 19s
#Wall time: 28min 20s
 
# Put LDA Probabilities into a Matrix and then DataFrame
 
probMatrix = np.zeros(shape=(num_patents,NUM_TOPICS))
for i,x in enumerate(listDocProb):      #each document i
    for t in x:     #each topic j
        probMatrix[i, t[0]] = t[1] 
         
df = pd.DataFrame(probMatrix)
 
  
# Topic word clouds
  
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
     
     
# Plot Probability heatmap    
 
plt.pcolor(df.transpose(), norm=None, cmap='Blues')
 
topic_labels = ['Synthetic_Material',
'Data_Information',
'Chemistry',
'Electrical',
'Energy_Turbine'
]
 
plt.yticks(np.arange(df.shape[1])+0.5, topic_labels)
 
plt.colorbar(cmap='Blues')
 
# K-Means the topics
 
k_range = range(1,11)
 
k_means_var = [KMeans(n_clusters=k).fit(df) for k in k_range]
 
centroids = [X.cluster_centers_ for X in k_means_var]
 
k_euclid = [cdist(df, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke,axis=1) for ke in k_euclid]
 
wcss = [sum(d**2) for d in dist]
 
tss = sum(pdist(df)**2)/df.shape[0]
 
bss = tss - wcss
 
# Select 5 Clusters 
 
k = 5
 
kmeans = KMeans(n_clusters=k).fit(df)
 
clusters = kmeans.labels_
 
dfclusters = pd.DataFrame(clusters)
 
# Add in Patent Number, Company and Class Name
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
 
# Export Patent Topic Dataset and Rerun Probability Plot ordered by Topic

# Save in a new directory
os.chdir("/home/ryanceros/Dropbox/Project - Big Data Analytics/WordCloud/CompetitorLDA")
 
dfwcss = pd.DataFrame(wcss)
dataset = 'WCSS.csv'
dfwcss.to_csv(dataset, quotechar='\"', quoting=csv.QUOTE_NONNUMERIC,delimiter=',')
 
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
