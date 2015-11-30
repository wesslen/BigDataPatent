# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:18:43 2015
KMeans to Determine Number of Topics
 
@author: ryan
"""
 
 
from sklearn.cluster import KMeans
 
 
# see http://sujitpal.blogspot.com/2014/08/topic-modeling-with-gensim-over-past.html
# project to 2 dimensions for visualization
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
 
# write out coordinates to file
fcoords = open(os.path.join("coords.csv"), 'wb')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()
 
# Exercise to find number of k
# see http://www.analyticbridge.com/profiles/blogs/identifying-the-number-of-clusters-finally-a-solution
# Source: num_topics.py
 
 
MAX_K = 10
 
X = np.loadtxt(os.path.join("coords.csv"), delimiter="\t")
ks = range(1, MAX_K + 1)
 
inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)
for k in ks:
    kmeans = KMeans(k).fit(X)
    inertias[k - 1] = kmeans.inertia_
    # first difference    
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]
 
elbow = np.argmin(diff3[3:]) + 3
 
plt.plot(ks, inertias, "b*-")
plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()
 
# Find k = 5
NUM_TOPICS = 5
 
X = np.loadtxt(os.path.join("coords.csv"), delimiter="\t")
kmeans = KMeans(NUM_TOPICS).fit(X)
y = kmeans.labels_
 
colors = ["b", "g", "r", "m", "c"]
for i in range(X.shape[0]):
    plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)    
plt.show()
