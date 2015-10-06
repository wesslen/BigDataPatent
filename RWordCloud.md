# Code for Initial WordCloud

## Working Directory & Packages
Set your working directory, install packages (if necessary, uncomment) and import the libraries.

```r
# change this to your working directory
setwd("~/Dropbox/Project - Big Data Analytics/Hadoop")

#if first time, download these libraries:
#install.packages('wordcloud')
#install.packages('tm')
#install.packages('SnowballC')

#import these librarys
library(tm)         #text mining package
library(wordcloud)  #wordcloud package
#library(SnowballC) #Used for Stemming, which is not being used yet so do not need
```

## Load Data

```r
#read in csv, strip white space, no header
data <- read.csv("3M onlyRyan.csv", header=FALSE, strip.white = TRUE, stringsAsFactors=FALSE)

#load column names
names(data) <- c("PatentNumber","CompanyName","PatentAssignee","YearGranted","YearApplied",
                 "PatentClass","PatentTitle","PatentAbstract")

#details about data's structure
str(data)
```

## Format Abstracts and Titles

```r
#Format Abstract
Abstracts <- paste(data$PatentAbstract, collapse=" ")
review_source <- VectorSource(Abstracts)
corpus <- Corpus(review_source)

#Format Titles
Titles <- paste(data$PatentTitle, collapse=" ")
review_source <- VectorSource(Titles)
corpusTitle <- Corpus(review_source)
```
## Clean Abstracts and Titles
Originally, I found a problem with the code. I needed the following site to correct the issue.
[Reference website: "Fun Error After Running to Lower While making Twitter WordCloud"](http://stackoverflow.com/questions/27756693/fun-error-after-running-tolower-while-making-twitter-wordcloud)

Background on stemming: [Reference website](http://l.rud.is/YiKB9G)

```r
#lower case
corpus <- tm_map(corpus, content_transformer(tolower), mc.cores=1)
corpusTitle <- tm_map(corpusTitle, content_transformer(tolower), mc.cores=1)

#got errors when ran: corpus <- tm_map(corpus, content_transformer(stri_trans_tolower))
#see reference document, bottom comments

#remove punctuation
corpus <- tm_map(corpus, content_transformer(removePunctuation))
corpusTitle <- tm_map(corpusTitle, content_transformer(removePunctuation))

#strip white space
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
corpusTitle <- tm_map(corpusTitle, content_transformer(stripWhitespace))

### Stemming -- this is commented out because it took too long; however, we'll need to run this evenutally
## save original corpus
## c_orig = corpus
### do the actual stemming
## corpus = tm_map(corpus, stemDocument)
## corpus = tm_map(corpus, stemCompletion, dictionary=c_orig)

#remove stop words; added first, second, one as additional common words that don't seem to add value
stopwords <- c(stopwords("english"),"first","second","one","two")
corpus.nostopwords <- tm_map(corpus, content_transformer(removeWords), stopwords)
corpusTitle.nostopwords <- tm_map(corpusTitle, content_transformer(removeWords), stopwords)
```

## Create Document Term Matrix

```r
#Abstract
dtm <- DocumentTermMatrix(corpus.nostopwords)
dtm2 <- as.matrix(dtm)
frequency <- colSums(dtm2)
frequency <- sort(frequency, decreasing=TRUE)

# look at the top 20 words
head(frequency,20)
   includes       light     surface       least  comprising       layer     optical      method 
        529         335         333         320         277         276         230         229 
   material   substrate     methods      device    provided   disclosed        film      system 
        208         201         198         194         189         188         183         164 
        can composition         may   plurality 
        159         159         154         153 

#Title
dtmTitle <- DocumentTermMatrix(corpusTitle.nostopwords)
dtmTitle2 <- as.matrix(dtmTitle)
frequencyTitle <- colSums(dtmTitle2)
frequencyTitle <- sort(frequencyTitle, decreasing=TRUE)

# look at the top 20 words
head(frequencyTitle,20)
      method      methods       making      optical       system       device compositions        using 
         234          163          131           92           89           87           72           70 
     article     articles     adhesive         film        light  composition    apparatus     assembly 
          69           65           61           59           59           53           52           50 
     devices        films      surface      display 
          49           49           40           37 

```

## Word Cloud

For this part, the following website was used. [Text Mining and Word Cloud Fundamentals](http://www.sthda.com/english/wiki/text-mining-and-word-cloud-fundamentals-in-r-5-simple-steps-you-should-know)

```r
words <- names(frequency)
set.seed(12345)
wordcloud(words = words, freq = frequency, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
# warning are ok -- they are words that couldn't fit on the plot

wordsTitle <- names(frequencyTitle)
set.seed(12345)
wordcloud(words = wordsTitle, freq = frequencyTitle, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
```

## Frequency Tables

```r
#Plot top 50 words (you can change by changing the "top" parameter)

#Abstract 
top <- 50

barplot(frequency[1:top], las = 2, names.arg = words[1:top],
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")

#Title
top <- 50

barplot(frequencyTitle[1:top], las = 2, names.arg = wordsTitle[1:top],
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")
```
