library(MASS)
library(plyr)
library(reshape2)
library(ggplot2)

corpus.loader <- function(corpus, datadir='~/CHILDESPy/bin/corpora/'){
    fullpath <- paste(datadir, corpus, '.csv', sep='')
    corpus <- read.table(fullpath, header=T)    

    corpus$word <- as.factor(corpus$word)
    corpus$tag <- as.factor(corpus$tag)
    corpus$age <- as.numeric(corpus$age)
    corpus$mlu <- as.numeric(corpus$mlu)
    corpus$speaker <- as.factor(corpus$speaker)
    corpus$corpus <- as.factor(corpus$corpus)
    corpus$child <- as.factor(corpus$child)
    corpus$sent <- as.factor(corpus$sent)
    corpus$lastsent <- as.factor(corpus$lastsent)

    return(corpus)
}

add.word.freqs <- function(df){
    wordfreq <- count(df, .(word))
    names(wordfreq)[2] <- 'wordfreq'
    
    df <- merge(df, wordfreq)

    return(df)
}

add.sent.lengths <- function(df){
    df <- subset(df, lastsent != -1)

    sentlengths <- count(df, .(corpus, child, sent))
    names(sentlengths)[4] <- 'sentlength'
    
    sentlengths <- ddply(sentlengths, .(corpus, child), transform, cumsentlength=cumsum(sentlength))
    
    df <- merge(df, sentlengths)

    sentlengths <- sentlengths[,c('corpus', 'child', 'sent', 'cumsentlength')]
    names(sentlengths)[c(3,4)] <- c('lastsent', 'cumlastsentlength')

    df <- merge(df, sentlengths)

    df$sent <- as.numeric(df$sent)
    df$lastsent <- as.numeric(df$lastsent)

    return(df)
}

gleason <- corpus.loader('gleason')
brown <- corpus.loader('brown')
rollins <- corpus.loader('rollins')
higginson <- corpus.loader('higginson')
newengland <- corpus.loader('newengland')

data <- rbind(gleason, brown, higginson, newengland)
data <- add.word.freqs(data)
data <- add.sent.lengths(data)

data.tagsub <- subset(data, tag=='n' | tag=='v' | tag=='adj' | tag=='prep' | tag=='det' | tag=='pro' | tag=='mod')
data.tagsub$tag <- data.tagsub$tag[drop=T,]

data.speakersub <- subset(data.tagsub, speaker == 'MOT' | speaker=='FAT')
data.speakersub$speaker <- data.speakersub$speaker[drop=T,]