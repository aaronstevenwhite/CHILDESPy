library(MASS)
library(plyr)
library(reshape)
library(ggplot2)

corpus.loader <- function(corpus, datadir='~/CHILDESPy/bin/corpora/'){
    fullpath <- paste(datadir, corpus, '.csv', sep='')
    corpus <- read.table(fullpath, header=T)    

    return(corpus)
}

gleason <- corpus.loader('gleason')
brown <- corpus.loader('brown')
rollins <- corpus.loader('rollins')
higginson <- corpus.loader('higginson')
newengland <- corpus.loader('newengland')

data <- rbind(gleason, brown, rollins, higginson, newengland)

add.word.freqs <- function(df){
    wordfreq <- as.data.frame(table(df$word))
    names(wordfreq)[2] <- 'wordfreq'
    
    df <- merge(df, wordfreq)

    return(df)
}

data <- add.word.freqs(data)

add.sent.lengths <- function(df){
    sentlengths <- as.data.frame(xtabs(~corpus+child+sent))
    names(sentlengths)[2] <- 'sentlengths'
    
    df <- merge(df, sentlengths)

    return(df)
}

data <- add.sent.lengths(data)