library(MASS)
library(plyr)
library(reshape2)
library(ggplot2)


########################################################################################################


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
    corpus$sent <- as.numeric(corpus$sent)
    corpus$lastsent <- as.numeric(corpus$lastsent)

    return(corpus)
}

add.word.freqs <- function(df){
    wordfreq <- count(df, .(word))
    names(wordfreq)[2] <- 'wordfreq'
    
    df <- merge(df, wordfreq)

    return(df)
}

add.sent.lengths <- function(df){
    df$sent <- as.factor(df$sent)
    df$lastsent <- as.factor(df$lastsent)

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

########################################################################################################

## load all corpora individually

gleason <- corpus.loader('gleason')
brown <- corpus.loader('brown')
rollins <- corpus.loader('rollins')
higginson <- corpus.loader('higginson')
newengland <- corpus.loader('newengland')

## merge all corpora

data <- rbind(gleason, brown, rollins, higginson, newengland)

##

data$context <- 'play'
data[data$corpus=='Dinner',]$context <- 'meal'

## create a column that gives the number of utterances between each instance of a word type

data$utterdiff <- data$sent - data$lastsent

## create columns for word frequencies and sentence lengths

data <- add.word.freqs(data)
data <- add.sent.lengths(data)

## create column that gives the number of utterances between each instance of a word type

data$worddiff <- data$cumsentlength - data$cumlastsentlength

## remove data points that do not have an age associated with them

data <- subset(data, !is.na(age))

## remove all tags except for: noun, verb, adjective, preposition, determiner, pronoun, and modal

data.tagsub <- subset(data, tag=='n' | tag=='v' | tag=='adj' | tag=='prep' | tag=='det' | tag=='pro' | tag=='mod')
data.tagsub$tag <- data.tagsub$tag[drop=T,]

## remove all speakers except for: mother and father

data.speakersub <- subset(data.tagsub, speaker == 'MOT' | speaker=='FAT')
data.speakersub$speaker <- data.speakersub$speaker[drop=T,]

## create the data frame we will be working worth

data.cleaned <- data.speakersub

## make noun the new reference level

data.cleaned$tag <- relevel(data.cleaned$tag, 'n')

########################################################################################################

## set the plottng theme to black and white

theme_set(theme_bw())

## plot histograms showing the distribution of sentence lengths

p.sentlength <- ggplot(data.cleaned, aes(x=sentlength, y=..density..)) + geom_bar(binwidth=1, fill="grey", color="black")
p.sentlength.dens <- ggplot(data.cleaned, aes(x=sentlength)) + geom_density(h=5)

## plot histograms showing the distribution of number of utterances between tokens of a word type

p.utterdiff <- ggplot(data.cleaned, aes(x=utterdiff, y=..density..)) + geom_bar(binwidth=1, fill="grey", color="black") + scale_x_continuous(limits=c(0,50))
p.utterdiff.tag <- ggplot(data.cleaned, aes(x=utterdiff, y=..density.., fill=tag)) + geom_bar(binwidth=1, color="black") + scale_x_continuous(limits=c(0,50))
p.utterdiff.tag.dens <- ggplot(data.cleaned, aes(x=utterdiff, linetype=tag)) + geom_density() + scale_x_continuous(limits=c(0,50))

p.utterdiff.tag.facet <- ggplot(data.cleaned, aes(x=utterdiff, y=..density..)) + geom_bar(binwidth=1, color="black") + scale_x_continuous(limits=c(0,50)) + facet_grid(tag~.)
p.utterdiff.tag.dens.facet <- ggplot(data.cleaned, aes(x=utterdiff)) + geom_density() + scale_x_continuous(limits=c(0,50)) + facet_grid(tag~.)

## plot histograms showing the distribution of number of words between tokens of a word type

p.worddiff <- ggplot(data.cleaned, aes(x=worddiff, y=..density..)) + geom_bar(binwidth=1, fill="grey", color="black") + scale_x_continuous(limits=c(0,50))
p.worddiff.tag <- ggplot(data.cleaned, aes(x=worddiff, y=..density.., fill=tag)) + geom_bar(binwidth=1, color="black") + scale_x_continuous(limits=c(0,50))
p.worddiff.tag.dens <- ggplot(data.cleaned, aes(x=worddiff, linetype=tag)) + geom_density() + scale_x_continuous(limits=c(0,50))

p.worddiff.tag.facet <- ggplot(data.cleaned, aes(x=worddiff, y=..density..)) + geom_bar(binwidth=1, color="black") + scale_x_continuous(limits=c(0,50)) + facet_grid(tag~.)
p.worddiff.tag.dens.facet <- ggplot(data.cleaned, aes(x=worddiff)) + geom_density() + scale_x_continuous(limits=c(0,50)) + facet_grid(tag~.)

## plot dot plot showing the correlation between number of utterances and number of words between tokens of a word type

p.utterword <- ggplot(data.cleaned, aes(x=utterdiff, y=worddiff)) + geom_point(alpha=.5) + geom_smooth(method="lm")

########################################################################################################

## get correlation (0.9798) between number of utterances and number of words between tokens of a word type

utterword.cor <- cor(data.cleaned$utterdiff, data.cleaned$worddiff)

## build intercept-only negative binomial model of word distances
## then step-wise constructive-destructive selection procedure using AIC

m.worddiff.interonly <- glm.nb(worddiff ~ 1, data=data.cleaned)
m.worddiff <- step(m.worddiff.interonly, scope=~tag*age*log(wordfreq))

## model with all interactions best: worddiff ~ tag*age*log(wordfreq)

## build intercept-only negative binomial model of word distances
## then step-wise constructive-destructive selection procedure using AIC

m.utterdiff.interonly <- glm.nb(utterdiff ~ 1, data=data.cleaned)
m.utterdiff <- step(m.utterdiff.interonly, scope=~tag*age*log(wordfreq))

## model with all interactions best: worddiff ~ tag*age*log(wordfreq)

########################################################################################################

## add predictions for word and utterance differences

data.cleaned$predword <- predict(m.worddiff)
data.cleaned$predutter <- predict(m.utterdiff)

## order tags by lexical v. functional

data.cleaned$tag <- ordered(data.cleaned$tag, levels=c('n', 'adj', 'v', 'prep', 'mod', 'det', 'pro'))

p.word.pred.freq.tag <- ggplot(data.cleaned, aes(x=log(wordfreq), y=predword, linetype=tag)) + geom_smooth(method="lm")
p.word.pred.age.tag <- ggplot(data.cleaned, aes(x=age, y=predword, linetype=tag)) + geom_smooth(method="lm")

p.utter.pred.freq.tag <- ggplot(data.cleaned, aes(x=log(wordfreq), y=predutter, linetype=tag)) + geom_smooth(method="lm")
p.utter.pred.age.tag <- ggplot(data.cleaned, aes(x=age, y=predutter, linetype=tag)) + geom_smooth(method="lm")

########################################################################################################

## step-wise constructive-destructive selection procedure using AIC of word distances model

m.worddiff.context <- step(m.worddiff, scope=list(lower=~tag*age*log(wordfreq), upper=~tag*age*log(wordfreq)*context))

## model with all interactions best: worddiff ~ tag*age*log(wordfreq)

## step-wise constructive-destructive selection procedure using AIC of utterance distances model

m.utterdiff.context <- step(m.utterdiff, scope=list(lower=~tag*age*log(wordfreq), upper=~tag*age*log(wordfreq)*context))