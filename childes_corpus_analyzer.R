library(MASS)
library(plyr)
library(reshape2)
library(ggplot2)


########################################################################################################

## reader function

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

## information extractor functions

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

create.transcript.freqs <- function(df){
    transcriptfreq <- count(df, .(corpus, child, word))
    names(transcriptfreq)[4] <- 'frequency'

    return(transcriptfreq)
}

run.dispersion.stats <- function(transcriptfreq){
    ## get by-word sums of frequencies in a given window for each transcript
    trans.freq.sums <- ddply(transcriptfreq, .(word), summarise, sum=sum(frequency))

    ## count how many windows in a transcript a word showed up in: c(k >= 1)
    count.gte.1 <- count(transcriptfreq, .(word))
    names(count.gte.1)[2] <- 'count.gte.1'

    ## count total number of windows in each transcript
    total <- dim(count(transcriptfreq, .(corpus, child)))[1]

    ## put the word and total counts together
    statistics <- merge(count.gte.1, trans.freq.sums)

    statistics$mean <- statistics$sum / total

    ## find each word's inverse document frequency
    statistics$idf <- -(log2(statistics$count.gte.1) - log2(total))

    ## find each word's burstiness
    statistics$burstiness <- (statistics$mean * total) / statistics$count.gte.1    

    return(statistics)

}

create.window.freqs <- function(df, window.size){
    ## create window numbers using window.size modulus
    df$window <- floor(df$sent / window.size)


    ## count number of times a word is seen in a window in a transcript
    windowfreq <- count(df, .(corpus, child, window, word))
    names(windowfreq)[5] <- 'frequency'

    ## add window size column
    windowfreq$window.size <- window.size
    
    return(windowfreq)
}


run.windowed.dispersion.stats <- function(windowfreq){ 
    ## run with windows subsetted
    
    ## get by-word sums of frequencies in a given window for each transcript
    print('sums')
    word.freq.sums <- ddply(windowfreq, .(corpus, child, word), summarise, sum=sum(frequency))

    ## count how many windows in a transcript a word showed up in: c(k >= 1)
    print('gte')
    count.gte.1 <- count(windowfreq, .(corpus, child, word))
    names(count.gte.1)[4] <- 'count.gte.1'

    ## count total number of windows in each transcript
    print('totals')
    totals <- ddply(windowfreq, .(corpus, child), summarize, totals=max(window))

    ## put the word and total counts together
    print('merge')
    statistics <- merge(merge(count.gte.1, word.freq.sums), totals)

    statistics$mean <- statistics$sum / statistics$totals

    ## find each word's inverse document frequency
    print('idf')
    statistics$idf <- log2(statistics$totals) - log2(statistics$count.gte.1)

    ## find each word's burstiness
    print('burstiness')
    statistics$burstiness <- (statistics$mean * statistics$totals) / statistics$count.gte.1    

    return(statistics)
}

## statistical functions (Church, 1995)

adaptation <- function(count.vector){
    count.gte.1 <- length(count.vector)
    count.gte.2 <- subset(count(count.vector >= 2), x==TRUE)$freq

    return(count.gte.2 / count.gte.1)
}

########################################################################################################

## load all corpora individually

gleason <- corpus.loader('gleason')
brown <- corpus.loader('brown')
rollins <- corpus.loader('rollins')
higginson <- corpus.loader('higginson')
newengland <- corpus.loader('newengland')

## load frame corpus

gleason.frame <- read.table('~/CHILDESPy/bin/corpora/gleason_frame.csv', header=T)

## merge all corpora

data <- rbind(gleason, brown, rollins, higginson, newengland)

## add context column

data$context <- 'play'
data[data$corpus=='Dinner',]$context <- 'meal'

## create a column that gives the number of utterances between each instance of a word type

data$utterdiff <- data$sent - data$lastsent

## create columns for word frequencies and sentence lengths

data <- add.word.freqs(data)
data <- add.sent.lengths(data)

## create transcript (document) and window (1-10 utterances) frequency dataframes
## calculate idf, burstiness, and adaptation for transcript and window frequencies

#transcript.frequencies <- create.transcript.freqs(data)

#dispersion.stats <- run.dispersion.stats(transcript.frequencies)

#window.frequencies <- data.frame()
#dispersion.stats.windowed <- data.frame()

#for (i in seq(1,10)){
#    print(i)
#    wf <- create.window.freqs(data, i)
#    window.frequencies <- rbind(window.frequencies, wf)

#    dispersion.stats.wf <- run.windowed.dispersion.stats(wf)
#    dispersion.stats.wf$window.size <- i
#    dispersion.stats.windowed <- rbind(dispersion.stats.windowed, dispersion.stats.wf)    
#}

## create column that gives the number of utterances between each instance of a word type
## the assumption here is that words in the same utterance don't count toward this number

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

## make noun the new reference level (for ease of model interpretation)

data.cleaned$tag <- relevel(data.cleaned$tag, 'n')

########################################################################################################

## set the plottng theme to black and white

theme_set(theme_bw())

## define Sol Lewitt color palette

slPalette <- c("#953536", "#f06b30", "#974f27", "#9dac73", "#9aabb2", "#eab41d", "#d2332d")

## plot histograms showing the distribution of sentence lengths

p.sentlength <- ggplot(data.cleaned, aes(x=sentlength, y=..density..)) + geom_bar(binwidth=1, fill=slPalette[1], color="black")
p.sentlength.dens <- ggplot(data.cleaned, aes(x=sentlength)) + geom_density(h=5)

## order tags by lexical v. functional

data.cleaned$tag.ord <- ordered(data.cleaned$tag, levels=c('n', 'adj', 'v', 'prep', 'mod', 'det', 'pro'))

## plot histograms showing the distribution of number of utterances between tokens of a word type

p.utterdiff <- ggplot(data.cleaned, aes(x=utterdiff, y=..density..)) + geom_bar(binwidth=1, fill="grey", color="black") + scale_x_continuous(limits=c(0,50))
p.utterdiff.tag <- ggplot(data.cleaned, aes(x=utterdiff, y=..density.., fill=tag)) + geom_bar(binwidth=1, color="black") + scale_x_continuous(limits=c(0,50)) + scale_fill_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro'))
p.utterdiff.tag.dens <- ggplot(data.cleaned, aes(x=utterdiff, linetype=tag)) + geom_density() + scale_x_continuous(limits=c(0,50))

p.utterdiff.tag.facet <- ggplot(data.cleaned, aes(x=utterdiff, y=..density..)) + geom_bar(binwidth=1, color="black") + scale_x_continuous(limits=c(0,50)) + facet_grid(tag~.)
p.utterdiff.tag.dens.facet <- ggplot(data.cleaned, aes(x=utterdiff)) + geom_density() + scale_x_continuous(limits=c(0,50)) + facet_grid(tag~.)

## plot histograms showing the distribution of number of words between tokens of a word type

p.worddiff <- ggplot(data.cleaned, aes(x=worddiff, y=..density..)) + geom_bar(binwidth=1, fill="grey", color="black") + scale_x_continuous(limits=c(0,50))
p.worddiff.tag <- ggplot(data.cleaned, aes(x=worddiff, y=..density.., fill=tag)) + geom_bar(binwidth=1, color="black") + scale_x_continuous(limits=c(0,50)) + scale_fill_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro'))
p.worddiff.tag.dens <- ggplot(data.cleaned, aes(x=worddiff, linetype=tag)) + geom_density() + scale_x_continuous(limits=c(0,50))

p.worddiff.tag.facet <- ggplot(data.cleaned, aes(x=worddiff, y=..density..)) + geom_bar(binwidth=1, color="black") + scale_x_continuous(limits=c(0,50)) + facet_grid(tag~.)
p.worddiff.tag.dens.facet <- ggplot(data.cleaned, aes(x=worddiff)) + geom_density() + scale_x_continuous(limits=c(0,50)) + facet_grid(tag~.)

## plot dot plot showing the correlation between number of utterances and number of words between tokens of a word type

p.utterword <- ggplot(data.cleaned, aes(x=utterdiff, y=worddiff)) + geom_point(alpha=.5) + geom_smooth(method="lm", se=F, color=slPalette[7])

########################################################################################################

## get correlation (0.9798) between number of utterances and number of words between tokens of a word type

utterword.cor <- cor(data.cleaned$utterdiff, data.cleaned$worddiff)

## build intercept-only negative binomial model of word distances
## then step-wise constructive-destructive selection procedure using AIC

#m.worddiff.interonly <- glm.nb(worddiff ~ 1, data=data.cleaned)
#m.worddiff <- step(m.worddiff.interonly, scope=~tag*age*log(wordfreq))

m.worddiff <- glm.nb(worddiff ~ tag*age*log(wordfreq), data=data.cleaned)


## model with all interactions best: worddiff ~ tag*age*log(wordfreq)

## build intercept-only negative binomial model of word distances
## then step-wise constructive-destructive selection procedure using AIC

#m.utterdiff.interonly <- glm.nb(utterdiff ~ 1, data=data.cleaned)
#m.utterdiff <- step(m.utterdiff.interonly, scope=~tag*age*log(wordfreq))

m.utterdiff <- glm.nb(utterdiff ~ tag*age*log(wordfreq), data=data.cleaned)

## model with all interactions best: worddiff ~ tag*age*log(wordfreq)

########################################################################################################

## add predictions for word and utterance differences

data.cleaned$predword <- predict(m.worddiff)
data.cleaned$predutter <- predict(m.utterdiff)

p.word.pred.freq.tag <- ggplot(data.cleaned, aes(x=log(wordfreq), y=predword, color=tag.ord)) + geom_smooth(method="lm", se=F) + scale_color_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro'))
p.word.pred.age.tag <- ggplot(data.cleaned, aes(x=age, y=predword, color=tag.ord)) + geom_smooth(method="lm", se=F)+ scale_color_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro'))

p.utter.pred.freq.tag <- ggplot(data.cleaned, aes(x=log(wordfreq), y=predutter, color=tag.ord)) + geom_smooth(method="lm", se=F) + scale_color_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro'))
p.utter.pred.age.tag <- ggplot(data.cleaned, aes(x=age, y=predutter, color=tag.ord)) + geom_smooth(method="lm", se=F) + scale_color_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro'))

########################################################################################################

## step-wise constructive-destructive selection procedure using AIC of word distances model

#m.worddiff.context <- step(m.worddiff, scope=list(lower=~tag*age*log(wordfreq), upper=~tag*age*log(wordfreq)*context))
#m.worddiff.context.gleason <- step(glm.nb(worddiff ~ tag*age*log(wordfreq), data=subset(data.cleaned, corpus=='Dinner' | corpus=='Father' | corpus=='Mother')), scope=list(lower=~tag*age*log(wordfreq), upper=~tag*age*log(wordfreq)*context))

m.worddiff.context <- glm.nb(worddiff ~ tag*age*log(wordfreq) + tag*age*context + tag*log(wordfreq)*context + age*log(wordfreq)*context, data=data.cleaned)
m.worddiff.context.gleason <- glm.nb(worddiff ~ tag*age*log(wordfreq) + tag*age*context + tag*log(wordfreq)*context + age*log(wordfreq)*context, data=subset(data.cleaned, corpus=='Dinner' | corpus=='Father' | corpus=='Mother'))

## model with all interactions best: worddiff ~ tag*age*log(wordfreq)

## step-wise constructive-destructive selection procedure using AIC of utterance distances model

#m.utterdiff.context <- step(m.utterdiff, scope=list(lower=~tag*age*log(wordfreq), upper=~tag*age*log(wordfreq)*context))
#m.utterdiff.context.gleason <- step(glm.nb(utterdiff ~ tag*age*log(wordfreq), data=subset(data.cleaned, corpus=='Dinner' | corpus=='Father' | corpus=='Mother')), scope=list(lower=~tag*age*log(wordfreq), upper=~tag*age*log(wordfreq)*context))

########################################################################################################

## add predictions for word and utterance differences

data.cleaned$predword <- predict(m.worddiff.context)
#data.cleaned$predutter <- predict(m.utterdiff.context)


p.word.pred.freq.tag.context <- ggplot(data.cleaned, aes(x=log(wordfreq), y=predword, color=tag.ord)) + geom_smooth(method="lm", se=F) + facet_grid(context~.) + scale_color_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro'))
p.word.pred.age.tag.context <- ggplot(data.cleaned, aes(x=age, y=predword, color=tag.ord)) + geom_smooth(method="lm", se=F) + scale_color_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro')) + facet_grid(context~.)

#p.utter.pred.freq.tag.context <- ggplot(data.cleaned, aes(x=log(wordfreq), y=predutter, linetype=tag.ord)) + geom_smooth(method="lm", se=F) + facet_grid(context~.)
#p.utter.pred.age.tag.context <- ggplot(data.cleaned, aes(x=age, y=predutter, linetype=tag.ord)) + geom_smooth(method="lm", se=F) + facet_grid(context~.)

data.cleaned$predword <- predict(m.worddiff.context.gleason)
#data.cleaned$predutter <- predict(m.utterdiff.context.gleason)

p.word.pred.freq.tag.context.gleason <- ggplot(data.cleaned, aes(x=log(wordfreq), y=predword, color=tag.ord)) + geom_smooth(method="lm", se=F) + facet_grid(context~.) + scale_color_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro'))
p.word.pred.age.tag.context.gleason <- ggplot(data.cleaned, aes(x=age, y=predword, color=tag.ord)) + geom_smooth(method="lm", se=F) + scale_color_manual(name='Tag', values=slPalette, labels=c('N', 'A', 'V', 'P', 'Mod', 'Det', 'Pro')) + facet_grid(context~.)