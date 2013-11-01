import os
import sys
import re
import nltk
import itertools
import numpy as np
import scipy as sp
import pandas

from nltk.downloader import Downloader
from nltk.corpus.reader import CHILDESCorpusReader
from nltk.probability import FreqDist, ConditionalFreqDist

from pandas import DataFrame

class CHILDESCorpora(object):

    def __init__(self, childes_corpus_path):
        corpus_walker = os.walk(childes_corpus_path)

        fileids = {parent : [os.path.join(child) for child in xml_files] for parent, _, xml_files in corpus_walker if xml_files}

        self.corpus_readers = [CHILDESCorpusReader(root, fileids[root]) for root in fileids]

    def search(self, corpus_name):
        matches = [reader for reader in self.corpus_readers if re.findall(corpus_name.lower(), reader.root.lower())]

        return matches


class CHILDESMetaData(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_name = self._extract_corpus_name(corpus)

    def _extract_corpus_name(self, corpus):
        corpus_path = corpus.root.title()
        corpus_name = os.path.basename(corpus_path)
        
        return corpus_name

    def create_new_instance(self, fid):
        transcript_metadata = CHILDESMetaData(self.corpus)

        transcript_metadata._set_metadata(fid)

        return transcript_metadata

    def _set_meta_data(self, fid):
        self.participants = self._participant_extractor(fid)
        self.age = self._age_extractor(fid)
        self.mlu = self._mlu_extractor(fid)


    def _age_extractor(self, fid):
        try:
            age = self.corpus.age(fid)[0]
            return np.int(self.corpus.convert_age(age))
        except TypeError:
            return np.nan

    def _mlu_extractor(self, fid):
        return self.corpus.MLU(fid)[0]


    def _participant_extractor(self, fid, speaker_regex):
        participants = self.corpus.participants(fid)[0]

        return [part for part in participants if re.findall(speaker_regex, part)]


class CHILDESTranscriptCollection(object):

    def __init__(self, corpora, corpus_search_term):
        self._corpora = corpora
        self._corpus_search_term = corpus_search_term

        self._get_transcripts

    def __iter__(self):
        return self

    def next(self):
        return self._transcript_iter.next()

    def __getitem__(self, corpus_name):
        return self.transcripts[corpus_name]

    def _get_transcripts(self, speaker, stem, tagged, relation, strip_space, replace, strip_affix):

        transcripts = {}
        corpus_search = self._corpora.search(self._corpus_search_term)

        for corpus in corpus_search:
            metadata_parent = CHILDESMetaData(corpus)
            transcripts[corpus_name] = {}
            for fid in corpus.fileids():
                raw_sentence_extractor = self._create_raw_sentence_extractor(fid, part, stem, relation, strip_space, replace)

                fid_stripped = os.path.splitext(fid)[0]

                transcript_metadata = metadata_parent.create_new_instance(fid)

                transcripts[corpus_name][fid_stripped] = CHILDESTranscript(raw_sentence_extractor, transcript_metadata, relation, strip_affix)

        self.transcripts = transcripts


    def _create_raw_sentence_extractor(self, fid, part, stem, relation, strip_space, replace): ## the self here is ugly
        if tagged:
            return lambda part: corpus.tagged_sents(fid, part, stem, relation, strip_space, replace)
        else:
            return lambda part: corpus.sents(fid, part, stem, relation, strip_space, replace)


class CHILDESTranscript(object):

    def __init__(self, raw_sentence_extractor, metadata, relation, strip_affix):
        self.metadata = metadata

        self.sentences, self.participants_by_sentence = self._get_sentences(raw_sentence_extractor, relation, strip_affix)
        self._sentence_iter = (sentence for sentence in self.sentences)

    def __iter__(self):
        return self

    def next(self):
        return self._sentence_iter.next()

    def _get_sentences(self, raw_sentence_extractor, relation, strip_affix):
        sentences = []
        participants = []

        for part in self.participants:
            for sent in raw_sentence_extractor(part):
                if sent:
                    if relation:
                        unit = CHILDESDependencyParse(sent)
                    else:
                        unit = CHILDESSentence(sent, strip_affix)
                            
                    sentences.append(unit)
                    participants.append(part)

        return sentences, participants


class CHILDESStemmedSentences(CHILDESTranscriptCollection):

    def __init__(self, corpora, corpus_search_term, speaker='^(?:(?!CHI).)*$', tagged=True, strip_affix=True):
        CHILDESTranscriptCollection.__init__(self, corpora, corpus_search_term)
        
        self.transcripts = self._get_stemmed_transcripts(speaker=speaker, tagged=tagged, strip_affix=strip_affix)
        self._transcript_iter = (transcript for transcript in self.transcripts)

    def __getitem__(self, item):
        return self.transcripts[item]

    def _get_stemmed_transcripts(self, speaker, tagged, strip_affix):
        return self._get_transcripts(speaker=speaker, stem=True, tagged=tagged, relation=None, 
                                     strip_space=True, replace=False, strip_affix=strip_affix)

    def get_corpus_indices(self):
        return self.transcripts.keys()

    def get_transcript_indices(self, corpus_index):
        return self.transcripts[corpus_index].keys()

class CHILDESParsedSentences(CHILDESTranscriptCollection):

    def __init__(self, corpora, corpus_search_term, speaker='^(?:(?!CHI).)*$'):
        CHILDESTranscriptCollection.__init__(self, corpora, corpus_search_term)
        
        self.sentences = self._get_sentences(speaker=speaker, relation=True)
        self._sentence_iter = (sentence for sentence in self.sentences)
                

class CHILDESSentence(object):
    
    def __init__(self, sentence, strip_affix):
        ## Needs to be fixed to allow untagged sentences!
        if strip_affix:
            self.sentence = [(word[0].split('-')[0], word[1]) for word in sentence]
        else:
            self.sentence = nltk.util.flatten([(word[0].split('-'), word[1]) for word in sentence])
        
        self._sent_iter = (word for word in self.sentence)

    def __iter__(self):
        return self

    def next(self):
        return self._sent_iter.next()



class CHILDESDependencyParse(object):

    def __init__(self, parse, age):
        self.age = age 

        self._format_parse(parse)

    def _format_parse(self, parse):
        new_parse = []

        bad_index = -1
        affix_index = 2

        for node in parse:
            if re.findall('-[A-Z1-9]*$', node[0]):
                root, affix = node[0].split('-')

                new_node = [root, node[1]] + node[2].split('|') + []
                new_affix = [affix, 'aff', len(parse)+affix_index, new_node[2], 'AFFIX']

                affix_index += 1

                new_parse.append(new_affix)
            else:
                try:
                    new_node = list(node[0:2]) + node[2].split('|')
                except IndexError:
                    new_node = list(node[0:2]) + [bad_index]*2 + ['BADINDEX']
                    bad_index -= 1

            new_parse.append(new_node)

        self.parse = DataFrame(new_parse, columns=['word', 'pos', 'ind', 'pind', 'relation'])


class CHILDESCooccurrenceMatrix(object):
    
    def __init__(self, transcript_collection, collapse=None):
        self.transcript_collection = transcript_collection
        self.collapse = collapse

        if isinstance(transcript_collection, CHILDESStemmedSentences):
            self._within_sentence_cooccurrence()
        elif isinstance(transcript_collection, CHILDESParsedSentences):
            self._dependency_cooccurrence(condition)

    def _create_cooccurrence_generator(self, corpus_index, transcript_index):
        transcript = self.transcript_collection[corpus_index][transcript_index]
        for j, sentence in enumerate(transcript):
            for word in sentence:
                yield (word[0].lower(), word[1]), j

    def _get_transcript_metadata(self, corpus_index, transcript_index):
        transcript = self.transcript_collection[corpus_index][transcript_index]

        return transcript.age, transcript.mlu, transcript.participants_by_sentence

    def _within_sentence_cooccurrence(self):
        cooccurrence_metadata = {}
        condfreqdists = {}

        cooccurrence_generators = []

        for corpus_index in self.transcript_collection.get_corpus_indices():
            condfreqdists[corpus_index] = {}
            cooccurrence_metadata[corpus_index] = {}
            for transcript_index in self.transcript_collection.get_transcript_indices(corpus_index):
                cooccur_gen = self._create_cooccurrence_generator(corpus_index, transcript_index)
                cooccurrence_generators.append(cooccur_gen)

                if not self.collapse:
                    cooccurrence_metadata[corpus_index][transcript_index] = self._get_transcript_metadata(corpus_index, transcript_index)
                    condfreqdists[corpus_index][transcript_index] = ConditionalFreqDist(cooccur_gen)

        if self.collapse:
            chained_generators = itertools.chain(*cooccurrence_generators)
            condfreqdists = ConditionalFreqDist(chained_generators)
        else:
            self.cooccurrence_metadata = cooccurrence_metadata

        self.condfreqdists = condfreqdists


    def _dependency_cooccurrence(self, condition):
        for parse in sentence_collection:
            pass


def main(corpus_search_term):
    user_data_path = Downloader.default_download_dir(Downloader())
    childes_corpus_path = os.path.join(user_data_path, 'corpora/CHILDES/')

    corpora = CHILDESCorpora(childes_corpus_path)
    stemmed_sentences = CHILDESStemmedSentences(corpora, corpus_search_term)
    cooccurrence_matrix = CHILDESCooccurrenceMatrix(stemmed_sentences)

    return corpora, stemmed_sentences, cooccurrence_matrix

if __name__=='__main__':
    corpora, stemmed_sentences, cooccurrence_matrix = main(sys.argv[1])

    metadata = cooccurrence_matrix.cooccurrence_metadata
    dists = cooccurrence_matrix.condfreqdists

    data = []

    for corpus, transcripts in dists.iteritems():
        for child, dist in transcripts.iteritems():
            age, mlu, speaker = metadata[corpus][child]
            for word in dist.conditions():
                sentence_indices = dist[word].samples()
                sentence_indices.sort()

                word, tag = word

                for i, sent_index in enumerate(sentence_indices):

                    if i > 0:
                        datum = [word, tag, age, mlu, speaker[sent_index], corpus, child, sent_index, last_sent_index]
                    else:
                        datum = [word, tag, age, mlu, speaker[sent_index], corpus, child, sent_index, -1]
                        
                    data.append(datum)

                    intra_sentence_repeat = dist[word][sent_index] - 1

                    for j in range(intra_sentence_repeat):
                        datum = [word, tag, age, mlu, speaker[sent_index], corpus, child, sent_index, sent_index]
                        data.append(datum)
                    
                    last_sent_index = sent_index
    
    data = pandas.DataFrame(data, columns=['word', 'tag', 'age', 'mlu', 'speaker', 'corpus', 'child', 'sent', 'lastsent'])

    data.to_csv('/home/aaronsteven/CHILDESPy/bin/dispersion_counts/'+sys.argv[1]+'.csv', sep='\t', quoting=1)

    # corpus_list = ['bates', 
    #                'bernstein', 
    #                'bloom', 
    #                'bohannon', 
    #                'brent', 
    #                'brown', 
    #                'cartarette', 
    #                'demetras', 
    #                'evans', 
    #                'gleason', 
    #                'higginson']
