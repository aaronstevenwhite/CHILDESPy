import os
import sys
import re
import nltk
import itertools
import numpy as np
import scipy as sp
import pandas

from collections import defaultdict
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

    def create_new_instance(self, fid, speaker_regex): ## I don't like how far speaker_regex has to get passed down; should be an attribute?
        transcript_metadata = CHILDESMetaData(self.corpus)

        transcript_metadata._set_metadata(fid, speaker_regex)

        return transcript_metadata

    def _set_metadata(self, fid, speaker_regex):
        self.participants = self._participant_extractor(fid, speaker_regex)

        self.fid = fid.split('.')[0]
        self.age = self._age_extractor(fid)
        self.mlu = self._mlu_extractor(fid)


    def _age_extractor(self, fid):
        try:
            age = self.corpus.age(fid)[0]
            return np.int(self.corpus.convert_age(age))
        except (TypeError, AttributeError):
            return np.nan

    def _mlu_extractor(self, fid):
        return self.corpus.MLU(fid)[0]

    def _participant_extractor(self, fid, speaker_regex):
        participants = self.corpus.participants(fid)[0]

        return [part for part in participants if re.findall(speaker_regex, part)]

    def extract(self):
        return self.age, self.mlu, self.participants_by_sentence # this only gets added later; so ugly


class CHILDESTranscriptCollection(object):

    def __init__(self, corpora, corpus_search_term, 
                 speaker_regex, stem, tagged, relation, strip_space, replace, strip_affix):
        self._corpora = corpora
        self._corpus_search_term = corpus_search_term

        self._transcript_iter = self._create_transcripts(speaker_regex=speaker_regex, 
                                                         stem=stem, 
                                                         tagged=tagged, 
                                                         relation=relation, 
                                                         strip_space=strip_space, 
                                                         replace=replace, 
                                                         strip_affix=strip_affix)

    def __iter__(self):
        return self

    def next(self):
        return self._transcript_iter.next()


    def _create_transcripts(self, speaker_regex, stem, tagged, relation, strip_space, replace, strip_affix):

        self.transcripts = defaultdict(dict)

        corpus_search = self._corpora.search(self._corpus_search_term)

        for corpus in corpus_search:
            metadata_parent = CHILDESMetaData(corpus)
            corpus_name = metadata_parent.corpus_name

            for fid in corpus.fileids():
                transcript_metadata = metadata_parent.create_new_instance(fid, speaker_regex)

                if tagged:
                    raw_sentence_extractor = lambda part: corpus.tagged_sents(fid, part, stem, relation, strip_space, replace)
                else:
                    raw_sentence_extractor = lambda part: corpus.sents(fid, part, stem, relation, strip_space, replace)

                fid_stripped = os.path.splitext(fid)[0]

                transcript = CHILDESTranscript(raw_sentence_extractor, 
                                               transcript_metadata, 
                                               relation, 
                                               strip_affix)

                self.transcripts[corpus_name][fid_stripped] = transcript

                yield transcript


    def get_corpus_indices(self):
        return self.transcripts_indices.keys()

    def get_transcript_indices(self, corpus_index):
        return self.transcripts_indices[corpus_index].keys()


class CHILDESSentences(CHILDESTranscriptCollection):

    def __init__(self, corpora, corpus_search_term, speaker_regex='^(?:(?!CHI).)*$', tagged=True, strip_affix=True):
        CHILDESTranscriptCollection.__init__(self, corpora, corpus_search_term,
                                             speaker_regex=speaker_regex,
                                             stem=False,
                                             tagged=tagged,
                                             relation=None,
                                             strip_space=True,
                                             replace=True,
                                             strip_affix=strip_affix)


class CHILDESStemmedSentences(CHILDESTranscriptCollection):

    def __init__(self, corpora, corpus_search_term, speaker_regex='^(?:(?!CHI).)*$', tagged=True, strip_affix=True):
        CHILDESTranscriptCollection.__init__(self, corpora, corpus_search_term,
                                             speaker_regex=speaker_regex,
                                             stem=True,
                                             tagged=tagged,
                                             relation=None,
                                             strip_space=True,
                                             replace=True,
                                             strip_affix=strip_affix)


class CHILDESParsedSentences(CHILDESTranscriptCollection):

    def __init__(self, corpora, corpus_search_term, speaker_regex='^(?:(?!CHI).)*$', strip_affix=False):
        CHILDESTranscriptCollection.__init__(self, corpora, corpus_search_term,
                                             speaker_regex=speaker_regex,
                                             stem=True,
                                             tagged=False,
                                             relation=True,
                                             strip_space=True,
                                             replace=True,
                                             strip_affix=strip_affix)
        

class CHILDESTranscript(object):

    def __init__(self, raw_sentence_extractor, metadata, relation, strip_affix):
        self.metadata = metadata

        self.sentences = self._get_sentences(raw_sentence_extractor, relation, strip_affix)
        self._sentence_iter = (sentence for sentence in self.sentences)

    def __iter__(self):
        return self

    def next(self):
        return self._sentence_iter.next()

    def _get_sentences(self, raw_sentence_extractor, relation, strip_affix):
        sentences = []
        participants = []

        for part in self.metadata.participants:
            for sent in raw_sentence_extractor(part):
                if sent:
                    if relation:
                        unit = CHILDESDependencyParse(sent, strip_affix)
                    else:
                        unit = CHILDESSentence(sent, strip_affix)
                            
                    sentences.append(unit)
                    participants.append(part)

        self.metadata.participants_by_sentence = participants ## this is ugly

        return sentences
                

class CHILDESSentence(object):
    
    def __init__(self, sentence, strip_affix):
        self.strip_affix = strip_affix

        self.sentence = [self._process_word(word) for word in sentence]

        self._sent_iter = (word for word in self.sentence)

    def __repr__(self):
        return self.sentence.__repr__()

    def __iter__(self):
        return self

    def next(self):
        return self._sent_iter.next()

    def _process_word(self, word):
        if isinstance(word, tuple):
            stemmed = word[0].split('-')

            if self.strip_affix:
                return (stemmed[0], word[1]) 
            else:
                return word
        else:
            stemmed = word.split('-')

            if self.strip_affix:
                return stemmed[0]
            else:
                return word[0]


class CHILDESDependencyParse(object):

    def __init__(self, parse, strip_affix):
        self.strip_affix = strip_affix
        self.sentence = self._format_parse(parse)

    def __repr__(self):
        return self.sentence.__repr__()

    def _format_parse(self, parse):
        new_parse = []

        bad_index = -1
        affix_index = 2

        for node in parse:
            if re.findall('-[a-zA-Z1-9]*$', node[0]):
                root, affix = node[0].split('-')

                new_node = [root, node[1]] + node[2].split('|') + []

                if not self.strip_affix:
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

        return np.array(new_parse)

def main():
    user_data_path = Downloader.default_download_dir(Downloader())
    childes_corpus_path = os.path.join(user_data_path, 'corpora/CHILDES/')

    corpora = CHILDESCorpora(childes_corpus_path)

    return corpora

if __name__=='__main__':
    corpora = main()

    corpus_search_term = sys.argv[1]

    unstemmed_sentences = CHILDESSentences(corpora, corpus_search_term)
    stemmed_sentences = CHILDESStemmedSentences(corpora, corpus_search_term)
    parsed_sentences = CHILDESParsedSentences(corpora, corpus_search_term)
