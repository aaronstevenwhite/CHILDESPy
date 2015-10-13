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

class CHILDESCollection(object):
    '''a collection of CHILDES corpora'''
    
    def __init__(self, corpus_root):
        
        self.corpus_root = corpus_root
        self.corpus_readers = [CHILDESCorpusReader(root, fids) for root, fids in self._walk_corpus()]

    def _walk_corpus(self):
        '''walk down corpus file hierarchy, collecting children'''
        
        collection_walker = os.walk(self.corpus_root)
        get_children = lambda xml_files: [os.path.join(child) for child in xml_files]

        collection_dict = {parent : get_children(d) for parent, _, d in collection_walker if d}

        return collection_dict.iteritems()
 
    def search(self, corpus_name):
        '''get all the corpora matching the corpus_name regex'''
        
        matches = lambda x: re.findall(corpus_name.lower(), x)

        return [reader for reader in self.corpus_readers if matches(reader.root.lower())]


class CHILDESMetaData(object):
    '''the metadata for a particular corpus'''
    
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


class CHILDESCorpus(object):

    def __init__(self, corpora, corpus_search_term, speaker_regex, stem,
                 tagged, relation, strip_space, replace, strip_affix):
        
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


    def _create_transcripts(self, speaker_regex, stem, tagged, relation,
                            strip_space, replace, strip_affix):

        self.transcripts = defaultdict(dict)

        corpus_search = self._corpora.search(self._corpus_search_term)

        for corpus in corpus_search:
            metadata_parent = CHILDESMetaData(corpus)
            corpus_name = metadata_parent.corpus_name

            for fid in corpus.fileids():
                transcript_metadata = metadata_parent.create_new_instance(fid, speaker_regex)

                if tagged:
                    raw_sentence_extractor = lambda part: corpus.tagged_sents(fid, part, stem, relation,
                                                                              strip_space, replace)
                else:
                    raw_sentence_extractor = lambda part: corpus.sents(fid, part, stem, relation,
                                                                       strip_space, replace)

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


class CHILDESSentences(CHILDESCorpus):

    def __init__(self, corpora, corpus_search_term, speaker_regex='^(?:(?!CHI).)*$', tagged=True, strip_affix=True):
        CHILDESCorpus.__init__(self, corpora, corpus_search_term,
                                             speaker_regex=speaker_regex,
                                             stem=False,
                                             tagged=tagged,
                                             relation=None,
                                             strip_space=True,
                                             replace=True,
                                             strip_affix=strip_affix)


class CHILDESStemmedSentences(CHILDESCorpus):

    def __init__(self, corpora, corpus_search_term, speaker_regex='^(?:(?!CHI).)*$',
                 tagged=True, strip_affix=True):
        
        CHILDESCorpus.__init__(self, corpora, corpus_search_term,
                                             speaker_regex=speaker_regex,
                                             stem=True,
                                             tagged=tagged,
                                             relation=None,
                                             strip_space=True,
                                             replace=True,
                                             strip_affix=strip_affix)


class CHILDESParsedSentences(CHILDESCorpus):

    def __init__(self, corpora, corpus_search_term, speaker_regex='^(?:(?!CHI).)*$',
                 strip_affix=False):
        
        CHILDESCorpus.__init__(self, corpora, corpus_search_term,
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

    def __getitem__(self, index):
        return self.sentences[index]
        
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
        new_parse = [['ROOT', 'ROOT', 0, 0, 'NONE']]

        self._parse_length = len(parse)
        
        self._bad_index = -1
        self._affix_index = 1

        for node in parse:
            node, func_morph = self._separate_functional_morpheme(node)
            root, affix = self._separate_affix_morpheme(node)

            new_parse.append(root)

            if affix is not None:
                new_parse.append(affix)

            if func_morph is not None:
                new_parse.append(func_morph)

        parse_df = pandas.DataFrame(new_parse,
                                    columns=['lemma', 'pos', 'ind',
                                             'pind', 'gramrel'])

        parse_df.ind = parse_df.ind.astype(int)
        parse_df.pind = parse_df.pind.astype(int)

        return parse_df
        
    def _update_affix_index(self):
        self._affix_index += 1
        
        return self._parse_length+self._affix_index
    
    def _separate_functional_morpheme(self, node):
        
        if re.findall('~[a-zA-Z1-9]*$', node[0]):
            lemma1, lemma2 = node[0].split('~')
            tag1, tag2 = node[1].split('~')

            node = [lemma1, tag1, node[2]]

            functional = [lemma2, tag2, self._update_affix_index(),
                          node[2].split('|')[0], 'FUNCTIONAL']

            self._affix_index += 1

            return node, functional   

        else: return node, None

    def _separate_affix_morpheme(self, node):

        affix = None
        
        if re.findall('-[a-zA-Z1-9]*$', node[0]):

            lemma_root, affix = node[0].split('-')

            root = [lemma_root, node[1]] + node[2].split('|')

            if not self.strip_affix:

                affix = [affix, 'aff', self._update_affix_index(),
                         root[2], 'AFFIX']

                self._affix_index += 1

        else:
            
            try:
                root = list(node[0:2]) + node[2].split('|')
            except IndexError:
                root = list(node[0:2]) + [self._bad_index]*2 + ['BADINDEX']
                self._bad_index -= 1

        return root, affix


    def has_lemma(self, lemma):
        '''check for whether sentence has particular lemma in it'''
        
        return (self.sentence.lemma==lemma).any()

    def has_pos(self, pos):
        '''check for whether sentence has a word with a particular part-of-speech'''
        
        return (self.sentence.pos==pos).any()
            
    def has_grammmatical_relation(self, gramrel):
        '''check for whether sentence has particular grammatical relation in it'''
        
        return (self.sentence.gramrel==gramrel).any()

    def _selector(self, lemma, pos, gramrel):
        '''select a node in the parse based on the lemma, pos, or gramrel in that line'''
        
        bool_index = self.sentence.lemma==self.sentence.lemma

        if lemma is not None:
            bool_index = np.logical_and(bool_index, self.sentence.lemma==lemma)

        if pos is not None:
            bool_index = np.logical_and(bool_index, self.sentence.pos==pos)

        if gramrel is not None:
            bool_index = np.logical_and(bool_index, self.sentence.gramrel==gramrel)

        return bool_index
    
    def get_parent(self, lemma=None, pos=None, gramrel=None):
        '''
        get the (unique) parent of a node in the parse
        based on that node's lemma, pos, or gramrel
        '''
        
        selected = self.sentence[self._selector(lemma, pos, gramrel)]
        parent_indices = selected.pind
        
        return [(selected.iloc[i], self.sentence[self.sentence.ind == pind]) for i, pind in enumerate(parent_indices)]

    def get_child(self, lemma=None, pos=None, gramrel=None):
        '''
        get the (nonunique) children of a node in the parse
        based on that node's lemma, pos, or gramrel
        '''
        
        selected = self.sentence[self._selector(lemma, pos, gramrel)]
        indices = selected.ind

        return [(selected.iloc[i], self.sentence[self.sentence.pind == ind]) for i, ind in enumerate(indices)]

def main():
    user_data_path = Downloader.default_download_dir(Downloader())
    childes_corpus_path = os.path.join(user_data_path, 'corpora/CHILDES/')

    corpora = CHILDESCollection(childes_corpus_path)

    return corpora

if __name__=='__main__':
    import argparse

    ## initialize parser
    parser = argparse.ArgumentParser(description='Load CHILDES')

    ## file handling
    parser.add_argument('--corpus', 
                        type=str, 
                        default='.*')

    ## parse arguments
    args = parser.parse_args()

    ## load corpora
    corpora = main()

    ## construct APIs to different annotations of those corpora
    unstemmed_sentences = CHILDESSentences(corpora, args.corpus)
    stemmed_sentences = CHILDESStemmedSentences(corpora, args.corpus)
    parsed_sentences = CHILDESParsedSentences(corpora, args.corpus)
