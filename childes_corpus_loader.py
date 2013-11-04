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

    def create_new_instance(self, fid, speaker_regex): ## I don't like how far speaker_regex has to get passed down; should be an attribute?
        transcript_metadata = CHILDESMetaData(self.corpus)

        transcript_metadata._set_metadata(fid, speaker_regex)

        return transcript_metadata

    def _set_metadata(self, fid, speaker_regex):
        self.participants = self._participant_extractor(fid, speaker_regex)
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

    def extract(self):
        return self.age, self.mlu, self.participants


class CHILDESTranscriptCollection(object):

    def __init__(self, corpora, corpus_search_term):
        self._corpora = corpora
        self._corpus_search_term = corpus_search_term

    def __iter__(self):
        return self

    def next(self):
        return self._transcript_iter.next()

    def __getitem__(self, corpus_name):
        return self.transcripts[corpus_name]

    def _get_transcripts(self, speaker_regex, stem, tagged, relation, strip_space, replace, strip_affix):

        transcripts = {}
        corpus_search = self._corpora.search(self._corpus_search_term)

        for corpus in corpus_search:
            metadata_parent = CHILDESMetaData(corpus)
            corpus_name = metadata_parent.corpus_name
            transcripts[corpus_name] = {}
            for fid in corpus.fileids():
                transcript_metadata = metadata_parent.create_new_instance(fid, speaker_regex)
                if tagged:
                    raw_sentence_extractor = lambda part: corpus.tagged_sents(fid, part, stem, relation, strip_space, replace)
                else:
                    raw_sentence_extractor = lambda part: corpus.sents(fid, part, stem, relation, strip_space, replace)

                fid_stripped = os.path.splitext(fid)[0]

                transcripts[corpus_name][fid_stripped] = CHILDESTranscript(raw_sentence_extractor, transcript_metadata, relation, strip_affix)

        return transcripts

    def get_corpus_indices(self):
        return self.transcripts.keys()

    def get_transcript_indices(self, corpus_index):
        return self.transcripts[corpus_index].keys()


class CHILDESStemmedSentences(CHILDESTranscriptCollection):

    def __init__(self, corpora, corpus_search_term, speaker_regex='^(?:(?!CHI).)*$', tagged=True, strip_affix=True):
        CHILDESTranscriptCollection.__init__(self, corpora, corpus_search_term)
        
        self.transcripts = self._get_stemmed_transcripts(speaker_regex=speaker_regex, tagged=tagged, strip_affix=strip_affix)
        self._transcript_iter = (transcript for transcript in self.transcripts)

    def _get_stemmed_transcripts(self, speaker_regex, tagged, strip_affix):
        return self._get_transcripts(speaker_regex=speaker_regex, stem=True, tagged=tagged, relation=None, 
                                     strip_space=True, replace=True, strip_affix=strip_affix)

class CHILDESParsedSentences(CHILDESTranscriptCollection):

    def __init__(self, corpora, corpus_search_term, speaker_regex='^(?:(?!CHI).)*$', strip_affix=False):
        CHILDESTranscriptCollection.__init__(self, corpora, corpus_search_term)

        self.transcripts = self._get_parsed_transcripts(speaker_regex=speaker_regex, strip_affix=strip_affix)
        self._transcript_iter = (transcript for transcript in self.transcripts)

    def _get_parsed_transcripts(self, speaker_regex, strip_affix):
        return self._get_transcripts(speaker_regex=speaker_regex, stem=True, tagged=False, relation=True, 
                                     strip_space=True, replace=True, strip_affix=strip_affix)
        

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

    def __init__(self, parse, strip_affix):
        self.strip_affix = strip_affix
        self._format_parse(parse)

    def _format_parse(self, parse):
        new_parse = []

        bad_index = -1
        affix_index = 2

        for node in parse:
            if re.findall('-[A-Z1-9]*$', node[0]):
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

        self.parse = DataFrame(new_parse, columns=['word', 'pos', 'ind', 'pind', 'relation'])


class CHILDESCooccurrenceCounts(object):
    
    def __init__(self, transcript_collection, sample_type='pos', frame=True, parent_condition=True, depth=1):
        self.transcript_collection = transcript_collection

        self.sample_type = sample_type
        self.frame = frame
        self.parent_condition = parent_condition
        self.depth = depth # doesn't do anything yet

        self._create_freqdist_dict()

    def _create_word_sent_cooccurrence_generator(self, corpus_index, transcript_index):
        transcript = self.transcript_collection[corpus_index][transcript_index]
        for j, sentence in enumerate(transcript):
            for word in sentence:
                yield (word[0].lower(), word[1]), j

    def _create_dependency_cooccurrence_generator(self, corpus_index, transcript_index):
        transcript = self.transcript_collection[corpus_index][transcript_index]
        for sentence in transcript:
            parse = sentence.parse
            for row_index in parse.T:
                row = parse.ix[row_index]
                if self.frame:
                    child_rows = parse[parse.pind == row.ix['ind']]
                    yield row.ix['word'], tuple(child_rows[self.sample_type])
                else:
                    parent_row = parse[parse.ind == row.ix['pind']]
                    if self.parent_condition:
                        yield parent_row.ix['word'], row.ix[self.sample_type]
                    else:
                        yield row.ix[self.sample_type], parent_row.ix['word']


    def _create_freqdist(self, corpus_index, transcript_index):
        if isinstance(self.transcript_collection, CHILDESStemmedSentences):
            cooccur_gen = self._create_word_sent_cooccurrence_generator(corpus_index, transcript_index)
        elif isinstance(self.transcript_collection, CHILDESParsedSentences):
            cooccur_gen = self._create_dependency_cooccurrence_generator(corpus_index, transcript_index)

        return ConditionalFreqDist(cooccur_gen)

    def _create_freqdist_dict(self):
        condfreqdists = {}

        for corpus_index in self.transcript_collection.get_corpus_indices():
            condfreqdists[corpus_index] = {}
            for transcript_index in self.transcript_collection.get_transcript_indices(corpus_index):
                condfreqdists[corpus_index][transcript_index] = self._create_freqdist(corpus_index, transcript_index)

        self.condfreqdists = condfreqdists


class CHILDESFrameCooccurrence(object):

    def __init__(self, transcript_collection, sample_type='pos', depth=1):
        self.transcript_collection = transcript_collection
        self.sample_type = sample_type
        self.depth = depth

        self._create_frame_dataframe()

    def _create_dataframe(self)
        for corpus_name, transcripts in self.transcript_collection.transcripts.iteritems():
            for transcript_name, transcript in transcripts.iteritems():
                metadata = self._get_metadata(corpus_name, transcript_name)
                age, mlu, participants = metadata.extract()
                for parse_index, parse in enumerate(transcript):
                    parse_dataframe = self._create_parse_dataframe(parse)
                    parse_dataframe.speaker = participants[parse_index]
                    
                    parse_dataframe.age = age
                    parse_dataframe.mlu = mlu
                    
                    parse_dataframe.corpus = corpus_name
                    parse_dataframe.transcript = transcript_name
                



    def _create_parse_dataframe(self, parse):
        frame_cooccurrence = []

        for row_index in parse.T:
            row = parse.ix[row_index]

            word = row.ix['word']
            child_rows = parse[parse.pind == row.ix['ind']]

            frame = self._extract_frame(child_rows, self.depth)

            frame_cooccurrence.append([word, frame])

        return DataFrame(frame_cooccurrence, columns=['word', 'frame'])

    def _extract_frame(self, rows, depth):
        frame = []

        for row_index in rows.T:
            child_row = rows.ix[row_index]
            child_item = child_row[self.sample_type]

            sub_rows = rows[rows.pind == child_row.ix['ind']]

            if sub_rows:
                element = (child_item, self._extract_frame(sub_rows, depth-1))
            else:
                element = child_item

            frame.append(element)
        
        return tuple(frame)


class CHILDESCooccurrenceDataFrame(object):

    def __init__(self, cooccurrence_matrix):
        self.cooccurrence_matrix = cooccurrence_matrix

        self._create_dataframe()

    def _get_metadata(self, corpus_name, transcript_name):
        return self.cooccurrence_matrix.transcript_collection[corpus_name][transcript_name].metadata

    def _create_dataframe(self):
        data = []

        for corpus_name, transcripts in self.cooccurrence_matrix.condfreqdists.iteritems():
            for transcript_name, dist in transcripts.iteritems():
                metadata = self._get_metadata(corpus_name, transcript_name)
                age, mlu, participants = metadata.extract()
                for word in dist.conditions():
                    sentence_indices = dist[word].samples()
                    sentence_indices.sort()

                    word, tag = word

                    for i, sent_index in enumerate(sentence_indices):

                        if i > 0:
                            datum = [word, tag, age, mlu, participants[sent_index], corpus_name, transcript_name, sent_index, last_sent_index]
                        else:
                            datum = [word, tag, age, mlu, participants[sent_index], corpus_name, transcript_name, sent_index, -1]

                        data.append(datum)

                        intra_sentence_repeat = dist[word][sent_index] - 1

                        for j in range(intra_sentence_repeat):
                            datum = [word, tag, age, mlu, participants[sent_index], corpus_name, transcript_name, sent_index, sent_index]
                            data.append(datum)

                        last_sent_index = sent_index

        self.dataframe = pandas.DataFrame(data, columns=['word', 'tag', 'age', 'mlu', 'speaker', 'corpus', 'child', 'sent', 'lastsent'])


    def write_data(self, file_path):
        self.dataframe.to_csv(file_path, sep='\t', quoting=1)





def main(corpus_search_term):
    user_data_path = Downloader.default_download_dir(Downloader())
    childes_corpus_path = os.path.join(user_data_path, 'corpora/CHILDES/')

    corpora = CHILDESCorpora(childes_corpus_path)

    # stemmed_sentences = CHILDESStemmedSentences(corpora, corpus_search_term)
    parsed_sentences = CHILDESParsedSentences(corpora, corpus_search_term, strip_affix=True)

    # cooccurrence_counts = CHILDESCooccurrenceCounts(stemmed_sentences)
    cooccurrence_counts = CHILDESCooccurrenceCounts(parsed_sentences)

    # cooccurrence_dataframe = CHILDESCooccurrenceDataFrame(cooccurrence_counts)

    # return corpora, stemmed_sentences, cooccurrence_matrix, cooccurrence_dataframe
    return corpora, parsed_sentences, cooccurrence_counts

if __name__=='__main__':
    # corpora, stemmed_sentences, cooccurrence_counts, cooccurrence_dataframe = main(sys.argv[1])
    corpora, stemmed_sentences, cooccurrence_counts = main(sys.argv[1])
#    cooccurrence_dataframe.write_data('/home/aaronsteven/CHILDESPy/bin/dispersion_counts/'+sys.argv[1]+'.csv')
