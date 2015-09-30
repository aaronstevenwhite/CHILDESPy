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

        self._create_frame_cooccurrences()

    def _get_metadata(self, corpus_name, transcript_name):
        return self.transcript_collection[corpus_name][transcript_name].metadata

    def _create_frame_cooccurrences(self):
        frames = []

        for corpus_name, transcripts in self.transcript_collection.transcripts.iteritems():
            for transcript_name, transcript in transcripts.iteritems():
                metadata = self._get_metadata(corpus_name, transcript_name)
                age, mlu, participants = metadata.extract()
                for parse_index, parse in enumerate(transcript):
                    parse_dataframe = self._create_parse_dataframe(parse.parse)
                    
                    parse_dataframe['ind'] = parse_index
                    parse_dataframe['speaker'] = participants[parse_index]
                    
                    parse_dataframe['age'] = age
                    parse_dataframe['mlu'] = mlu
                    
                    parse_dataframe['corpus'] = corpus_name
                    parse_dataframe['transcript'] = transcript_name

                    frames.append(parse_dataframe)
        
        self.frame_cooccurrences = pandas.concat(frames)

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

            if child_row['ind'] < 0:
                continue

            sub_rows = rows[rows.pind == child_row.ix['ind']]

            if depth - 1:
                node = [child_item, self._extract_frame(sub_rows, depth-1)]
                node = '*'.join(node)
            else:
                node = child_item

            frame.append(node)
        
        frame = '_'.join(frame)

        print frame

        return frame


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
                age, mlu, participants = metadata.extract() ## figure out why this worked with the old CHILDESMetaData.extract
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
