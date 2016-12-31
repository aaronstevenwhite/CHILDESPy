from .sentence import CHILDESSentence

class CHILDESCorpus(object):
    '''
    Container for CHILDES sentences.
    
    This is a thin wrapper around CHILDESCorpusReader. Its main purpose is to
    contain a dictionary of lists that map transcripts' relative paths to lists
    of CHILDESSentences.
    '''
    
    
    def __init__(self, corpus_root):
        self.corpusreader = CHILDESCorpusReader(root=corpus_root, fileids='.*.xml')
        
        self._process_corpus()
        self._initialize_transcript_iterator()
    
    def __getitem__(self, key):
        return self.transcripts[key]
    
    def __iter__(self):
        return self
    
    def next(self):
        try:
            self._transcript_iter.next()
        except StopIteration as e:
            self._initialize_transcript_iterator()
            raise e
  
    def _process_corpus(self):
        transcripts = {}

        participants = self.corpusreader.participants()

        for parts, fid in zip(participants, self.corpusreader.fileids()):
            sents_raw = zip(self.corpusreader.sents(fileids=fid, speaker='MOT'), 
                            self.corpusreader.sents(fileids=fid, speaker='MOT', relation=True))

            transcripts[fid] = [CHILDESSentence(raw, parsed, fid, parts['CHI'], parts['MOT']) 
                                for raw, parsed in sents_raw]
            
        self.transcripts = transcripts
    
    def _initialize_transcript_iterator(self):
        self._transcript_iter = ((fid,s) 
                                 for fid, sents in self.transcripts.iteritems() 
                                 for s in sents)
