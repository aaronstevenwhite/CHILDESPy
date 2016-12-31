import pandas as pd

from nltk.corpus.reader import CHILDESCorpusReader

class CHILDESSentence(object):
    '''
    Represents a sentence in CHILDES.
    
    This is a thin wrapper around a pandas.DataFrame containing MEGRASP dependency parses
    from CHILDES. It also contains metadata for the child and the speaker.
    '''

    def __init__(self, sent_unanalyzed, sent_parsed, fileid, child_metadata, speaker_metadata):
        ## set sentence representations
        self.sent_unanalyzed = sent_unanalyzed
        self.sent_parsed = sent_parsed
        
        ## process sentence representations
        self._filter_sentence()
        self._separate_contractions()
        self._separate_morphological_analyses()
        
        ## set metadata
        self.fileid = fileid
        self.child_metadata = child_metadata
        self.speaker_metadata = speaker_metadata
        
        ## process metadata
        self._process_metadata()
    
    def __repr__(self):
        return self.sentence.__repr__()
    
    def __str__(self):
        return self.__repr__()
    
    def _filter_sentence(self):
        '''
        filter words that do not have a POS tag
        (usually, "xxx" but sometimes things like "hmm")
        '''
        
        good_word = [bool(word[1]) for word in self.sent_parsed]
        
        self.sent_unanalyzed = [word 
                                for i, word in enumerate(self.sent_unanalyzed) if good_word[i]]
        self.sent_parsed = [list(word[:2]) + word[2].split('|') 
                            for i, word in enumerate(self.sent_parsed) if good_word[i]]        
    
    def _separate_contractions(self):
        '''
        Separate constituents of a contraction onto separate lines
        of a MEGRASP dependency parse
        '''
        
        parse_list_processed = []
        
        sent_pairs = zip(self.sent_unanalyzed, self.sent_parsed)
        
        for word, (wordanalyzed, pos, ind, parind, rel) in sent_pairs:
            wa_pos = zip(wordanalyzed.split('~'), pos.split('~'))
            curr_ind = ind = int(ind)
            parind = int(parind)
    
            for wa, p in wa_pos:
                if curr_ind == ind:
                    line = [word, wa, p, curr_ind, parind, rel]
                elif curr_ind == int(parind):
                    line = ['', wa, p, curr_ind, 0, 'ROOT']
                else:
                    line = ['', wa, p, curr_ind, parind, rel]
        
                parse_list_processed.append(line)
        
                curr_ind += 1
        
        self.sentence = pd.DataFrame(parse_list_processed, 
                                     columns=['word', 'wordanalyzed', 'pos', 
                                              'ind', 'parind', 'relation'])
    
    def _separate_morphological_analyses(self):
        '''separate root and morphological analysis into distinct columns'''
        
        def get_root(wordanalyzed):
            return wordanalyzed.split('-')[0]
        
        def get_affix(wordanalyzed):
            wordanalyzed_list = wordanalyzed.split('-')

            if len(wordanalyzed_list) == 1:
                return 'NONE'
            else:
                return '-'.join(wordanalyzed_list[1:])

        self.sentence['wordroot'] = self.sentence.wordanalyzed.map(get_root)
        self.sentence['affix'] = self.sentence.wordanalyzed.map(get_affix)

        self.sentence = self.sentence[['word', 'wordanalyzed', 'wordroot', 'affix', 
                                       'pos', 'ind', 'parind', 'relation']]

    def _process_metadata(self):
        '''
        copy the child and speaker metadata dicts to the CHILDESSentence object's 
        internal dictionary, prepending 'child_' and 'speaker_' accordingly
        
        This exposes the child and speaker metadata as object attributes.
        '''
        
        for k, v in self.child_metadata.items():
            if k != 'age':
                self.__dict__['child_'+k] = v
            else:
                self.__dict__['child_'+k] = CHILDESCorpusReader('','').convert_age(v)
        
        for k, v in self.speaker_metadata.items():
            self.__dict__['speaker_'+k] = v
        
    def create_tikz_dependency(self, wordcol='word'):
        '''Create a tikz dependency representation from the dependency parse'''
        
        def master_temp(sent, deps): 
            sent_wrapped = '\t\\begin{deptext}\n\t\t'+sent+'\n\t\end{deptext}'
            return '\\begin{dependency}\n'+sent_wrapped+'\n'+deps+'\n\end{dependency}'

        sent = ' \& '.join(['\\textit{'+p+'}' for p in self.sentence.pos]) + ' \\\\\n'+\
               ' \& '.join(self.sentence[wordcol]) + ' \\\\'
        deps = []

        for l in np.array(self.sentence[['ind', 'parind', 'relation']]):
            if l[2] == 'ROOT':
                deps.append('\t\deproot{'+str(l[0])+'}{'+l[2]+'}')
            else:
                deps.append('\t\depedge{'+str(l[1])+'}{'+str(l[0])+'}{'+l[2]+'}')

        deps = '\n'.join(deps)

        return master_temp(sent, deps)
