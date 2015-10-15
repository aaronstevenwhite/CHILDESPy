import argparse
import pandas

from childespy.corpusloader import main, CHILDESParsedSentences

## initialize parser
parser = argparse.ArgumentParser(description='Load CHILDES')

## file handling
parser.add_argument('--corpus', 
                    type=str, 
                    default='.*')
parser.add_argument('--verblist', 
                    type=str, 
                    default='attitude_verbs.csv')

## parse arguments
args = parser.parse_args()

## load corpora
corpora = main()

## load verbs
verbs = [v.strip() for v in open(args.verblist)]

## represent the parsed sentences
parsed_sentences = CHILDESParsedSentences(corpora, args.corpus)

## define clausal feature extractor
class ClausalFeatureExtractor(object):

    def __init__(self, parsed_sentence, lemmas_of_interest):
        self.parse = parsed_sentence
        self._extract_features(lemmas_of_interest)
        
    def _extract_features(self, lemmas_of_interest):
        self._features = []

        lemmas = self.parse.has_which_lemmas(lemmas_of_interest)
        
        for lemma in lemmas:
            parent_children = self.parse.get_children(lemma=[lemma],
                                                      pos=['v'],
                                                      gramrel=['ROOT'])
            for _, children in parent_children:
                self._features.append([lemma]+['NONE']*8)


    def _matrix_subject_detector(self, children):
        subj_lemma = children[children.gramrel=='SUBJ'].lemma

        if subj_lemma:
            self._features[-1][1] = 'EXP' if subj_lemma[0] in ['it', 'there'] else 'LEX' 

    def _matrix_object_detector(self, children):
        obj_lemmas = children[children.gramrel=='OBJ'].lemma

        for i, lemma in enumerate(obj_lemmas):
            if i < 2:
                self._features[-1][i+2] = 'TRUE'

    def _matrix_prep_detector(self, children):
        prep_lemmas = children[children.pos=='prep'].lemma

        if prep_lemmas.shape[0]:
            self._features[-1][4] = prep_lemmas[0]

    def _complementizer_detector(self, children):
        _, grand_children = self.sentence.get_children(lemma=children.lemma,
                                                       gramrel=['COMP'], # should maybe specify pos=['v']
                                                       chilemma=['that', 'what', 'who', 'when',
                                                                 'where', 'how', 'why'])
                                                       
        if grand_children.shape[0]:
            self._features[-1][5] = grand_children.lemma[0]

    def _embedded_tense_detector(self, children):
        ## detect "to"
        self._features[-1][6] = 'to' if children.pos=='inf' else self._features[-1][6]
                           
    def _embedded_subject_detector(self, children):
        _, grand_children = self.sentence.get_children(lemma=children.lemma,
                                                       gramrel=['COMP', 'XCOMP'],
                                                       chigramrel=['SUBJ'])

        if grand_children.shape[0]:
            self._features[-1][7] = 'ACC' if grand_children.pos[0] == 'pro:obj' else 'UNKNOWN'
            self._features[-1][7] = 'NOM' if grand_children.pos[0] == 'pro:subj' else self._features[-1][6]
        

    @property
    def features(self):
        cols = ['lemma', 'mat_subj', 'mat_DO1', 'mat_DO2', 'mat_prep',
                'complementizer', 'emb_tense', 'emb_subj', 'emb_verb']
        return pandas.DataFrame(self._features, columns=)

for transcript in parsed_sentences:
    for sentence in transcript:
        
