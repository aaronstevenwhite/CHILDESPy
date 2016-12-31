import os, argparse

from nltk.downloader import Downloader
from childespy import CHILDESCorpus, CHILDESSentence

## initialize command line argument parser
parser = argparse.ArgumentParser(description='Load a CHILDES corpus')

parser.add_argument('--verbdata',
                    type=str,
                    default='Eng-NA-MOR/Brown/')

## parse command line argumentd
args = parser.parse_args()

user_data_path = Downloader.default_download_dir(Downloader())
childes_corpus_path = os.path.join(user_data_path, 'corpora/CHILDES')

corpus_root = os.path.join(childes_corpus_path, args.corpus_root)

corpus = CHILDESCorpus(corpus_root)
