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
user_data_path = Downloader.default_download_dir(Downloader())
childes_corpus_path = os.path.join(user_data_path, 'corpora/CHILDES/')

corpora = CHILDESCollection(childes_corpus_path)


## represent the parsed sentences
parsed_sentences = CHILDESParsedSentences(corpora, args.corpus)
