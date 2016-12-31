import os

from nltk.downloader import Downloader
from childespy import CHILDESDir

childes_xml_remote_root='http://childes.psy.cmu.edu/data-xml/'

user_data_path = Downloader.default_download_dir(Downloader())
childes_corpus_path = os.path.join(user_data_path, 'corpora/CHILDES/')

try:
    os.makedirs(childes_corpus_path)
except OSError:
    pass

CHILDESDir(childes_xml_remote_root, childes_corpus_path).download()
