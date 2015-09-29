import os
import re
import nltk
import zipfile
import StringIO
import requests
import bs4

from nltk.downloader import Downloader

class CHILDESDir(object):
    
    def __init__(self, url, local_path):
        req = requests.get(url)
        soup = bs4.BeautifulSoup(req.text)

        anchors = soup.find_all('a', href=re.compile('^[A-Z]'))

        corpus_urls = [os.path.join(url, anch['href']) for anch in anchors]
        corpus_paths = [os.path.join(local_path, anch['href']) for anch in anchors]
        
        self.local_path = local_path
        self.name = os.path.basename(local_path.strip('/')).lower()

        self.subcorpora = self._get_subcorpora(corpus_urls, corpus_paths)

    def _get_subcorpora(self, corpus_urls, corpus_paths):
        children = {}

        for url, path in zip(corpus_urls, corpus_paths):
            if re.findall('\.zip$', url):
                child = CHILDESTranscripts(url, self.local_path)
            else:
                child = CHILDESDir(url, path)

            children[child.name] = child

        return children

    def download_subcorpus(self, name):
        child_name = name.lower()

        self.subcorpora[name].download()

        print '{} from {} successfully downloaded and extracted'.format(child_name, self.name)

    def download_subcorpora(self, corpus_names):
        for name in corpus_names:
            self.download_subcorpus(name)

    def download(self):
        for corpus in self.subcorpora.values():
            corpus.download()

        print '{} successfully downloaded and extracted'.format(self.name)

    def subcorpora_names(self):
        return self.subcorpora.keys()

class CHILDESTranscripts(object):

    def __init__(self, url, local_path):
        self.url = url
        self.local_path = local_path

        basename = os.path.basename(url)
        self.name = os.path.splitext(basename)[0].lower()

    def download(self):
        req = requests.get(self.url)
        strio = StringIO.StringIO(req.content)
        
        zfile = zipfile.ZipFile(strio)
        zfile.extractall(path=self.local_path)


def main():
    childes_xml_url = 'http://childes.psy.cmu.edu/data-xml/'

    user_data_path = Downloader.default_download_dir(Downloader())
    childes_corpus_path = os.path.join(user_data_path, 'corpora/CHILDES/')

    try:
        os.makedirs(childes_corpus_path)
    except OSError:
        pass

    return CHILDESDir(childes_xml_url, childes_corpus_path)


if __name__ == '__main__':
    childes_root = main()
    childes_root.download()
