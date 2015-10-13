'''
Download part or all of the CHILDES child-directed speech database as XML

Exposes one class ``CHILDESDir``, which takes a remote corpus root and
a local corpus root and directly copies either the entire file hierarchy
associated with the corpus root (all corpora in the database) or a
corpus named by the user

Terminology: a root or subroot is a nonterminal node in the CHILDES file
hierarchy that does not directly contain a node containing transcripts;
a corpus is a non-terminal node in the hierarchy that a collection of
does directly contain transcripts
'''

import os
import re
import nltk
import zipfile
import StringIO
import requests
import bs4

from nltk.downloader import Downloader

class CHILDESDir(object):
    
    def __init__(self, remote_root, local_root):
        self.remote_root = remote_root        
        self.local_root = local_root
        self.name = os.path.basename(local_root.strip('/')).lower()
        
        self.corpora = self._get_corpora()

    def _request_children(self):
        '''
        get location of children of remote root and construct pathnames
        for both remote root and local root
        '''
        
        req = requests.get(self.remote_root)
        soup = bs4.BeautifulSoup(req.text)

        anchors = soup.find_all('a', href=re.compile('^[A-Z]'))

        remote_children = [os.path.join(self.remote_root, anch['href']) for anch in anchors]
        local_children = [os.path.join(self.local_root, anch['href']) for anch in anchors]

        return zip(remote_children, local_children)
        
    def _get_corpora(self):
        '''
        for each child of the remote root, create a CHILDESTranscript
        if that child is a corpus or another CHILDESDir if it is a subroot
        '''
        
        children = {}
        
        for remote_root, path in self._request_children():
            if re.findall('\.zip$', remote_root):
                child = CHILDESCorpus(remote_root, self.local_root)
            else:
                child = CHILDESDir(remote_root, path)

            children[child.name] = child

        return children

    def download_corpus(self, name):
        '''download particular corpus in CHILDES'''
        
        child_name = name.lower()

        self.corpora[name].download()

        print '{} from {} successfully downloaded and extracted'.format(child_name, self.name)

    def download_corpora(self, corpus_names):
        for name in corpus_names:
            self.download_corpus(name)

    def download(self):
        '''download all of CHILDES'''
        
        for corpus in self.corpora.values():
            corpus.download()

        print 'successfully downloaded and extracted\t' + self.name

    def corpora_names(self):
        '''get names of corpora directly below this root'''
        
        return self.corpora.keys()

class CHILDESCorpus(object):

    def __init__(self, remote_root, local_root):
        self.remote_root = remote_root
        self.local_root = local_root

        basename = os.path.basename(remote_root)
        self.name = os.path.splitext(basename)[0].lower()

    def download(self):
        '''download all transcripts from this corpus'''
        
        req = requests.get(self.remote_root)
        strio = StringIO.StringIO(req.content)
        
        zfile = zipfile.ZipFile(strio)
        zfile.extractall(path=self.local_root)


def main(childes_xml_remote_root='http://childes.psy.cmu.edu/data-xml/'):
    user_data_path = Downloader.default_download_dir(Downloader())
    childes_corpus_path = os.path.join(user_data_path, 'corpora/CHILDES/')

    try:
        os.makedirs(childes_corpus_path)
    except OSError:
        pass

    CHILDESDir(childes_xml_remote_root, childes_corpus_path).download()


if __name__ == '__main__':
    main()
