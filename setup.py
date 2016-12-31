from setuptools import setup

setup(name='CHILDESPy',
      version='0.1dev0',
      description='Classes for easily searching and summarizing CHILDES corpora',
      url='http://github.com/aaronstevenwhite/CHILDESPy',
      author='Aaron Steven White',
      author_email='aswhite@jhu.edu',
      license='MIT',
      packages=['childespy'],
      install_requires=['requests',
                        'beautifulsoup4',
                        'nltk',
                        'numpy',
                        'scipy',
                        'pandas'],
      scripts=['bin/download_childes.py',
               'bin/load_childes_corpus.py'],      
      zip_safe=False)
