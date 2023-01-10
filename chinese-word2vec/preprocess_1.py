from loguru import logger
import os
import sys
import argparse
from gensim.corpora import WikiCorpus
from tqdm import tqdm

@logger.catch
def parse_corpus(infile, outfile):
    space = ' '
    with open(outfile, 'w', encoding='utf-8') as fout:
        wiki = WikiCorpus(infile, dictionary={})
        for text in tqdm(wiki.get_texts()):
            fout.write(space.join(text) + '\n')


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger.info(f"{program}: parse the chinese corpus")

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',dest='infile',default='data/zhwiki-20230101-pages-articles-multistream.xml.bz2',help='input: Wiki corpus')
    parser.add_argument('-o','--output',dest='outfile',default='data/corpus.zhwiki.txt',help='output: Wiki corpus')
    args = parser.parse_args()

    infile = args.infile
    outfile = args.outfile

    parse_corpus(infile, outfile)
