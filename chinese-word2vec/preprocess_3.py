'''Use gensim to generate word2vec model from file'''

import os
import jiagu
import sys
from loguru import logger
from tqdm import tqdm

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger.info(f"{program}: segmenting")

    fout = open(sys.argv[2], 'w', encoding='utf-8')
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = jiagu.seg(line)
            fout.write(" ".join(line))
    fout.close()
    