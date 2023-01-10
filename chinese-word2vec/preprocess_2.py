import re
from loguru import logger
import os.path
import sys
from tqdm import tqdm

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger.info(f"{program}: parse the chinese corpus and delete english char and blank")

    fout = open(sys.argv[2], 'w', encoding='utf-8')
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        relu = re.compile(r'[ a-zA-Z〡ÀÂÆÇÈÉÊËÎÏÔÙÛÜàâæçèéêëîïôùûüŒœ]')  # delete english char and blank
        for line in tqdm(f):
            line = relu.sub('', line)
            fout.write(line)
    fout.close()
    