import os, sys
import logging
import multiprocessing
from optparse import OptionParser
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def word2vec_train(infile,outmodel,outvector,size,window,min_count):
    '''train the word vectors by word2vec'''

    # train model
    model = Word2Vec(LineSentence(infile),vector_size=size,window=window,min_count=min_count,workers=multiprocessing.cpu_count(), epochs=6, max_vocab_size=25600)

    # # save model
    model.save(outmodel)
    model.wv.save_word2vec_format(outvector, binary=False)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(program)

    logger.info('running ' + program)

    # parse the parameters
    parser = OptionParser()
    parser.add_option('-i','--input',dest='infile',default='data/corpus.final.txt',help='zhwiki corpus')
    parser.add_option('-m','--outmodel',dest='wv_model',default='data/kun.w2v.ep6.trimmed.model',help='word2vec model')
    parser.add_option('-v','--outvec',dest='wv_vectors',default='data/kun.w2v.ep6.trimmed.vectors',help='word2vec vectors')
    parser.add_option('-s',type='int',dest='size',default=1024, help='word vector size')
    parser.add_option('-w',type='int',dest='window',default=5, help='window size')
    parser.add_option('-n',type='int',dest='min_count',default=8, help='min word frequency')

    (options,argv) = parser.parse_args()
    infile = options.infile
    outmodel = options.wv_model
    outvec = options.wv_vectors
    vec_size = options.size
    window = options.window
    min_count = options.min_count

    try:
        word2vec_train(infile, outmodel, outvec, vec_size, window, min_count)
        logger.info('word2vec model training finished')
    except Exception as err:
        logger.info(err)
    