import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.bert2vec import Bert2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None, bert_model=None, to_vec="word2vec"):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        if to_vec == "word2vec":
            cls.word2vec = Word2vecUtils(word2vec_path)
        elif to_vec == "bert2vec":
            cls.word2vec = Bert2vecUtils(bert_model)
            raise NotImplementedError
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, is_correct=False):
        # dataset = json.load(open(data_path, 'r'))
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            examples = []
            for di, data in enumerate(dataset):
                for ui, utt in enumerate(data):
                    ex = cls(utt, f'{di}-{ui}', is_correct=is_correct)
                    examples.append(ex)
            return examples

    def __init__(self, ex: dict, did, is_correct=False):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did

        if is_correct:
            self.utt = ex['manual_transcript']
        else:
            self.utt = ex['asr_1best']
        
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
