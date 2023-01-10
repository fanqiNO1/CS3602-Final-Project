import numpy as np
from utils.vocab import PAD, UNK
import torch

from transformers import BertTokenizer, BertModel, BertConfig

class Bert2vecUtils():
    def __init__(self, bert_model):
        super(Bert2vecUtils, self).__init__()
        self.bert_model = bert_model
        self.config = BertConfig.from_pretrained(bert_model)
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertModel.from_pretrained(bert_model, config=self.config)

    def load_embeddings(self, module, vocab, device='cpu'):
        """ Initialize the embedding with glove and char embedding
        """
        emb_size = module.weight.data.size(-1)
        outliers = 0
        for word in vocab.word2id:
            if word == PAD: # PAD symbol is always 0-vector
                module.weight.data[vocab[PAD]] = torch.zeros(emb_size, dtype=torch.float, device=device)
                continue
            # word_emb = self.word2vec.get(word, self.word2vec[UNK])
            # module.weight.data[vocab[word]] = torch.tensor(word_emb, dtype=torch.float, device=device)
            word_input = self.tokenizer.encode_plus(
                word, 
                padding="max_length", 
                truncation=True, 
                max_length=16, 
                add_special_tokens=True,
                return_tensors="pt"
            )
            word_out = self.model(**word_input)
            word_emb = word_out[0][0][0].detach().numpy()
            module.weight.data[vocab[word]] = torch.tensor(word_emb, dtype=torch.float, device=device)
        return 1 - outliers / float(len(vocab))
