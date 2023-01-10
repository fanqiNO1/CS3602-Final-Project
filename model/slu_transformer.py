#coding=utf8
import math
import torch
import torch.nn as nn
from transformers import BertTokenizer


class SLUTransformer(nn.Module):

    def __init__(self, config):
        super(SLUTransformer, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.tag_embed = nn.Embedding(config.num_tags, config.embed_size, padding_idx=0)
        self.pos_embed = PositionalEmbedding(num_features=config.embed_size, dropout=config.dropout)
        self.transformer = nn.Transformer(
            d_model=config.embed_size,
            nhead=config.num_head,
            num_encoder_layers=config.num_layer,
            num_decoder_layers=config.num_layer,
            dim_feedforward=config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.embed_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids # size: (32, c)
        tag_mask = batch.tag_mask # size: (32, c)
        input_ids = batch.input_ids # size: (32, c)
        lengths = batch.lengths # [c, 10, 9, 9, 8, 8, ...] len: 32

        tag_mask_bool = (1 - tag_mask).bool()

        if tag_ids is None:
            raise NotImplementedError

        src_embed = self.word_embed(input_ids)
        src_embed = self.pos_embed(src_embed)
        
        tgt_embed = self.tag_embed(tag_ids)
        tgt_embed = self.pos_embed(tgt_embed)

        mask = self.transformer.generate_square_subsequent_mask(max(lengths)).to(tgt_embed.device)
        out = self.transformer(
            src=src_embed, 
            tgt=tgt_embed, 
            tgt_mask=mask, 
            src_key_padding_mask=tag_mask_bool, 
            tgt_key_padding_mask=tag_mask_bool
        )
        out = self.dropout_layer(out)
        tag_output = self.output_layer(out, tag_mask, tag_ids)
        return tag_output
    
    
    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class PositionalEmbedding(nn.Module):
    def __init__(self, num_features, dropout, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, num_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_features, 2).float() * (-math.log(10000.0) / num_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, data):
        data = data + self.pe[:, :data.size(1)]
        return self.dropout(data)


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )