#coding=utf8
import math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class SLUMultiHead(nn.Module):

    def __init__(self, config):
        super(SLUMultiHead, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.multi_head = MultiHeadAttention(config.embed_size, config.num_head, config.dropout)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids # size: (32, c)
        tag_mask = batch.tag_mask # size: (32, c)
        input_ids = batch.input_ids # size: (32, c)
        lengths = batch.lengths # [c, 10, 9, 9, 8, 8, ...] len: 32

        embed = self.word_embed(input_ids) # size: (32, c, 768)
        embed, _ = self.multi_head(embed) # size: (32, c, 768)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True) # PackedSequence(), data size: (s, 768), batch_sizes size: (32,), s is sum of batch_sizes
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # packed_rnn_out: PackedSequence(), data size: (s, 768), batch_sizes size: (32,), s is sum of batch_sizes
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

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


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_head, dropout):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_head = num_head
        self.head_size = embed_size // num_head
        self.dropout = dropout
        self.qkv_proj = nn.Linear(embed_size, embed_size * 3)
        self.out_proj = nn.Linear(embed_size, embed_size)

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size(-1)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e32)
        attention = torch.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_size = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_head, 3 * self.head_size)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = self.scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, embed_size)
        values = self.out_proj(values)
        return values, attention
        