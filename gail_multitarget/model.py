import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Variable
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

optimizer_item = {
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adamax": torch.optim.Adamax,
    "rprop": torch.optim.Rprop,
    "sgd": torch.optim.SGD,
    'adaw': torch.optim.AdamW,
}


class SmilesGAILModel:
    def __init__(self, config, vocab):
        self.device = config.device
        model_config = config.model.transformers

        self.generator = Generator(model_config, vocab, device=self.device)
        disc_config = config.model.rnnnet
        self.discriminator = Discriminator(disc_config, vocab, self.device)


class GPTDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(GPTDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, tgt_mask=None,
                tgt_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class GPTDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(GPTDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(GPTDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # type:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # FeedForward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # add
        tgt = tgt + self.dropout2(tgt2)
        # norm
        tgt = self.norm2(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=140):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=140):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        a = weight[:x.size(0), :]
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length,
                 pos_dropout, trans_dropout,device):
        super().__init__()
        self.d_model = d_model
        # self.embed_src = nn.Embedding(vocab_size, d_model)
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionEncoding(d_model, pos_dropout, max_seq_length)

        decoder_layers = GPTDecoderLayer(d_model, nhead, dim_feedforward, trans_dropout)
        self.transformer_encoder = GPTDecoder(decoder_layers, num_decoder_layers)

        self.fc = nn.Linear(d_model, vocab_size)
        self.device = device

    def __call__(self, x):
        tgt_mask = self.gen_nopeek_mask(x.shape[1])
        return self.forward(x, tgt_key_padding_mask=None, tgt_mask=tgt_mask)

    def gen_nopeek_mask(self, length):
        # mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def forward(self, tgt, tgt_key_padding_mask, tgt_mask):
        # (S ,N ,EMBEDDING_DIM) 所以先要转置
        tgt = tgt.transpose(0, 1)
        tgt = self.pos_enc((self.embed_tgt(tgt)) * math.sqrt(self.d_model))
        output = self.transformer_encoder(tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.transpose(0, 1)
        return self.fc(output)


class Generator:
    def __init__(self, config, vocab, device):
        self.config = config
        self.device = device
        self.vocab = vocab
        self.voc = vocab.get_default_tokens()
        self.voc_length = vocab.get_tokens_length()

        self.model = DecoderTransformer(self.voc_length,
                                        config.d_model,
                                        config.nhead,
                                        config.num_decoder_layers,
                                        config.dim_feedforward,
                                        config.max_seq_length + 1,
                                        config.pos_dropout,
                                        config.trans_dropout,device).to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad, reduction='sum')
        opt_config = config.optimization
        optimizer = optimizer_item.get(opt_config["optimizer_name"].lower())
        self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), lr=opt_config.lr)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def train_batch_smiles(self, batch):
        self.optimizer.zero_grad()
        logits = self.model(batch['index'])
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = self.criterion(log_probs.view(-1, log_probs.size(-1)), batch['label'].view(-1))
        num = batch['lens'].sum()
        loss = loss / num
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def load_model(self, path, device):
        self.model.load_state_dict(torch.load(path, map_location=torch.device(device)))

    def save_model(self, path):
        file_path = os.path.split(path)[0]
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save(self.model.state_dict(), path)

    def generate(self, batch_size, max_length=140):
        start_token = Variable(torch.zeros(batch_size, 1).long().to(self.device), self.device)
        start_token[:] = self.voc.index('<')
        input_vector = start_token
        # print(batch_size)
        sequences = start_token
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        for step in range(max_length):
            logits = self.model(input_vector)
            logits_step = logits[:, step, :]

            prob = F.softmax(logits_step, dim=1)
            input_vector = torch.multinomial(prob, 1)
            input_vector_fi = torch.mul(input_vector, ~finished.view(-1, 1))

            EOS_sampled = input_vector.view(-1) == self.voc.index('>')
            finished = torch.ge(finished + EOS_sampled, 1)

            # because there are no hidden layer in transformer, so we need to append generated word in every step as the input_vector
            # need to concat prior words as the sequences and input
            sequences = torch.cat((sequences, input_vector_fi), 1)
            input_vector = sequences
            if torch.prod(finished) == 1:
                break
        smiles = []
        for seq in sequences.cpu().numpy():
            smiles.append(self.vocab.decode(seq))
        return smiles

    def generate_prob(self, batch_size, max_length=140):
        start_token = Variable(torch.zeros(batch_size, 1).long().to(self.device), self.device)
        start_token[:] = self.voc.index('<')
        input_vector = start_token
        # print(batch_size)
        sequences = start_token
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        log_probs = torch.zeros((batch_size, 1)).to(self.device)
        for step in range(max_length):
            logits = self.model(input_vector)
            logits_step = logits[:, step, :]

            prob = F.softmax(logits_step, dim=1)
            log_prob = prob.log()
            input_vector = torch.multinomial(prob, 1)
            input_vector_fi = torch.mul(input_vector, ~finished.view(-1, 1))

            EOS_sampled = input_vector.view(-1) == self.voc.index('>')
            finished = torch.ge(finished + EOS_sampled, 1)

            sep_log_probs = torch.gather(log_prob, dim=-1, index=input_vector_fi)
            sep_log_probs = torch.mul(sep_log_probs, ~finished.view(-1, 1))

            log_probs = torch.cat((log_probs, sep_log_probs), 1)

            sequences = torch.cat((sequences, input_vector_fi), 1)
            input_vector = sequences
            if torch.prod(finished) == 1:
                break
        smiles = []
        for seq in sequences.cpu().numpy():
            smiles.append(self.vocab.decode(seq))
        return smiles, log_probs[:, 1:], log_probs.sum(dim=1)

    def compute_log_probs(self, batch):
        logits = self.model(batch['index'])
        log_probs = torch.log_softmax(logits, dim=-1)
        select_log_probs = torch.gather(log_probs, dim=-1, index=batch['label'].long().unsqueeze(-1)).squeeze(-1)
        nonzero_ind = torch.not_equal(batch['label'], self.vocab.pad)
        out = select_log_probs.mul(nonzero_ind)
        out_sum = out.sum(-1)

        loss = self.criterion(log_probs.view(-1, log_probs.size(-1)), batch['label'].view(-1))
        num = batch['lens'].sum()
        loss = loss / num
        return {"log_probs": out_sum, "loss": loss}

class RnnNet_DLGN(nn.Module):
    def __init__(self, config, vocab, device):
        super(RnnNet_DLGN, self).__init__()

        self.vocab = vocab
        self.voc_length = vocab.get_tokens_length()
        self.embed_dim = config.embed_dim
        self.blstm_dim = config.blstm_dim
        self.hidden_size = config.blstm_dim
        self.liner_out_dim = config.liner_out_dim
        self.num_dir = config.num_dir
        self.num_layers = config.num_layers
        self.out_dim = config.out_dim
        self.bidirectional = config.bidirectional

        self.num_dir = 1
        if self.bidirectional:
            self.num_dir += 1

        self.embeddings = nn.Embedding(self.voc_length, self.embed_dim, padding_idx=0)
        self.gru1 = nn.GRU(self.embed_dim, self.blstm_dim, num_layers=self.num_layers,
                          bidirectional=self.bidirectional, dropout=0.3,
                          batch_first=True)
        self.gru2 = nn.GRU(self.embed_dim, self.blstm_dim, num_layers=self.num_layers,
                          bidirectional=self.bidirectional, dropout=0.3,
                          batch_first=True)
        self.hidden2out1 = nn.Sequential(nn.Linear(2 * self.num_dir * self.blstm_dim, self.blstm_dim), nn.Sigmoid(),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(self.blstm_dim, self.blstm_dim), nn.Sigmoid(), nn.Dropout(p=0.3))
        self.hidden2out2 = nn.Sequential(nn.Linear(2 * self.num_dir * self.blstm_dim, self.blstm_dim), nn.Sigmoid(),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(self.blstm_dim, self.blstm_dim), nn.Sigmoid(), nn.Dropout(p=0.3))

        self.double_out1 = nn.Sequential(nn.Linear(2 * self.hidden_size, self.blstm_dim), nn.Sigmoid(), nn.Dropout(p=0.3))
        self.double_out2 = nn.Sequential(nn.Linear(2 * self.hidden_size, self.blstm_dim), nn.Sigmoid(), nn.Dropout(p=0.3))

        self.fc = nn.Sequential(nn.Linear(2 * self.blstm_dim, self.out_dim), nn.Sigmoid(), nn.Dropout(p=0.3))

        self.device = device

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers * self.num_dir, batch_size, self.hidden_size).to(self.device)

    def forward(self, inp1, inp2, inp3):
        '''

        :param inp1: demonstration1 [batch_size, length]
        :param inp2: demonstration2 [batch_size,length]
        :param inp3: gneration [batch_size,length]
        :return:
        '''
        hidden_h1 = self.init_hidden(inp1.size(0))
        hidden_h2 = self.init_hidden(inp2.size(0))
        hidden_h3 = self.init_hidden(inp3.size(0))

        # contrasive discriminator
        x1 = self.embeddings(inp1.long())
        x2 = self.embeddings(inp2.long())
        x3 = self.embeddings(inp3.long())


        _, hidden_x1 = self.gru1(x1, hidden_h1)
        _, hidden_x2 = self.gru2(x2, hidden_h2)

        _, hidden_x31 = self.gru1(x3, hidden_h3)
        _, hidden_x32 = self.gru2(x3, hidden_h3)

        # batch_size x 4 x hidden_dim
        hidden_x1 = hidden_x1.permute(1, 0, 2).contiguous()
        hidden_x2 = hidden_x2.permute(1, 0, 2).contiguous()

        hidden_x31 = hidden_x31.permute(1, 0, 2).contiguous()
        hidden_x32 = hidden_x32.permute(1, 0, 2).contiguous()

        # # batch_size x 4*hidden_dim
        x1_out = self.hidden2out1(hidden_x1.view(inp1.size(0), -1))
        x2_out = self.hidden2out2(hidden_x2.view(inp2.size(0), -1))

        x31_out = self.hidden2out1(hidden_x31.view(inp3.size(0), -1))
        x32_out = self.hidden2out2(hidden_x32.view(inp3.size(0), -1))


        logits_prop1 = torch.cat([x1_out, x31_out], dim=1)
        logits_prop1 = self.double_out1(logits_prop1)

        logits_prop2 = torch.cat([x2_out, x32_out], dim=1)
        logits_prop2 = self.double_out2(logits_prop2)

        logits = torch.cat([logits_prop1, logits_prop2], dim=1)
        out = self.fc(logits)

        # out = F.softmax(out, dim=-1)

        return out

class RnnNet(nn.Module):
    def __init__(self, config, vocab, device):
        super(RnnNet, self).__init__()

        self.vocab = vocab
        self.voc_length = vocab.get_tokens_length()
        self.embed_dim = config.embed_dim
        self.blstm_dim = config.blstm_dim
        self.hidden_size = config.blstm_dim
        self.liner_out_dim = config.liner_out_dim
        self.num_dir = config.num_dir
        self.num_layers = config.num_layers
        self.out_dim = config.out_dim
        self.bidirectional = config.bidirectional

        self.num_dir = 1
        if self.bidirectional:
            self.num_dir += 1

        self.embeddings = nn.Embedding(self.voc_length, self.embed_dim, padding_idx=0)
        self.gru = nn.GRU(self.embed_dim, self.blstm_dim, num_layers=self.num_layers,
                          bidirectional=self.bidirectional, dropout=0.3,
                          batch_first=True)
        self.hidden2out = nn.Sequential(nn.Linear(2 * self.num_dir * self.blstm_dim, self.blstm_dim), nn.Sigmoid(),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(self.blstm_dim, self.blstm_dim), nn.Sigmoid(), nn.Dropout(p=0.3))

        self.Linear = nn.Sequential(nn.Linear(3 * self.hidden_size, self.blstm_dim), nn.Sigmoid(), nn.Dropout(p=0.3))
        self.fc = nn.Sequential(nn.Linear(self.blstm_dim, self.out_dim), nn.Sigmoid(), nn.Dropout(p=0.3))

        self.device = device

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers * self.num_dir, batch_size, self.hidden_size).to(self.device)

    def forward(self, inp1, inp2, inp3):
        '''

        :param inp1: demonstration1 [batch_size, length]
        :param inp2: demonstration2 [batch_size,length]
        :param inp3: generation [batch_size,length]
        :return:
        '''
        hidden_h1 = self.init_hidden(inp1.size(0))
        hidden_h2 = self.init_hidden(inp2.size(0))
        hidden_h3 = self.init_hidden(inp3.size(0))

        # contrasive discriminator
        x1 = self.embeddings(inp1.long())
        x2 = self.embeddings(inp2.long())
        x3 = self.embeddings(inp3.long())

        _, hidden_x1 = self.gru(x1, hidden_h1)
        _, hidden_x2 = self.gru(x2, hidden_h2)
        _, hidden_x3 = self.gru(x3, hidden_h3)

        # batch_size x 4 x hidden_dim
        hidden_x1 = hidden_x1.permute(1, 0, 2).contiguous()
        hidden_x2 = hidden_x2.permute(1, 0, 2).contiguous()
        hidden_x3 = hidden_x3.permute(1, 0, 2).contiguous()
        # # batch_size x 4*hidden_dim
        x1_out = self.hidden2out(hidden_x1.view(inp1.size(0), -1))
        x2_out = self.hidden2out(hidden_x2.view(inp2.size(0), -1))
        x3_out = self.hidden2out(hidden_x3.view(inp3.size(0), -1))

        logits_r = torch.cat([x1_out, x2_out, x3_out], dim=1)
        logits_r = self.Linear(logits_r)

        out = self.fc(logits_r)
        # probs = F.softmax(out, dim=-1)
        return out


class Discriminator:
    def __init__(self, config, vocab, device):
        self.device = device
        # self.model = RnnNet(config, vocab, self.device).to(self.device)
        self.model = RnnNet_DLGN(config, vocab, self.device).to(self.device)
        opt_config = config.optimization
        optimizer = optimizer_item.get(opt_config["optimizer"].lower(), None)
        self.optimizer = optimizer(self.model.parameters(), lr=opt_config.lr)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save_model(self, path):
        file_path = os.path.split(path)[0]
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save(self.model.state_dict(), path)

    def compute_reward_gru(self, input1, input2, gen_input):
        out = self.model.forward(input1, input2, gen_input)
        log_probs = F.log_softmax(out, dim=-1)
        loss = -log_probs[:, 0].mean()
        return {"reward": log_probs[:, 1].exp(), "loss": loss}
