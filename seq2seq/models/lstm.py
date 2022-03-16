import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture
# 继承了Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
# 需要重写build_model(), forward()等;

BI = False

@register_model('lstm')
class LSTMModel(Seq2SeqModel):
    """ Defines the sequence-to-sequence model class. """

    def __init__(self,
                 encoder,
                 decoder):

        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-embed-dim', type=int, help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', type=int, help='encoder hidden size')
        parser.add_argument('--encoder-num-layers', type=int, help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', help='bidirectional encoder')
        parser.add_argument('--encoder-dropout-in', help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', help='dropout probability for encoder output')

        parser.add_argument('--decoder-embed-dim', type=int, help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, help='decoder hidden size')
        parser.add_argument('--decoder-num-layers', type=int, help='number of decoder layers')
        parser.add_argument('--decoder-dropout-in', type=float, help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, help='dropout probability for decoder output')
        parser.add_argument('--decoder-use-attention', help='decoder attention')
        parser.add_argument('--decoder-use-lexical-model', help='toggle for the lexical model')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """ Constructs the model. """
        base_architecture(args)
        encoder_pretrained_embedding = None
        decoder_pretrained_embedding = None

        # Load pre-trained embeddings, if desired
        if args.encoder_embed_path:
            encoder_pretrained_embedding = utils.load_embedding(args.encoder_embed_path, src_dict)
        if args.decoder_embed_path:
            decoder_pretrained_embedding = utils.load_embedding(args.decoder_embed_path, tgt_dict)

        # Construct the encoder
        encoder = LSTMEncoder(dictionary=src_dict,
                              embed_dim=args.encoder_embed_dim,
                              hidden_size=args.encoder_hidden_size,
                              num_layers=args.encoder_num_layers,
                              bidirectional=args.encoder_bidirectional,
                              dropout_in=args.encoder_dropout_in,
                              dropout_out=args.encoder_dropout_out,
                              pretrained_embedding=encoder_pretrained_embedding)

        # Construct the decoder
        decoder = LSTMDecoder(dictionary=tgt_dict,
                              embed_dim=args.decoder_embed_dim,
                              hidden_size=args.decoder_hidden_size,
                              num_layers=args.decoder_num_layers,
                              dropout_in=args.decoder_dropout_in,
                              dropout_out=args.decoder_dropout_out,
                              pretrained_embedding=decoder_pretrained_embedding,
                              use_attention=bool(eval(args.decoder_use_attention)),
                              use_lexical_model=bool(eval(args.decoder_use_lexical_model)))
        return cls(encoder, decoder)


class LSTMEncoder(Seq2SeqEncoder):
    """ Defines the encoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=64,
                 num_layers=1,
                 bidirectional=True,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None):

        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_dim = 2 * hidden_size if bidirectional else hidden_size
        print(">>> initial encoder.")
        print("1.bidirectional:", bidirectional, self.bidirectional)
        # assert self.bidirectional == str(BI)  # string(False)

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)
            # -- 一个简单的存储固定大小的词典的嵌入向量的查找表
            # -- 给一个编号，嵌入层就能返回这个编号对应的嵌入向量
        print("[test-02: self.embd]:", type(self.embedding), '\n', self.embedding)
        # print("self.embedding[0]:", self.embedding[0])  # 不可以这样使用

        dropout_lstm = dropout_out if num_layers > 1 else 0.
        self.lstm = nn.LSTM(input_size=embed_dim,    # 看这里的prompt提示, return: [output, (hn, cn)]
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_lstm,
                            bidirectional=bool(bidirectional))

    def forward(self, src_tokens, src_lengths):
        """ Performs a single forward pass through the instantiated encoder sub-network. """
        # Embed tokens and apply dropout
        print(">>> forward():")
        # print("[test-01: src_tokens.size()]")
        # print(src_tokens.size())  # torch.Size([10, 22])  [本batch的句子数量, 单个句子的长度]
        batch_size, src_time_steps = src_tokens.size()  # -- 多少句子, 每个句子多少单词(time_step)
        src_embeddings = self.embedding(src_tokens)  # -- 获取根据src_tokens的词嵌入
        # print("[test-03: src_embeddings]")
        # print(type(src_embeddings))  # <class 'torch.Tensor'>
        # print(src_embeddings.size())  # torch.Size([10, 22, 64])
        # print("-----")
        _src_embeddings = F.dropout(src_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
        src_embeddings = _src_embeddings.transpose(0, 1)
        # print("[test-05: src_embeddings.T]")             # [句子数(batch), 单词数(length), hidden_size]
        # print("src_emb.T.size:", src_embeddings.size())  # tensor, torch.Size([6, 10, 64]) -> torch.Size([22, 10, 64])
        # print("src_emb.T:", src_embeddings)
        # 在本epoch内, 遍历了一个batch,该batch有10个句子, 每个句子有6~22个单词.

        # Pack embedded tokens into a PackedSequence
        packed_source_embeddings = nn.utils.rnn.pack_padded_sequence(src_embeddings, src_lengths.data.tolist())
        # PackedSequence将长度不同的序列数据封装成一个batch --- 表示一个batch
        # def pack_padded_sequence(input: Any, lengths: Any, ...)
        # - input – padded batch of variable length sequences. # 输入的多个句子
        # - lengths – list of sequences lengths of each batch element # 一个list, 包含每个句子的长度.
        # print("[test-04: src_lengths]")  # [每个句子的长度]. 其len=10表示有10个句子.
        # print("- type():", type(src_lengths))
        # print("- size():", src_lengths.size())
        # print("- value:", src_lengths)
        # print("- 2. type():", type(src_lengths.data.tolist()))
        # print("- 2. len():", len(src_lengths.data.tolist()))
        # print("- PSEmb-type:", type(packed_source_embeddings))
        # print("- PSEmb:", packed_source_embeddings)
        # print("- PSE.data.size:", packed_source_embeddings.data.size())
        # print("- PSE.data:", packed_source_embeddings.data)
        # ---------------------
        # [test-04: src_lengths]
        # - type(): <class 'torch.Tensor'>
        # - size(): torch.Size([10])
        # - value:
        # tensor([ 7,  7,  7,  7,  7,  7,  7,  6,  6,  6])  # 10个句子, 每个句子的单词数. (本次forward, 执行一个batch)
        # tensor([22, 21, 21, 21, 21, 21, 21, 20, 20, 20])
        # - 2. type(): <class 'list'>
        # - 2. len(): 10
        # - PSEmb-type: <class 'torch.nn.utils.rnn.PackedSequence'>
        # - PSEmb: PackedSequence(data=tensor([[ 0.9948, -1.4781, -0.8279,  ...,  0.5874,  0.3822,  0.8345],
        #         [-0.2574, -1.1080,  0.6396,  ...,  0.1324, -1.9230, -0.4358],
        #         [ 0.8931, -0.0028, -0.2772,  ..., -0.9611,  0.1570,  0.1099],
        #         ...,
        #         [ 2.1548,  1.4835,  2.6627,  ...,  0.5063,  0.9463,  0.1058],
        #         [ 2.1548,  1.4835,  2.6627,  ...,  0.5063,  0.9463,  0.1058],
        #         [ 2.1548,  1.4835,  2.6627,  ...,  0.5063,  0.9463,  0.1058]]),
        #         batch_sizes=tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        #         10, 10,  7,  1]), sorted_indices=None, unsorted_indices=None)
        # PSEmb的长度是什么?
        # 前10个是一个句子, 前10个是一个句子 ... 前7个是一个句子, 前1个是一个句子...
        # 总记: 20*10 + 7 + 1 = 208个单词.
        #
        # - PSE.data.size: torch.Size([208, 64]) # 总结208个单词, 每个单词hidden=64; - 208可变, 可以是共192个单词, etc.
        # - PSE.data: tensor([[ 0.9948, -1.4781, -0.8279,  ...,  0.5874,  0.3822,  0.8345],
        #         ...,
        #         [ 2.1548,  1.4835,  2.6627,  ...,  0.5063,  0.9463,  0.1058]])

        # Pass source input through the recurrent layer(s)
        # print(">>> check:", self.bidirectional)

        if self.bidirectional:  # biRNN的 encoder_layer, 双层
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:  # self.num_layers在输入时为1(lstm), 若bi, 在这里*2;
            state_size = self.num_layers, batch_size, self.hidden_size

        hidden_initial = src_embeddings.new_zeros(*state_size)
        context_initial = src_embeddings.new_zeros(*state_size)
        # [2, batch_size(本batch句子数量), 64]  # 2表示两个方向
        # print('[test-06-1]: hidden_initial')
        # print(f'size: {hidden_initial.size()}')  # [2, 10, 64]

        packed_outputs, (final_hidden_states, final_cell_states) = self.lstm(packed_source_embeddings,
                                                                             (hidden_initial, context_initial))
        # return [output, (hn, cn)]  # 最后一阶段的hidden和cell信息
        # print('[test-06-2]: final_hidden_states')
        # print(f'size-h: {final_hidden_states.size()}')  # [2, 10, 64]
        # print(f'size-c: {final_cell_states.size()}')  # [2, 10, 64]

        # Unpack LSTM outputs and optionally apply dropout (dropout currently disabled)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=0.)
        lstm_output = F.dropout(lstm_output, p=self.dropout_out, training=self.training)
        assert list(lstm_output.size()) == [src_time_steps, batch_size, self.output_dim]  # sanity check

        '''
        ___QUESTION-1-DESCRIBE-A-START___
        1.  Add tensor shape annotation to each of the output tensor
            output tensor means tensors on the left hand side of "="
            e.g., 
                sent_tensor = create_sentence_tensor(...) 
                # sent_tensor.size = [batch, sent_len, hidden]
        2.  Describe what happens when self.bidirectional is set to True. 
            #-- If self.bidirectional is True, the final_hidden_states and final_cell_states of two directions will be 
            # concatenated separately by the dimension of hidden units
            # 两个方向: the hidden units along the two directions are concatenated together.
            如果self.bidirectional为True，两个方向的final_hidden_states和final_cell_states会分别按照隐藏单元的维度拼接起来
        3.  What is the difference between final_hidden_states and final_cell_states?
            #-- The final_hidden_states is the last hidden unit of each sentence and the final_cell_states 
            # is the last cell unit of lstm cells
            # todo: 是否还要解释cell unit和hidden unit的区别&具体含义. 
        '''
        # print("[test-06-2]: final_hidden_states")
        # print(f'biRNN?: {self.bidirectional}')
        # print(f"size: {final_cell_states.size()}")  # torch.Size([2, 10, 64])
        if self.bidirectional:
            def combine_directions(outs):
                # 在dim2(hidden)上合并 (合并的是hidden unites)
                # print('out.size(0):', outs.size(0))
                return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
            final_hidden_states = combine_directions(final_hidden_states)
            # print(f"cat-size: {final_hidden_states.size()}")  # torch.Size([1, 10, 128])
            # -- # [2, 10, 64] => [1, 10, 128], 把两个方向的hidden64拼接 [加入batch, 表示10句话用以填充一次hidden]
            final_cell_states = combine_directions(final_cell_states)
            # [2, 10, 64] => [1, 10, 128]
            # --
        '''___QUESTION-1-DESCRIBE-A-END___'''

        # Generate mask zeroing-out padded positions in encoder inputs
        src_mask = src_tokens.eq(self.dictionary.pad_idx)  # ele-wise 判断是否被mask. return tensor, masked的位置为True;

        return {'src_embeddings': _src_embeddings.transpose(0, 1),  # [句子数(batch), 单词数(length), hidden_size]
                'src_out': (lstm_output, final_hidden_states, final_cell_states),
                'src_mask': src_mask if src_mask.any() else None}  # [batch_size, src_time_steps]


class AttentionLayer(nn.Module):
    """ Defines the attention layer class. Uses Luong's global attention with the general scoring function. """
    def __init__(self, input_dims, output_dims):
        super().__init__()
        # Scoring method is 'general'
        self.src_projection = nn.Linear(input_dims, output_dims, bias=False)
        self.context_plus_hidden_projection = nn.Linear(input_dims + output_dims, output_dims, bias=False)
        print("[test-09] attn-layer")
        print("input-dims:", input_dims)    # 128
        print("output-dims:", output_dims)  # 128
        print("src_proj:", self.src_projection)  # Linear(in_features=128, out_features=128, bias=False)
        print("con_+_hidden:", self.context_plus_hidden_projection)  # Linear(in_features=256, out_features=128, bias=False)

    def forward(self, tgt_input, encoder_out, src_mask):
        # tgt_input has shape = [batch_size, input_dims]
        # encoder_out has shape = [src_time_steps, batch_size, output_dims]
        # src_mask has shape = [batch_size, src_time_steps]
        # todo: 这里input/output是啥? encoder_out是上面return的 (三维tuple吗?)
        # todo: tgt_input是要输入的target句子吗?

        # Get attention scores
        # [batch_size, src_time_steps, output_dims]
        # print('[test-09]')
        # print("size-0::encoder_out:", encoder_out.size())  # torch.Size([22, 10, 128])
        # print("value-0::encoder_out:", encoder_out)
        encoder_out = encoder_out.transpose(1, 0)  # -- encoder的输出张量 [句子数，单词数，隐藏单元数]
        # print("size-1::encoder_out:", encoder_out.size())  # torch.Size([10, 22, 128])
        # print("value-1::encoder_out:", encoder_out)

        # [batch_size, 1, src_time_steps]
        attn_scores = self.score(tgt_input, encoder_out)
        # general_score: score(h(t), h(s)_) = h(t).T * Wa * h(s)_;

        '''
        ___QUESTION-1-DESCRIBE-B-START___
        1.  Add tensor shape annotation to each of the output tensor
            output tensor means tensors on the left hand side of "="
            e.g., 
                sent_tensor = create_sentence_tensor(...) 
                # sent_tensor.size = [batch, sent_len, hidden]
        2.  Describe how the attention context vector is calculated. 
            - First, we compute the attention score, by adding a new dimension to the masked token vector 'src_mask' 
              and adapting masked_filled function to it to make all the False value equal to '-inf'. 
              This way we can make the masked token account for nothing in the attention score matrix. 
            - Then delete the dimension added just now and the score remains and we get attention weights. 
            - At last, we multiple the attention weights and the output of the encoder to compute the attention 
              context of the model.
        3.  Why do we need to apply a mask to the attention scores?
            - To make sure that the attention mechanism will not share any information of the tokens in the future, 
              when we predict the token with previous ones.
         # # # 是说把padded_0的零分attn转换为-inf吗? 
         # 更详细的, 因为还要把attn_scores进行归一化, 有效的attn_score有正有负, 无效的attn_score=0; 
        '''
        if src_mask is not None:
            # print("[test-07]: unsqueeze")
            # print("size-0::src-mask:", src_mask.size())     # torch.Size([10, 22])
            # print("value-0::src-mask:", src_mask)

            src_mask = src_mask.unsqueeze(dim=1)
            # print("size-1::src-mask:", src_mask.size())     # torch.Size([10, 1, 22])  # 插入指定dim
            # print("value-1::src-mask:", src_mask)

            print("[test-08]: attn-score")
            # print("size::attn_score:", attn_scores.size())  # torch.Size([10, 1, 22])
            # print("value-0::attn_score:", attn_scores)
            attn_scores.masked_fill_(src_mask, float('-inf'))    # todo: 设定成-inf. 其余的attn有负值, 因此使用-inf;
            # print("value-1::attn_score:", attn_scores)         # 把在attn_scores中的value_0转化为value_-inf;

        # 公式7: alignment_vec = align(h_t, h_s) = softmax(score(h_t, h_s));
        # 所以, 这里的weights就是 alignment vector;
        attn_weights = F.softmax(attn_scores, dim=-1)           # 将attn_scores归一化[0, 1];
        # 使得每一句子(list[])的sum=1, 从而对每个word分配attn;  # 想象一下一个句子里面对不同的word有不同的attn权重(关注度);

        # print("size::attn_weights:", attn_weights.size())     # torch.Size([10, 1, 22])
        # print("value::attn_weights:", attn_weights)
        attn_context = torch.bmm(attn_weights, encoder_out).squeeze(dim=1)  # [10,1,22], [10,22,128]) => [10,1,128] => [10,128]
        # [√] todo: attn_context 表示什么?
        # context (with attention) vector: c = attn_w * encoder_hidden.T,  # 对src_hidden的加权;
        # 即: computed as the weighted average over all the source hidden states  [成立√]
        context_plus_hidden = torch.cat([tgt_input, attn_context], dim=1)
        # todo: context_plus_hidden 表示什么?
        # 公式(5):  h(t)~ = tanh(Wc[c(t); h(t)]  # 文中: 简单的把ct和ht的信息联合起来一起使用;
        # Wc来自哪里? - 使用linear(), 会自动introduce这个weight;
        attn_out = torch.tanh(self.context_plus_hidden_projection(context_plus_hidden))
        # todo: projection的作用是什么? 一个fc全连接层, 然后把"context_plus_hidden"的特征映射到这个dim=[x x]的空间?

        # 步骤总结: p4, 3.1, 最后一段: go from ht -> at -> ct -> ht~;
        '''___QUESTION-1-DESCRIBE-B-END___'''

        return attn_out, attn_weights.squeeze(dim=1)

    def score(self, tgt_input, encoder_out):
        """ Computes attention scores. """  # todo: 到底是 attention score 还是 alignment score?

        '''
        ___QUESTION-1-DESCRIBE-C-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  How are attention scores calculated? 
        3.  What role does batch matrix multiplication (i.e. torch.bmm()) play in aligning encoder and decoder representations?
            - bmm([b, n, m], [b, m, p]) => [b, m, p]
            - batch维度不变, [1, 22] * [22, 128] => [1, 128], 
        '''
        # general_score: score(h(t), h(s)_) = h(t).T * Wa * h(s)_;
        # 在哪里用了attention_weight Wa?
        print("[test-10]: Proj in Score")
        # print("size::encoder_out:", encoder_out.size())                            # torch.Size([10, 6, 128])
        # print("value::encoder_out:", encoder_out)
        # print("size::encoder_out_proj:", self.src_projection(encoder_out).size())  # torch.Size([10, 6, 128])
        # print("value::encoder_out_prj:", self.src_projection(encoder_out))
        projected_encoder_out = self.src_projection(encoder_out).transpose(2, 1)
        # print("size::encoder_out_proj_tran:", projected_encoder_out.size())        # torch.Size([10, 128, 6])
        # projection操作的作用:
        # 扩增维度. 若有linear=Linear(20, 30), 有输入x=(128, 20), 则linear(x) = (128, 30)
        # => linear(encoder_out) = (10, 22, 128) * (128, 128) => (10, 22, 128)  # 虽然size相同, 但是value不同;
        # 22或者6, 都是句子长度(单词个数);
        # y = x(W.T)  这里W.T是tensor_var, 属于内部变量;
        # todo: 那么这里projected_encoder_out表示什么? 表示被[权重]处理过的encoder_output吗? (Wa * hs)
        # todo: 疑问: Wa, 是指attention weight吗?, 实际上是从linear()函数中获取的, 不是我们算出来的吗?
        # todo: 调用线性变换linear()的作用? 数值确实变了, 有什么作用?

        attn_scores = torch.bmm(tgt_input.unsqueeze(dim=1), projected_encoder_out)  # todo: 这里用的不是简单的ht*hs吗?
        # todo: 解答: 即: 不是简单的 ht*hs, 而是 ht * (Wa*hs), 即: general_score;
        # todo: 但是一定是 ht * (Wa * hs)吗? 可以是 (ht * Wa) * hs 吗?
        # todo: 这个Wa到底表示什么? attention weight? alignment weight? 但是在上面已经求出来attn_weight了.

        # print("size::attn_score:", attn_scores.size())  # torch.Size([10, 1, 22])
        # [10, 1, 128] * [10, 128, 22] = [10, 1, 22]
        # 表示: 10个句子, 每个句子有一个score; ... 怎么解释? [22个浮点数], 有什么含义?
        print("score:", attn_scores)

        '''___QUESTION-1-DESCRIBE-C-END___'''

        return attn_scores


class LSTMDecoder(Seq2SeqDecoder):
    """ Defines the decoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=128,
                 num_layers=1,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None,
                 use_attention=True,
                 use_lexical_model=False):

        super().__init__(dictionary)

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        # Define decoder layers and modules
        self.attention = AttentionLayer(hidden_size, hidden_size) if use_attention else None

        self.layers = nn.ModuleList([nn.LSTMCell(
            input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
            hidden_size=hidden_size)
            for layer in range(num_layers)])

        self.final_projection = nn.Linear(hidden_size, len(dictionary))

        self.use_lexical_model = use_lexical_model
        if self.use_lexical_model:
            # __QUESTION-5: Add parts of decoder architecture corresponding to the LEXICAL MODEL here
            # TODO: --------------------------------------------------------------------- CUT
            pass
            # TODO: --------------------------------------------------------------------- /CUT

    def forward(self, tgt_inputs, encoder_out, incremental_state=None):
        """ Performs the forward pass through the instantiated model. """
        # Optionally, feed decoder input token-by-token
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]

        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Recover encoder input
        src_embeddings = encoder_out['src_embeddings']

        src_out, src_hidden_states, src_cell_states = encoder_out['src_out']
        src_mask = encoder_out['src_mask']
        src_time_steps = src_out.size(0)

        # Embed target tokens and apply dropout
        batch_size, tgt_time_steps = tgt_inputs.size()
        tgt_embeddings = self.embedding(tgt_inputs)
        tgt_embeddings = F.dropout(tgt_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, tgt_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        tgt_embeddings = tgt_embeddings.transpose(0, 1)

        # Initialize previous states (or retrieve from cache during incremental generation)
        '''
        ___QUESTION-1-DESCRIBE-D-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  Describe how the decoder state is initialized. 
        3.  When is cached_state == None? 
        4.  What role does input_feed play?
        '''
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            tgt_hidden_states, tgt_cell_states, input_feed = cached_state
        else:
            tgt_hidden_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            tgt_cell_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            input_feed = tgt_embeddings.data.new(batch_size, self.hidden_size).zero_()
        '''___QUESTION-1-DESCRIBE-D-END___'''

        # Initialize attention output node
        attn_weights = tgt_embeddings.data.new(batch_size, tgt_time_steps, src_time_steps).zero_()
        rnn_outputs = []

        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Cache lexical context vectors per translation time-step
        lexical_contexts = []

        for j in range(tgt_time_steps):
            # Concatenate the current token embedding with output from previous time step (i.e. 'input feeding')
            lstm_input = torch.cat([tgt_embeddings[j, :, :], input_feed], dim=1)

            for layer_id, rnn_layer in enumerate(self.layers):
                # Pass target input through the recurrent layer(s)
                tgt_hidden_states[layer_id], tgt_cell_states[layer_id] = \
                    rnn_layer(lstm_input, (tgt_hidden_states[layer_id], tgt_cell_states[layer_id]))

                # Current hidden state becomes input to the subsequent layer; apply dropout
                lstm_input = F.dropout(tgt_hidden_states[layer_id], p=self.dropout_out, training=self.training)

            '''
            ___QUESTION-1-DESCRIBE-E-START___
            1.  Add tensor shape annotation to each of the output tensor
            2.  How is attention integrated into the decoder? 
            3.  Why is the attention function given the previous target state as one of its inputs? 
            4.  What is the purpose of the dropout layer?
            '''
            if self.attention is None:
                input_feed = tgt_hidden_states[-1]
            else:
                input_feed, step_attn_weights = self.attention(tgt_hidden_states[-1], src_out, src_mask)
                attn_weights[:, j, :] = step_attn_weights

                if self.use_lexical_model:
                    # __QUESTION-5: Compute and collect LEXICAL MODEL context vectors here
                    # TODO: --------------------------------------------------------------------- CUT
                    pass
                    # TODO: --------------------------------------------------------------------- /CUT

            input_feed = F.dropout(input_feed, p=self.dropout_out, training=self.training)
            rnn_outputs.append(input_feed)
            '''___QUESTION-1-DESCRIBE-E-END___'''

        # Cache previous states (only used during incremental, auto-regressive generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (tgt_hidden_states, tgt_cell_states, input_feed))

        # Collect outputs across time steps
        decoder_output = torch.cat(rnn_outputs, dim=0).view(tgt_time_steps, batch_size, self.hidden_size)

        # Transpose batch back: [tgt_time_steps, batch_size, num_features] -> [batch_size, tgt_time_steps, num_features]
        decoder_output = decoder_output.transpose(0, 1)

        # Final projection
        decoder_output = self.final_projection(decoder_output)

        if self.use_lexical_model:
            # __QUESTION-5: Incorporate the LEXICAL MODEL into the prediction of target tokens here
            # TODO: --------------------------------------------------------------------- CUT
            pass
            # TODO: --------------------------------------------------------------------- /CUT

        return decoder_output, attn_weights


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 64)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', 'True')
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.25)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.25)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 128)
    args.decoder_num_layers = getattr(args, 'decoder_num_layers', 1)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.25)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.25)
    args.decoder_use_attention = getattr(args, 'decoder_use_attention', 'True')
    args.decoder_use_lexical_model = getattr(args, 'decoder_use_lexical_model', 'False')
