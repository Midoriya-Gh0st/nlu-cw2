import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.fc1 = generate_linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, state, encoder_padding_mask):
        """Forward pass of a single Transformer Encoder Layer"""
        residual = state.clone()

        '''
        ___QUESTION-6-DESCRIBE-D-START___
        1.  Add tensor shape annotation to EVERY TENSOR below (NOT just the output tensor)
        2.  What is the purpose of encoder_padding_mask? 
            - 一批次多个句子, 选取最长的句子的长度作为该批次每个句子的长度, 
            - 使用encoder_padding_mask标记长度以给那些长度不足的句子补零;
            - TODO: 这里是标记的[111100000]还是[0000011111]?
            - 即: 标记的是有效位还是零位;
        3.  What will the output shape of `state' Tensor be after multi-head attention?
            - TODO: 和1是一样的吗? 
        '''
        # print("size::state-0,", state.size())  # torch.Size([11, 10, 128]) [src_time_steps, batch_size, embed_dim]
        # print("size::encoder_padding_mask,", encoder_padding_mask.size())
        # TODO: encoder_padding_mask=None, 虽然在tras.py定义了, 但是没有调用?

        # QKV: # torch.Size([11, 10, 128]) [src_time_steps, batch_size, embed_dim]
        # mask: NoneType [batch_size, src_time_steps]
        state, _ = self.self_attn(query=state, key=state, value=state, key_padding_mask=encoder_padding_mask)
        # print("size::state-1:", state.size())
        # torch.Size([11, 10, 128]), [src_time_steps, batch_size, encoder_embed_dim]
        # TODO: 是encoder_embed_dim, 还是encoder_hidden_state;
        # input()

        '''
        ___QUESTION-6-DESCRIBE-D-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )

        self.encoder_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = generate_linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self,
                state,
                encoder_out=None,
                encoder_padding_mask=None,
                incremental_state=None,
                prev_self_attn_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False):
        """Forward pass of a single Transformer Decoder Layer"""

        # need_attn must be True if need_head_weights
        need_attn = True if need_head_weights else need_attn

        residual = state.clone()
        state, _ = self.self_attn(query=state,
                                  key=state,
                                  value=state,
                                  key_padding_mask=self_attn_padding_mask,
                                  need_weights=False,
                                  attn_mask=self_attn_mask)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        '''
        ___QUESTION-6-DESCRIBE-E-START___
        1.  Add tensor shape annotation to EVERY TENSOR below (NOT just the output tensor)
        2.  How does encoder attention differ from self attention? 
            - 结合图来讲, encoder-attn是en-de之间, self是只在en或者只在de的 彼此之间;
        3.  What is the difference between key_padding_mask and attn_mask? 
            - key_padding_mask是用来mask一个input_sent之于一个batch内最长的那个句子的的padding, 即: [11111000]
            - attn_mask: self_attn_mask, 防止生成tgt_2的时候, 被tgt_5的影响, 所以先把curren_tgt后面的给mask掉;
        4.  If you understand this difference, then why don't we need to give attn_mask here?
            - [6b] self_attn_mask;
            - [6e] encoder-decoder_attn_mask [不需要];
            - 这里的attn_mask是encoder-decoder attn_mask, 不包含decoder中tgt_words之间的上下文信息(句内影响);
              -- 即: attn_a = f(a-[1,2,3,4,5]), attn_b = f(b-[1,2,3,4,5]);
              -- attn_a和attn_b之间不会产生影响, 所以不需要使用;
              -- 而在transformer.py里面, 是self_attn_mask, 是关于: 
                 --- self_attn_a = f(a-[b,c,d,e,]), self_attn_b = f(b-[a,c,d,e,]) ... 
                 --- 这会产生 [句内, 词间]的影响;
                 --- 因此self_attn需要mask; 
        when we predict the first word, 
        '''
        state, attn = self.encoder_attn(query=state,
                                        key=encoder_out,
                                        value=encoder_out,
                                        key_padding_mask=encoder_padding_mask,
                                        need_weights=need_attn or (not self.training and self.need_attn))
        '''
        ___QUESTION-6-DESCRIBE-E-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.encoder_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self,
                 embed_dim,
                 num_attn_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.,
                 self_attention=False,
                 encoder_decoder_attention=False):
        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-NOTE
        You shouldn't need to change the __init__ of this class for your attention implementation
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim

        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads  # this is d_k in the paper
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.self_attention = self_attention
        self.enc_dec_attention = encoder_decoder_attention

        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
        assert self.head_embed_size * self.num_heads == self.embed_dim, "Embed dim must be divisible by num_heads!"
        assert not self.self_attention or kv_same_dim, "Self-attn requires query, key and value of equal size!"
        assert self.enc_dec_attention ^ self.self_attention, "One of self- or encoder- attention must be specified!"

        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                need_weights=True):
        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim

        # TODO: 为什么QKV的size都相等? 在源码, src_len = k.size(1);
        # print(">>> Q:", query.size())
        # print(">>> K:", key.size())  # torch.Size([11, 10, 128])
        # print(">>> V:", value.size())

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-START
        Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
        Note that you will have to handle edge cases for best model performance. Consider what behaviour should
        be expected if attn_mask or key_padding_mask are given?
        '''

        # attn is the output of MultiHead(Q,K,V) in Vaswani et al. 2017
        # attn must be size [tgt_time_steps, batch_size, embed_dim]
        # attn_weights is the combined output of h parallel heads of Attention(Q,K,V) in Vaswani et al. 2017
        # attn_weights must be size [num_heads, batch_size, tgt_time_steps, key.size(0)]
        # TODO: REPLACE THESE LINES WITH YOUR IMPLEMENTATION ------------------------ CUT

        # nn.MultiheadAttention()
        attn = torch.zeros(size=(tgt_time_steps, batch_size, embed_dim, self.num_heads))
        attn_weights = torch.zeros(size=(self.num_heads, batch_size, tgt_time_steps, key.size(0))) if need_weights else None
        # TODO: attn = attn_weights * value

        # 一组(单个head)的QKV:
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        # torch.size(tgt_time_steps, batch_size, embed_dim)

        # 单个head的embed_size:
        d_k = self.head_embed_size
        # self.num_heads * self.head_embed_size = embed_size (即: embed_dim)

        # 拆分为multi-head, 即: 把 [*, *, embed_size] -> [*, *, self.num_heads, self.head_embed_size]
        q, k, v = [x.view(x.size(0), x.size(1), self.num_heads, d_k) for x in (q, k, v)]
        # [tgt_time_steps, batch_size, self.num_heads, self.head_embed_size]

        # Transpose: 把head提到前面, 即: 第一维度为head_i. 所以之后方便把几个head给concatenate起来:
        q, k, v = [x.transpose(0, 2) for x in (q, k, v)]
        # [self.num_heads, batch_size, tgt_time_steps, self.head_embed_size]
        # torch.Size([2, 10, 11, 64])

        # 因为计算attn_score需要使用torch.bmm(), 要把qkv转换回3D:
        q, k, v = [x.contiguous().view(self.num_heads * batch_size, -1, self.head_embed_size) for x in (q, k, v)]
        # [self.num_heads * batch_size, -1, self.head_embed_size]
        # QKV:: torch.Size([20, 11, 64])

        # TODO: 这里为什么没有src_len和tgt_len的区别? 为什么是 [tgt_time_steps, tgt_time_steps], 不是[tgt, src]?
        # attn_score: [num_heads, batch_size, tgt_time_steps, self.head_embed_size]
        attn_score = torch.bmm(q, k.transpose(1, 2))
        # [num_heads * batch_size, tgt_time_steps, tgt_time_steps]

        # attn 缩放:
        scaled_score = attn_score / self.head_scaling
        # [num_heads * batch_size, tgt_time_steps, tgt_time_steps]
        # torch.Size([20, 11, 11])

        # 在进行softmax之前应用mask:
        # 在decoder, 二者都需要; 在encoder, 只需要padding_mask;
        # 2D-attn_mask, [tgt_time_steps, tgt_time_steps], torch.Size([11, 11])
        if attn_mask is not None:  # l,s -> tgt_seq_len, src_seq_len;
            # 把attn_mask提升到和score相同的3D:
            attn_mask = attn_mask.unsqueeze(dim=0)
            # [1, tgt_time_steps, tgt_time_steps]
            # [1, 11, 11]

            # 根据torch的源码, 元素非bool_type需要使用 "additive";
            scaled_score += attn_mask
            # [num_heads * batch_size, tgt_time_steps, tgt_time_steps]
            #  torch.Size([20, 11, 11])

        if key_padding_mask is not None:
            scaled_score = scaled_score.view(self.num_heads, batch_size, -1, scaled_score.size(-1))
            # [num_heads, batch_size, tgt_time_steps, tgt_time_steps]
            # torch.Size([2, 10, 14, 14])

            # generate key_padding_masks for each heads
            # key_padding_mask: (batch_size, source_seq_len)  # torch.Size([10, 14])
            key_padding_mask = key_padding_mask.unsqueeze(0).unsqueeze(2)  # 这里实际沿着num_heads发生了repeat()
            
            # [1, batch_size, 1, tgt_time_steps]
            # torch.Size([1, 10, 1, 14])

            # mask the scaled attn_weights
            scaled_score = scaled_score.masked_fill(key_padding_mask, float('-inf'))
            # [num_heads, batch_size, tgt_time_steps, tgt_time_steps]
            # torch.Size([2, 10, 14, 14])

            # 转回3d, 因为后面要用bmm
            scaled_score = scaled_score.view(self.num_heads * batch_size, -1, scaled_score.size(-1))
            # [num_heads * batch_size, tgt_time_steps, tgt_time_steps]

        attn_weights = F.softmax(scaled_score, dim=-1)
        # [num_heads * batch_size, tgt_time_steps, tgt_time_steps]
        # torch.Size([20, 11, 11])
        # print("the attn_weights:", attn_weights.size())
        # input()

        # 原先: 
        # ---------- 
            # key_padding_mask = key_padding_mask.unsqueeze(dim=1).repeat(self.num_heads,1,1)# generate key_padding_masks for each heads
            # scaled_attn_weights.masked_fill(key_padding_mask, -1e10) # mask the scaled attn_weights
        # if attn_mask is not None:
        #     scaled_attn_weights += attn_mask.unsqueeze(dim=0)
        # attn_weights = F.softmax(scaled_attn_weights, dim=-1)  # torch.size(num_head * batch_size, tgt_time_steps, tgt_time_steps)
        # apply softmax function
        # -----------


        # optional: apply dropout functions
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # [num_heads * batch_size, tgt_time_steps, tgt_time_steps]
        # torch.Size([20, 11, 11])
        # 同上

        # attention = attention_weights * value
        attn = torch.bmm(attn_weights, v)
        # [num_heads * batch_size, tgt_time_steps, head_embed_size]
        # torch.Size([20, 11, 64])
        # print("bmm-attn:", attn.size())
        # input()

        # 3. 把多个head, 拼接在一起
        # attn = torch.zeros(size=(tgt_time_steps, batch_size, embed_dim))
        attn = attn.view(self.num_heads, batch_size, -1, self.head_embed_size)
        # [num_heads, batch_size, tgt_time_steps, head_embed_size]

        attn = attn.transpose(0, 2)
        # [tgt_time_steps, batch_size, num_heads, head_embed_size]

        cat_attn = attn.contiguous().view(-1, batch_size, embed_dim)
        # [tgt_time_steps, batch_size, embed_dim = num_heads * head_embed_size]
        # print("cat-attn:", cat_attn.size())
        # input()

        # attn = torch.zeros(size=(tgt_time_steps, batch_size, embed_dim))
        # self.out_proj: nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # output_projection & Wo
        attn = self.out_proj(cat_attn)
        # [tgt_time_steps, batch_size, embed_dim]
        # torch.Size([11, 10, 128])
        # print("output_attn:", output_attn.size())
        # input()

        # TODO: key.size(0), 应该为src_time_steps, 为什么和tgt_time_steps一样?
        # TODO: 还要在结尾更改attn_weights吗? 结尾是否应该合并attn_weights? 在官方文档里, 把head合并了;
        # attn_weights = torch.zeros(size=(self.num_heads, batch_size, tgt_time_steps, key.size(0))) if need_weights else None
        # torch.Size([20, 11, 11])
        attn_weights = attn_weights.view(self.num_heads, batch_size, tgt_time_steps, key.size(0))
        # [self.num_heads, batch_size, tgt_time_steps, key.size(0)]
        # torch.Size([2, 10, 11, 11])
        # print("final-attn-weights:", attn_weights.size())
        # input()

        if not need_weights:
            return attn, None
        # TODO: --------------------------------------------------------------------- CUT

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-END
        '''

        return attn, attn_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEmbedding.get_embedding(init_size, embed_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embed_dim, padding_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).contiguous().view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # Zero pad in specific mismatch case
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.
        return emb

    def forward(self, inputs, incremental_state=None, timestep=None):
        batch_size, seq_len = inputs.size()
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # Expand embeddings if required
            self.weights = PositionalEmbedding.get_embedding(max_pos, self.embed_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            #   Positional embed is identical for all tokens during single step decoding
            pos = timestep.contiguous().view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(batch_size, 1, 1)

        # Replace non-padding symbols with position numbers from padding_idx+1 onwards.
        mask = inputs.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(inputs) * mask).long() + self.padding_idx

        # Lookup positional embeddings for each position and return in shape of input tensor w/o gradient
        return self.weights.index_select(0, positions.contiguous().view(-1)).contiguous().view(batch_size, seq_len, -1).detach()


def LayerNorm(normal_shape, eps=1e-5):
    return torch.nn.LayerNorm(normalized_shape=normal_shape, eps=eps, elementwise_affine=True)


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def generate_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
