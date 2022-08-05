import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class BAKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4,
            kq_same=1, final_fc_dim=512, num_attn_heads=8, seq_len=200, emb_type="qid", emb_path="", pretrain_dim=768, 
            use_pos=True, qmatrix=None):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "bakt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.model_type = self.model_name
        self.emb_type = emb_type
        self.use_pos = use_pos
        embed_l = d_model
        
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            if emb_type.find("aktrasch") != -1:
                self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上

        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question, embed_l)
        self.qa_embed = nn.Embedding(2, embed_l)
        if emb_type.find("aktrasch") == -1 or emb_type.find("predcurc") != -1:
            self.que_embed = nn.Embedding(self.n_pid + 1, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  seq_len=seq_len, kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type, use_pos=self.use_pos)


        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                    embed_l), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(embed_l,
                    final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 1)
        )

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = loss1
            self.l2 = loss2
            self.l3 = loss3
            num_layers = num_layers
            self.emb_size, self.hidden_size = d_model, d_model
            self.num_q, self.num_c = n_pid, n_question
            
            # if self.num_q > 0:
            #     self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = nheads
                # d_model = self.hidden_size# * 2
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
                encoder_norm = LayerNorm(d_model)
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
            elif self.emb_type.find("lstm") != -1:    
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            # self.qdrop = Dropout(dropout)
            self.qclasifier = Linear(self.hidden_size, self.num_c)
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb
            self.closs = CrossEntropyLoss()
            # 加一个预测历史准确率的任务
            if self.emb_type.find("his") != -1:
                self.start = start
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size//2), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size//2, 1))
                self.hisloss = nn.MSELoss()

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def predcurc(self, qemb, cemb, xemb, dcur, train):
        y2 = 0
        sm, c, cshft = dcur["smasks"], dcur["cseqs"], dcur["shft_cseqs"]
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)
        c = torch.cat([c[:,0:1], cshft], dim=-1)
        chistory = xemb
        if self.num_q > 0:
            catemb = qemb + chistory
        else:
            catemb = chistory
        # if self.emb_type.find("cemb") != -1: akt本身就加了cemb
        #     catemb += cemb

        if self.emb_type.find("trans") != -1:
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:])
            flag = sm[:,start:]==1
            y2 = self.closs(cpreds[flag], c[:,start:][flag])

        xemb = xemb + qh# + cemb
        if self.emb_type.find("qemb") != -1:
            xemb = xemb+qemb
        
        return y2, xemb

    def predcurc2(self, qemb, cemb, xemb, dcur, train):
        y2 = 0
        sm, c, cshft = dcur["smasks"], dcur["cseqs"], dcur["shft_cseqs"]
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)
        c = torch.cat([c[:,0:1], cshft], dim=-1)
        chistory = cemb
        if self.num_q > 0:
            catemb = qemb + chistory
        else:
            catemb = chistory

        if self.emb_type.find("trans") != -1:
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:])
            flag = sm[:,start:]==1
            y2 = self.closs(cpreds[flag], c[:,start:][flag])

        # xemb = xemb+qh
        # if self.separate_qa:
        #     xemb = xemb+cemb
        cemb = cemb + qh
        xemb = xemb+qh
        if self.emb_type.find("qemb") != -1:
            cemb = cemb+qemb
            xemb = xemb+qemb
        
        return y2, cemb, xemb

    def changecemb(self, qemb, cemb):
        catemb = cemb
        if self.emb_type.find("qemb") != -1:
            catemb += qemb
        if self.emb_type.find("trans") != -1:
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
        else:
            qh, _ = self.qlstm(catemb)
        
        cemb = cemb + qh
        if self.emb_type.find("qemb") != -1:
            cemb = cemb+qemb
        
        return cemb

    def afterpredcurc(self, h, dcur):
        y2 = 0
        sm, c, cshft = dcur["smasks"], dcur["cseqs"], dcur["shft_cseqs"]
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)
        c = torch.cat([c[:,0:1], cshft], dim=-1)
        
        start = 1
        cpreds = self.qclasifier(h[:,start:,:])
        flag = sm[:,start:]==1
        y2 = self.closs(cpreds[flag], c[:,start:][flag])
        
        return y2

    def predhis(self, h, dcur):
        sm = dcur["smasks"]
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)

        # predict history correctness rates
        
        start = self.start
        rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
        rsm = sm[:,start:]
        rflag = rsm==1
        # rtrues = torch.cat([dcur["historycorrs"][:,0:1], dcur["shft_historycorrs"]], dim=-1)[:,start:]
        padr = torch.zeros(h.shape[0], 1).to(device)
        rtrues = torch.cat([padr, dcur["historycorrs"]], dim=-1)[:,start:]
        # rtrues = dcur["historycorrs"][:,start:]
        # rtrues = dcur["totalcorrs"][:,start:]
        # print(f"rpreds: {rpreds.shape}, rtrues: {rtrues.shape}")
        y3 = self.hisloss(rpreds[rflag], rtrues[rflag])

        # h = self.dropout_layer(h)
        # y = torch.sigmoid(self.out_layer(h))
        return y3


    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        sm = dcur["smasks"]
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        emb_type = self.emb_type

        # Batch First
        q_embed_data = self.q_embed(q_data)
        pid_embed_data = None

        if self.n_pid > 0: # have problem id
            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                # q_embed_data = q_embed_data + pid_embed_data * \
                #     q_embed_diff_data  # uq *d_ct + c_ct # question encoder
                final_q_embed_data = q_embed_data + pid_embed_data + \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                final_qa_embed_data = self.qa_embed(target)+q_embed_data
            else:
                qa_embed_data = self.qa_embed(target)+q_embed_data
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
                qa_embed_data = qa_embed_data + pid_embed_data * \
                        (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
                # c_reg_loss = (pid_embed_data ** 2.).sum() * self.rashl2 # rasch部分loss
                final_q_embed_data, final_qa_embed_data = q_embed_data, qa_embed_data

        # print(f"final_qa_embed_data: {final_qa_embed_data}")

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        y2, y3 = 0, 0
        if emb_type in ["qid", "qidaktrasch"]:
            d_output = self.model(final_q_embed_data, final_qa_embed_data)

            concat_q = torch.cat([d_output, final_q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        elif emb_type.endswith("predcurc"): # predict current question' current concept
            q_embed_data, qa_embed_data = final_q_embed_data, final_qa_embed_data
            # predict concept
            qemb = self.que_embed(pid_data)

            # predcurc(self, qemb, cemb, xemb, dcur, train):
            cemb = q_embed_data
            if emb_type.find("noxemb") != -1:
                y2, q_embed_data, qa_embed_data = self.predcurc2(qemb, cemb, qa_embed_data, dcur, train)
            else:
                y2, qa_embed_data = self.predcurc(qemb, cemb, qa_embed_data, dcur, train)
            
            # q_embed_data = self.changecemb(qemb, cemb)

            # predict response
            d_output = self.model(q_embed_data, qa_embed_data)
            # if emb_type.find("after") != -1:
            #     curh = self.model(q_embed_data+qemb, qa_embed_data+qemb)
            #     y2 = self.afterpredcurc(curh, dcur)
            if emb_type.find("his") != -1:
                y3 = self.predhis(d_output, dcur)

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)

        if train:
            return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, seq_len, dropout, kq_same, model_type, emb_type, use_pos):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type
        self.use_pos = use_pos
        self.seq_len = seq_len

        if self.use_pos:
            self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=self.seq_len)  
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
            for _ in range(n_blocks)
        ])

    def forward(self, q_embed_data, qa_embed_data, forget_rate=None, pid_embed_data=None):
        # target shape  bs, seqlen
        # print(f"forget_rate: {forget_rate}")
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        if self.use_pos:
            q_posemb = self.position_emb(q_embed_data)
            q_embed_data = q_embed_data + q_posemb
            qa_posemb = self.position_emb(qa_embed_data)
            qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True, forget_rate=forget_rate, pdiff=pid_embed_data) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, emb_type):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, forget_rate=None, pdiff=None):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, forget_rate=forget_rate, pdiff=pdiff) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, forget_rate=forget_rate, pdiff=pdiff)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.emb_type = emb_type

        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            # constant_(self.attnlinear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, forget_rate=None, pdiff=None):

        bs = q.size(0)


        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        if self.emb_type.find("pdiff") == -1:
            pdiff = None
        scores = attention(q, k, v, self.d_k,
                        mask, self.dropout, zero_pad, gammas, forget_rate, pdiff)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            # # need: torch.Size([64, 1, 200]), scores: torch.Size([64, 200, 200]), v: torch.Size([64, 200, 32])
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1) # 所有v后置一位
        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, forget_rate=None, pdiff=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    # print(f"scores: {scores.shape}")

    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    # x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    # x2 = x1.transpose(0, 1).contiguous()

    # with torch.no_grad():
    #     scores_ = scores.masked_fill(mask == 0, -1e32)
    #     scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
    #     scores_ = scores_ * mask.float().to(device) # 结果和上一步一样
    #     distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
    #     disttotal_scores = torch.sum(
    #         scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
    #     # print(f"distotal_scores: {disttotal_scores}")
    #     position_effect = torch.abs(
    #         x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
    #     # bs, 8, sl, sl positive distance
    #     dist_scores = torch.clamp(
    #         (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
    #     dist_scores = dist_scores.sqrt().detach()
    # m = nn.Softplus()
    # gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    # # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    # if pdiff == None:
    #     total_effect = torch.clamp(torch.clamp(
    #         (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
    # else:
    #     diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
    #     diff = diff.sigmoid().exp()
    #     total_effect = torch.clamp(torch.clamp(
    #         (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
    # scores = scores * total_effect
    # print(f"forget_rate: {forget_rate.shape}")
    if forget_rate is not None:
        forget_rate = forget_rate.repeat(1,1,scores.shape[3]).unsqueeze(-1).permute(0,3,1,2)
        # print(f"forget_rate: {forget_rate.shape}")
        scores = scores * forget_rate
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
