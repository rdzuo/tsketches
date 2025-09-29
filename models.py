from torch import nn
import torch
from torch.nn.functional import linear, softmax, dropout, pad
from torch.nn import functional as F
from sklearn import metrics
import numpy as np
from math import sqrt

import math
from typing import List, Tuple
from sympy import Poly, legendre, Symbol, chebyshevt
from scipy.special import eval_legendre
from functools import partial
from torch import Tensor
'''
def choose_model(model_tobuild):
    if model_tobuild = 'transformerbase':
        model = Transformer_base()
'''



def analyze_classification(y_pred, y_true, class_names):
    maxcharlength = 35

    in_pred_labels = set(list(y_pred))
    y_true = y_true.astype(int)
    in_true_labels = set(list(y_true))

    existing_class_ind = sorted(list(in_pred_labels | in_true_labels))
    class_strings = [str(name) for name in class_names]  # needed in case `class_names` elements are not strings
    existing_class_names = [class_strings[ind][:min(maxcharlength, len(class_strings[ind]))] for ind in
                                 existing_class_ind]  # a little inefficient but inconsequential

    # Confusion matrix
    ConfMatrix = metrics.confusion_matrix(y_true, y_pred)


    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    ConfMatrix_normalized_row = ConfMatrix.astype('float') / ConfMatrix.sum(axis=1)[:, np.newaxis]

    # Analyze results
    total_accuracy = np.trace(ConfMatrix) / len(y_true)
    print('Overall accuracy: {:.3f}\n'.format(total_accuracy))
    return total_accuracy


def iid_gaussian(m, d):
    return torch.randn((m, d))

def orthogonal_gaussian(m, d):
    def orthogonal_square():
        q, _ = torch.qr(iid_gaussian(d, d))
        return q.t()

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = torch.vstack(blocks)
    matrix /= torch.sqrt(torch.tensor(num_squares + remainder / d))

    return matrix


def phi_trig(h,m,random_feats):
    sin = lambda x: torch.sin(2 * np.pi * x)
    cos = lambda x: torch.cos(2 * np.pi * x)
    fs = [sin, cos]


    def func(x):
        return (h(x)/torch.sqrt(torch.tensor(m)) *
                torch.cat([f(torch.einsum("bld,bmd->blm", x, random_feats)) for f in fs], dim=-1)
        )

    return func


def phi_postive(h,m,random_feats):
    def func(x):
        return (h(x)/torch.sqrt(torch.tensor(m)) *
                torch.cat([torch.exp(torch.einsum("bld,bmd->blm", x, random_feats))], dim=-1)
        )

    return func

def mse(a, b):
    return torch.square(a - b).mean()



class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(1), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)



class Transformer_vanilla(nn.Module):
    def __init__(self, number_att, d_max_len, d_length, d_model,number_shape,number_class, device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')):
        super(Transformer_vanilla, self).__init__()
        self.d_length = d_length
        self.d_model = d_model
        self.number_shape = number_shape
        self.number_class = number_class
        self.relu = nn.ReLU()

        self.num_att = number_att
        self.linear = nn.Linear(d_max_len, d_length)


        self.layer_q = nn.Linear(d_length,d_model,bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length,d_model,bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)
        self.one = torch.unsqueeze(torch.eye(d_length),0).to(device)

        self.layer_add = nn.Linear(d_length, d_model)
        self.dropout = nn.Dropout(0.1)
        # self.pe = nn.Parameter(torch.empty(1, number_shape, d_length))
        # nn.init.uniform_(self.pe, -0.02, 0.02)
    def forward(self,input_embedding):

        input_embedding = self.linear(input_embedding)

        for i in range(0,self.num_att):
            query = self.layer_q(input_embedding)
            key = self.layer_k(input_embedding)
            value = self.layer_v(input_embedding)

            attn_weight = torch.bmm(query, torch.transpose(key, 1, 2))
            attn_weight = softmax(attn_weight, dim=-1)
            output = torch.bmm(attn_weight, value) + self.layer_add(input_embedding)
            output = self.dropout(output)
            output = output.reshape(output.shape[0], -1)
            output = self.layer_final(output)

        return output





class Transformer_darker(nn.Module):
    def __init__(self, number_att, d_max_len,d_length,d_model,number_shape,number_class, device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')):
        super(Transformer_darker, self).__init__()
        self.d_length = d_length
        self.d_model = d_model
        self.number_shape = number_shape
        self.number_class = number_class
        self.relu = nn.ReLU()
        self.layer_q = nn.Linear(d_length,d_model,bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length,d_model,bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)
        self.one = torch.unsqueeze(torch.eye(d_length),0).to(device)
        self.device = device
        self.num_att = number_att

        self.linear = nn.Linear(d_max_len, d_length)
        self.d_max_len = d_max_len

        self.linear2 = nn.Linear(d_length, d_length)



        self.layer_add = nn.Linear(d_length, d_model)
        self.dropout= nn.Dropout(0.1)
        self.pe = nn.Parameter(torch.empty(1, number_shape, d_length))
        nn.init.uniform_(self.pe, -0.02, 0.02)
    def forward(self,input_embedding,fai_x=None, fai_x_prime=None,
                           w_1=None, b_1=None, w_2=None, b_2=None):
        input_embedding = self.linear(input_embedding)
        batch_size = input_embedding.shape[0]
        num_embed = input_embedding.shape[1]
        d_length = input_embedding.shape[2]

        if fai_x is None:
            fai_x = torch.ones(batch_size, num_embed,self.d_length).to(self.device)
            fai_x_2 = torch.ones(batch_size, 30 * num_embed, self.d_length).to(self.device)
            fai_x_prime = torch.ones(batch_size,self.d_length,num_embed).to(self.device)
            w_1 = torch.ones(batch_size,self.d_length,512).to(self.device)
            b_1 =torch.ones(batch_size,self.d_length,512).to(self.device)
            w_2 =torch.ones(batch_size,512,self.d_length).to(self.device)
            b_2 =torch.ones(batch_size,self.d_length,self.d_length).to(self.device)

        for i in range(0, self.num_att):
            value = self.layer_v(input_embedding)
            # output = torch.bmm(fai_x_prime,value)

            # Attn = X * (W_Q * W_K.T) * X.T * V
            # Attn_learn = fai(X) * fai(W_Q * W_K.T) * fai(X.T) * V
            w_in = self.linear2(fai_x_2)


            # get W_Q from the layer Q
            w_q = self.layer_q(((self.one).expand(input_embedding.shape[0], self.d_length, self.d_length)))
            # get W_K from the layer K
            w_k = self.layer_k(((self.one).expand(input_embedding.shape[0], self.d_length, self.d_length)))
            # W_in = W_Q * W_K.T
            w_in = torch.bmm(w_q, torch.transpose(w_k, 1, 2))

            w_out = torch.bmm(w_in, w_1) + b_1
            w_out = self.relu(w_out)
            w_out = torch.bmm(w_out, w_2) + b_2
            w_out = self.layer_norm(w_out)

            # output = fai(X.T) * V
            output = torch.bmm(fai_x_prime, value)
            output = torch.bmm(w_out, output)
            output = torch.bmm(fai_x, output)

            output = output + self.layer_add(input_embedding)

            output = self.dropout(output)

            output = output.reshape(output.shape[0], -1)
            output = self.layer_final(output)
        return output







class Transformer_performer(nn.Module):
    def __init__(self, number_att,d_max_len,d_length,d_model,number_shape,number_class, m, device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')):
        super(Transformer_performer, self).__init__()
        self.d_length = d_length
        self.d_model = d_model
        self.number_shape = number_shape
        self.number_class = number_class
        self.relu = nn.ReLU()

        self.num_att = number_att
        self.linear = nn.Linear(d_max_len, d_length)

        self.layer_q = nn.Linear(d_length,d_model,bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length,d_model,bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)
        self.one = torch.unsqueeze(torch.eye(d_length),0).to(device)

        self.layer_add = nn.Linear(d_length, d_model)
        self.pe = nn.Parameter(torch.empty(1, number_shape, d_length))
        nn.init.uniform_(self.pe, -0.02, 0.02)
        self.dropout = nn.Dropout(0.1)
        ## addtional parameters for performer
        self.m = m
        self.device =device



    def forward(self,input_embedding, normalize=False):
        input_embedding = self.linear(input_embedding)

        for i in range(0,self.num_att):
            query = self.layer_q(input_embedding)
            key = self.layer_k(input_embedding)
            value = self.layer_v(input_embedding)

            batch_size, l, d = query.shape
            normalizer = 1 / (d ** 0.25) if normalize else 1

            m = self.m

            random_feats = orthogonal_gaussian(m, d)
            random_feats = torch.unsqueeze(random_feats, dim=0).expand(batch_size, m, d).to(self.device)

            def h(x):
                return torch.exp(-torch.square(x).sum(dim=-1, keepdim=True) / 2)

            phi = phi_postive(h, m, random_feats)

            query_prime = phi(query * normalizer)
            key_prime = phi(key * normalizer)

            d_inv = torch.squeeze((torch.matmul(query_prime, torch.matmul(key_prime.transpose(1, 2),
                                                                          torch.ones((batch_size, l, 1)).to(
                                                                              self.device)))),
                                  -1)
            d_inv = torch.diag_embed(1 / d_inv)

            output = torch.matmul(key_prime.transpose(1, 2), value)
            output = torch.matmul(query_prime, output)
            output = torch.matmul(d_inv, output)

            output = output + self.layer_add(input_embedding)
            output = self.dropout(output)
            output = output.reshape(output.shape[0], -1)
            output = self.layer_final(output)


        return output

#Others class are for comparing the attention weights with transformer base



class ProbAttention(nn.Module):
    def __init__(self, number_att, d_max_len,d_length,d_model,number_shape,number_class, device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu'),
                 mask_flag=False, factor=20, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.d_max_len = d_max_len
        self.d_length = d_length
        self.d_model = d_model
        self.number_shape = number_shape
        self.number_class = number_class
        self.relu = nn.ReLU()
        # self.layer_q = nn.Linear(d_length,d_model,bias=False)
        # self.layer_k = nn.Linear(d_length,d_model,bias=False)
        # w_q = torch.ones((d_length,d_model), requires_grad= True).to(device)
        # self.w_q = torch.nn.parameter(w_q)
        # w_k = torch.ones((d_length, d_length), requires_grad=True).to(device)
        # self.w_k = Parameter(w_k)

        self.num_att = number_att
        self.linear = nn.Linear(d_max_len, d_length)


        self.layer_q = nn.Linear(d_length, d_model, bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length, d_model, bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)

        self.layer_add = nn.Linear(d_length, d_model)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape


        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self,input_embedding):
        input_embedding = self.linear(input_embedding)

        for i in range(0,self.num_att):
            B = input_embedding.shape[0]
            N = input_embedding.shape[1]
            h = 1
            queries = self.layer_q(input_embedding)  # batch * N * d
            keys = self.layer_k(input_embedding)  #
            values = self.layer_v(input_embedding)  #

            queries = queries.view(B, N, h, -1)
            keys = keys.view(B, N, h, -1)
            values = values.view(B, N, h, -1)

            B, L_Q, H, D = queries.shape
            _, L_K, _, _ = keys.shape

            queries = queries.transpose(2, 1)
            keys = keys.transpose(2, 1)
            values = values.transpose(2, 1)

            attn_mask = None  # assume

            U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
            u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

            U_part = U_part if U_part < L_K else L_K
            u = u if u < L_Q else L_Q

            scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

            # add scale factor
            scale = self.scale or 1. / sqrt(D)
            if scale is not None:
                scores_top = scores_top * scale
            # get the context
            context = self._get_initial_context(values, L_Q)
            # update the context with selected top_k queries
            context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

            context = context.transpose(2, 1).contiguous()
            context = context.view(B, N, -1)

            context = context + self.layer_add(input_embedding)

            context = self.dropout(context)

            context = context.reshape(context.shape[0], -1)
            context = self.layer_final(context)

        return context




def phi_(phi_c, x, lb = 0, ub = 1):
    mask = np.logical_or(x<lb, x>ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)

def legendreDer(k, x):
    def _legendre(k, x):
        return (2*k+1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k-1,-1,-2):
        out += _legendre(i, x)
    return out

def get_phi_psi(k, base):
    x = Symbol('x')
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
            phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
            phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                a = phi_2x_coeff[ki, :ki + 1]
                b = phi_coeff[i, :i + 1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                a = phi_2x_coeff[ki, :ki + 1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            a = psi1_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm1 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()

            a = psi2_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm2 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * (1 - np.power(0.5, 1 + np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]

    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi)
                phi_2x_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2 * x - 1), x).all_coeffs()
                phi_coeff[ki, :ki + 1] = np.flip(2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4 * x - 1), x).all_coeffs()
                phi_2x_coeff[ki, :ki + 1] = np.flip(
                    np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))

        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]

        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5 + 1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1)

    return phi, psi1, psi2

def get_filter(base, k):
    def psi(psi1, psi2, i, inp):
        mask = (inp <= 0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1 - mask)

    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')

    x = Symbol('x')
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    G0 = np.zeros((k, k))
    G1 = np.zeros((k, k))
    PHI0 = np.zeros((k, k))
    PHI1 = np.zeros((k, k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1 / k / legendreDer(k, 2 * x_m - 1) / eval_legendre(k - 1, 2 * x_m - 1)

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

        PHI0 = np.eye(k)
        PHI1 = np.eye(k)

    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

                PHI0[ki, kpi] = (wm * phi[ki](2 * x_m) * phi[kpi](2 * x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2 * x_m - 1) * phi[kpi](2 * x_m - 1)).sum() * 2

        PHI0[np.abs(PHI0) < 1e-8] = 0
        PHI1[np.abs(PHI1) < 1e-8] = 0

    H0[np.abs(H0) < 1e-8] = 0
    H1[np.abs(H1) < 1e-8] = 0
    G0[np.abs(G0) < 1e-8] = 0
    G1[np.abs(G1) < 1e-8] = 0

    return H0, H1, G0, G1, PHI0, PHI1

class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        # l = N//2+1
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x

class MWT_CZ1d(nn.Module):
    def __init__(self,
                 k=3, alpha=64,
                 L=0, c=1,
                 base='legendre',
                 initializer=None,
                 **kwargs):
        super(MWT_CZ1d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)

        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, k)
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0:nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        #         decompose
        for i in range(ns - self.L):
            # print('x shape',x.shape)
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)  # coarsest scale transform

        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :N, :, :]

        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x




class MultiWaveletTransform(nn.Module):
    """
    1D multiwavelet block.
    """

    def __init__(self, ich=1, k=8, alpha=16, c=128,
                 nCZ=1, L=0, base='legendre', attention_dropout=0.1):
        super(MultiWaveletTransform, self).__init__()
        print('base', base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        values = values.view(B, L, -1)

        V = self.Lk0(values).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)

        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        return (V.contiguous(), None)


# cross
class MultiWaveletCross(nn.Module):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64,
                 k=8, ich=512,
                 L=0,
                 base='legendre',
                 mode_select_method='random',
                 initializer=None, activation='tanh',
                 **kwargs):
        super(MultiWaveletCross, self).__init__()
        print('base', base)

        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.attn1 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn2 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn3 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn4 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape  # (B, N, H, E) torch.Size([3, 768, 8, 2])
        _, S, _, _ = k.shape  # (B, S, H, E) torch.Size([3, 96, 8, 2])

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            # print('q shape',q.shape)
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [self.attn1(dq[0], dk[0], dv[0], mask)[0] + self.attn2(dq[1], dk[1], dv[1], mask)[0]]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return (v.contiguous(), None)

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


#

class FourierCrossAttentionW(nn.Module):
    def __init__(self, number_att, d_max_len, d_length,d_model,number_shape,number_class,attention_dropout=0.1, modes=16, activation='tanh',
                 mode_select_method='random'):
        super(FourierCrossAttentionW, self).__init__()
        print('corss fourier correlation used!')
        self.modes1 = modes
        self.activation = activation
        self.d_model = d_model
        self.d_length = d_length

        self.linear = nn.Linear(d_max_len, d_length)
        self.num_att = number_att

        self.layer_q = nn.Linear(d_length, d_model, bias=False)
        self.layer_k = nn.Linear(d_length, d_model, bias=False)
        self.layer_v = nn.Linear(d_length, d_model, bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_length, d_length)
        self.act = F.gelu
        self.layer_final = nn.Linear(d_model * number_shape, number_class)

        self.layer_add = nn.Linear(d_length, d_model)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, input_embedding):

        input_embedding = self.linear(input_embedding)



        B = input_embedding.shape[0]
        N = input_embedding.shape[1]
        h = 1
        queries = self.layer_q(input_embedding)          # batch * N * d
        keys = self.layer_k(input_embedding)            #
        values = self.layer_v(input_embedding)          #

        queries = queries.view(B, N, h, -1)
        keys = keys.view(B, N, h, -1)
        values = values.view(B, N, h, -1)

        q = queries.permute(0, 1, 3, 2)
        k = keys.permute(0, 1, 3, 2)
        v = values.permute(0, 1, 3, 2)
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # size = [B, H, E, L] torch.Size([3, 8, 64, 512])
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)

        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(out_ft / self.d_model / self.d_model, n=xq.size(-1)).permute(0, 3, 2, 1)
        # size = [B, L, H, E]
        out = out.permute(0, 1, 3, 2)
        out = out.view(B, N, -1)

        out = out + self.layer_add(input_embedding)

        out = self.dropout(out)

        out = out.reshape(out.shape[0], -1)
        out = self.layer_final(out)

        return out