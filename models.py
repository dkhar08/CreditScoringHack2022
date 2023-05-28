import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence

from typing import *


#I use dropconnection like in AWD LSTM for RNN regularization. 
#https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            print(x)
            x, batch_sizes, si, usi = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes, si, usi)
        else:
            return x

class LSTMb(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state
    

def dropout_mask(x, size, prob):
     """
     We pass size in so that we get broadcasting along the sequence dimension
     in RNNDropout.
     """
     return x.new(*size).bernoulli_(1-prob).div_(1-prob)

class RNNDropout(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x
        mask = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.prob)
        return x * mask


#from baseline
def compute_embed_dim(n_cat: int) -> int:
    return min(600, 2*round(1.6 * n_cat**0.56))

#Autoencoder. I use the same lstm for encoder and decoder part        
class LM(nn.Module):
    def __init__(self, id2feat, pad_values):
        super(LM, self).__init__()
        
        lstm_size = 812
        hidden_size = 400
        max_pos_len  = 50
        
        number_of_features = len(id2feat)
        
        self.emb_sizes = [compute_embed_dim(pad_values[id2feat[i]]-3) for i in range(number_of_features)]
        print("embedding size", self.emb_sizes)
        self.number_of_features = number_of_features
        self.max_pos_len = max_pos_len
              
        self.embedding_layers = nn.ModuleList([nn.Embedding(pad_values[id2feat[i]], compute_embed_dim(pad_values[id2feat[i]]-3)) for i in range(number_of_features)])
        
        self.lstm = LSTMb(input_size=sum(self.emb_sizes), hidden_size=lstm_size, num_layers=1, dropoutw=0.2, batch_first=False)
        
        self.linears = nn.ModuleList([torch.nn.Linear(lstm_size, pad_values[id2feat[i]], bias=True) for i in range(len(self.emb_sizes))])
        
        self.sm = nn.LogSoftmax(dim=2)
                
    def forward(self, list_of_tensor_features, lens):
        rezults = []
        for i in range(self.number_of_features):
            x = list_of_tensor_features[i]
            
            x = self.embedding_layers[i](x)
            
            rezults.append(x)
        
        x = torch.cat(rezults, axis=2)
        x = x.transpose(0, 1)
        
        x_enc = pack_padded_sequence(x, lens, enforce_sorted=False)
        
        
        output, (hn, cn) = self.lstm(x_enc)
        output, (hn, cn) = self.lstm(x, (hn, cn))
        
        return [self.sm(self.linears[i](output)).transpose(0,1).transpose(1,2) for i in range(self.number_of_features)] #torch.sigmoid(x)


#classifier           
class Net(nn.Module):
    def __init__(self, id2feat, pad_values):
        super(Net, self).__init__()
        lstm_size = 812
        hidden_size = 400
        max_pos_len  = 50
        number_of_features = len(id2feat)
        
        self.emb_sizes = [compute_embed_dim(pad_values[id2feat[i]]-3) for i in range(number_of_features)]
        print("embedding size", self.emb_sizes)
        self.number_of_features = number_of_features
        self.max_pos_len = max_pos_len
              
        self.embedding_layers = nn.ModuleList([nn.Embedding(pad_values[id2feat[i]], compute_embed_dim(pad_values[id2feat[i]]-3)) for i in range(number_of_features)])
        
        self.lstm = LSTMb(input_size=sum(self.emb_sizes), hidden_size=lstm_size, num_layers=1, dropoutw=0.2, batch_first=False)
        
        
        self.linear1 = torch.nn.Linear(lstm_size, hidden_size, bias=True)
        self.linear3 = torch.nn.Linear(hidden_size, 2, bias=True)
        
        self.sm = nn.LogSoftmax(dim=1)
                
    def forward(self, list_of_tensor_features, lens):
        rezults = []
        for i in range(self.number_of_features):
            x = list_of_tensor_features[i]
            
            x = self.embedding_layers[i](x)
            
            rezults.append(x)
        
        x = torch.cat(rezults, axis=2)
        x = x.transpose(0, 1)
        x = pack_padded_sequence(x, lens, enforce_sorted=False)
        
        output, (hn, cn) = self.lstm(x)
        
        x = self.linear1(cn[0,:,:])
        
        x = nn.functional.relu(x)
        
        x = self.linear3(x)
        
        return self.sm(x) 

