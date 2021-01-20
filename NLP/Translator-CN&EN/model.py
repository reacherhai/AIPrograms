from __future__ import unicode_literals, print_function, division
from io import open
import  unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 32


class EncoderRNN(nn.Module):

    # Input: (*), LongTensor of arbitrary shape containing the batch size
    # Output: (*, H), where * is the input shape and H = embedding_dim
    def __init__(self,in_size,hidden_size,dropout = 0.1 ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #in_size -> hidden_size
        self.embedding = nn.Embedding(in_size,hidden_size)
        self.dropout = dropout
        self.lstm = nn.LSTM(hidden_size,hidden_size,dropout = self.dropout, batch_first = True)

    #待修改！
    def forward(self,input, prevState,seq_length):
        embedded = self.embedding(input)
        #tot_length = input.size(1)
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedding, input_len, batch_first = True)
        out,state = self.lstm(embedded, prevState)
        #out, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first = True,total_length = tot_length)
        return out,state

    def initHidden(self):
        return (torch.zeros(1, batch_size,self.hidden_size,device = device),
                torch.zeros(1, batch_size, self.hidden_size, device=device) )


#continue

class DecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size,hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size,batch_first = True)
        self.out = nn.Linear (hidden_size,output_size)
        #self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self,input, prevState):
        embedding = self.embedding(input).view(1,1,-1)
        embedding = F.relu(embedding)
        output,state = self.lstm(embedding, prevState)
        output = self.out(output)
        return output,state

    def initHidden(self):
        return  ( torch.zeros(1, batch_size, self.hidden_size,device = device),
                  torch.zeros(1, batch_size, self.hidden_size, device=device) )

MAX_LENGTH = 50
class AttenDecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size,dropout_p = 0.1, max_length = MAX_LENGTH):
        super(AttenDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size,self.hidden_size)
        self.attn = nn.Linear(self.hidden_size *2, self.max_length)
        self.atten_combine = nn.Linear(self.hidden_size *2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size,self.hidden_size, batch_first = True)
        self.out = nn.Linear (self.hidden_size,self.output_size)

    def forward(self,input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)
        out = torch.cat((embedded,hidden[0].transpose(0,1)), dim = 2)
        out =  self.attn(out).squeeze(1)
        attn_weights  =F.softmax( out, dim =1 )
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),encoder_outputs)

        output = torch.cat( (embedded[0],attn_applied[0]),1 )
        output = self.atten_combine(output).unsqueeze(0)

        output = F.relu(output)
        output,state = self.lstm(output,hidden)
        output = self.out(output)

        return output,hidden,attn_weights

    def initHidden(self):
        return (torch.zeros(1,batch_size,self.hidden_size,device = device),
                torch.zeros(1, batch_size, self.hidden_size, device=device) )

