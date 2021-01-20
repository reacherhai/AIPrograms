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

import time
import math

from model import *
from data_prepare import *
from plot import showPlot


device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asMinutes(s):
    m =  math.floor(s/60)
    s -=  m*60
    return "%dm %ds" %(m,s)

def timesince(since,percent):
    now = time.time()
    s = now - since
    es = s/ percent
    rs = es - s
    return "%s (- %s)" % (asMinutes(s),asMinutes(rs))

#train

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length = MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device = device)

    loss = 0

    for ei in range(input_length):
        encoder_output, (encoder_ht,encoder_ct) = encoder(input_tensor[ei], (encoder_ht,encoder_ct),input_length)
        encoder_outputs[ei] = encoder_output[0,0]

    decoder_input = torch.tensor([], device = device)
    decoder_ht, decoder_ct = encoder_ht,encoder_ct

    decoder_hidden = (decoder_ht,decoder_ct)

    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if(use_teacher_forcing):
        #teacher forcing: feed the target as the next input, else use net's own output
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention  = \
                decoder(decoder_input,decoder_hidden,encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] #detach from history as input
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention  = \
                decoder(decoder_input,decoder_hidden,encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach() #detach from history as input

            loss += criterion(decoder_output,target_tensor[di])

            if(decoder_input.item() == EOS_token):
                break
    loss.backword()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

'''
now, start training as following steps:
Start a timer
Initialize optimizers and criterion
Create set of training pairs
Start empty losses array for plotting
'''


def trainIters(encoder, decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(),lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(),lr = learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    #ignore the PAD
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_token)

    train_loader = data.DataLoader(dataset=train_set, batch_size= batch_size, shuffle=True)

    loss = criterion

    print("Begin training")

    for iter in range(n_iters):
        encoder_ht, encoder_ct = encoder.initHidden(batch_size)
        decoder_ht, decoder_ct = decoder.initHidden(batch_size)

        #train_step:
        for step,(input_tensor,target_tensor,lenx,leny) in enumerate(train_loader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_length = lenx
            target_length = leny

            seq_lengths, idx = torch.tensor(lenx).sort(0, descending=True)

            input_tensor = torch.tensor(input_tensor).to(torch.int64).to(device) # (batch_size, seq_size)
            target_tensor = torch.tensor(target_tensor).to(torch.int64).to(device) # (batch_size, seq_size)
            input_tensor = input_tensor.reshape(batch_size,-1)
            target_tensor = target_tensor.reshape(batch_size,-1)
            print (input_tensor.shape)
            print(target_tensor.shape)

            #print (input_tensor.shape)  torch.Size([32, 81]) batch_size,seq_size
            #print (target_tensor.shape) torch.Size([32, 94]) batch_size,seq_size
            encoder_outputs, (encoder_ht, encoder_ct) = encoder(input_tensor, (encoder_ht, encoder_ct), seq_lengths)

            decoder_input = torch.tensor([BOS_token] * batch_size).reshape(batch_size, 1).to(device)  # <BOS>
            decoder_ht, decoder_ct = encoder_ht, encoder_ct

            decoder_hidden = (decoder_ht, decoder_ct)

            maxlen = target_tensor.shape[1]
            all_decoder_outputs = torch.zeros((maxlen, batch_size, decoder.output_size))

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if (use_teacher_forcing):
                # teacher forcing: feed the target as the next input, else use net's own output
                for di in range(maxlen):
                    decoder_output, decoder_hidden, decoder_attention = \
                        decoder(decoder_input, decoder_hidden, encoder_outputs)
                    decoder_input = target_tensor[:,di].reshape(batch_size,1)  # detach from teacher as input
                    all_decoder_outputs[di] = decoder_output.transpose(1, 0)
            else:
                for di in range(maxlen):
                    decoder_output, decoder_hidden, decoder_attention = \
                        decoder(decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input
                    all_decoder_outputs[di] = decoder_output.transpose(1, 0)

            loss_f = criterion(all_decoder_outputs.permute(1, 2, 0).to(device).to(device), target_tensor)
            loss = loss_f.item()

            loss.backword()

            encoder_optimizer.step()
            decoder_optimizer.step()

        print_loss_total += loss
        plot_loss_total += loss

        if(iter % print_every == 0 ):
            print_loss_avg = print_loss_total / print_every
            print_loss_total  = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)

#train:

print ("checkpoint saved")
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words,hidden_size).to(device)
atten_decoder = AttenDecoderRNN(hidden_size,output_lang.n_words,dropout_p=0.1).to(device)
trainIters(encoder, atten_decoder, 70000, print_every = 50)

torch.save(encoder,'{}/encoder-final.pth'.format("model"))
torch.save(decoder,'{}/decoder-final.pth'.format("model"))