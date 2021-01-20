
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
# https://cloud.tencent.com/developer/article/1042161

def testrow(input, output, src_lang, dst_lang, encoder, decoder, beam_width=2, print_flag=False):
    input, input_len = input
    if PAD_token in input[:input_len] or PAD_token in output or len(input) > flags.seq_size or len(output) > flags.seq_size:
        return None, None, -1
    encoder.eval() # set in evaluation mode
    decoder.eval()

    x = torch.tensor(input).to(device).reshape(1,-1)
    seq_len = torch.tensor([input_len]).to(torch.int64).to(device)
    # encoder
    encoder_ht, encoder_ct = encoder.initHidden(1)
    encoder_outputs, (encoder_ht, encoder_ct) = encoder(x, (encoder_ht, encoder_ct), seq_len)

    decoder_input = torch.tensor([BOS_token] * 1).reshape(1,1).to(device) # <BOS> token
    decoder_ht, decoder_ct = encoder_ht, encoder_ct # use last hidden state from encoder

    # decoder
    # run through decoder one time step at a time
    max_len = int(flags.seq_size*1.5)
    decoder_attentions = torch.zeros(max_len,flags.seq_size)
    path = [(BOS_token, 0, [])]  # input, value, words on the path
    for t in range(max_len):
        new_path = []
        flag_done = True
        for decoder_input, value, indices in path:
            if decoder_input == EOS_token:
                new_path.append((decoder_input, value, indices))
                continue
            elif len(path) != 1 and decoder_input in [BOS_token, PAD_token]:
                continue
            flag_done = False
            decoder_input = torch.tensor([decoder_input]).reshape(1, 1).to(device)

            decoder_output, (decoder_ht, decoder_ct), decoder_attn = decoder(decoder_input,
                                                                                 (decoder_ht, decoder_ct),
                                                                                 encoder_outputs)
            decoder_attentions[t] = decoder_attn.transpose(1, 2).cpu().data

            softmax_output = F.log_softmax(decoder_output, dim=2)
            top_value, top_index = softmax_output.data.topk(beam_width)
            top_value = top_value.cpu().squeeze().numpy() + value
            top_index = top_index.cpu().squeeze().numpy()
            for i in range(beam_width):
                ni = int(top_index[i])
                new_path.append((ni, top_value[i], indices + [ni]))
        if flag_done:
            _, value, decoded_index = new_path[0]
            break
        else:
            new_path.sort(key=lambda x: x[1] / len(x[2]), reverse=True)  # normalization
            path = new_path[:beam_width]

    if not flag_done:
        _, value, decoded_index = path[0]
    decoded_words = []
    for ni in decoded_index:
        word = dst_lang.index2word[ni]
        decoded_words.append(word)


    pad_index = np.where(output == PAD_token)
    if len(pad_index[0]) == 0:
        pad_index = len(output)
    else:
        pad_index = pad_index[0][0]
    filter_outtext = list(filter("<PAD>".__ne__,output[:pad_index]))
    decoded_index = list(filter("<PAD>".__ne__,decoded_index))
    sm = SmoothingFunction()
    bleu = sentence_bleu([filter_outtext],decoded_index,smoothing_function=sm.method4)
    print(output[:pad_index])
    print(decoded_index)
    print("Bleu score: {}".format(bleu))
    res_words = " ".join(decoded_words)
    print("< {}".format(src_lang.getSentenceFromIndex(input)))
    print("= {}".format(dst_lang.getSentenceFromIndex(filter_outtext)))
    print("> {}".format(res_words))

    return decoded_words, decoder_attentions[:t+1, :flags.seq_size], bleu


def evaluation(dataset, src_lang, dst_lang, encoder, decoder, beam_search=False, beam_width=2):
    start_time = time.time()
    bleus = []
    for i,(input, output) in enumerate(dataset):
        input = src_lang.getSentenceIndex(input,0,False)
        input_len = len(input)
        input = src_lang.padIndex(input,flags.seq_size)
        if len(input) == 0:
            continue
        output = dst_lang.getSentenceIndex(output,0,False)
        res_words, attention, bleu = testrow((input,input_len),output,src_lang,dst_lang,encoder,decoder,beam_width=beam_width)
        if res_words != None:
            bleus.append(bleu)
    avg_bleu = np.mean(bleus)
    return avg_bleu