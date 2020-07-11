from model.pre_process import *
sosToken = 1
eosToken = 0

def predictbygreedysearch(inputSeq,model_name='modelB.pkl',maxAnswerLength=32):
    modelDict = torch.load(model_name,map_location='cpu')
    encoderRNN,decoderRNN = modelDict['encoder'], modelDict['decoder']
    word2id,id2word = modelDict['word2id'], modelDict['id2word']
    hiddenSize = encoderRNN.hiddenSize 
    encoderRNN.eval(), decoderRNN.eval()
    
    inputSeq = filter_sent(inputSeq)
    inputSeq = [w for w in jieba.lcut(inputSeq) if w in word2id.keys()]
    X = seq2id(word2id,inputSeq) + [eosToken]
    XLens = torch.tensor([len(X)],dtype=torch.int) #4
    X = torch.tensor([X],dtype=torch.long) #torch.Size([1, 4]) #encodeInput
    d = int(encoderRNN.bidirectional)+1
    hidden = torch.zeros((d*encoderRNN.numLayers, 1, hiddenSize), dtype=torch.float32)
    encoderOutput, hidden = encoderRNN(X, XLens, hidden)
    hidden = hidden[-d*decoderRNN.numLayers::2].contiguous()

    attentionArrs = []
    Y = []
    decoderInput = torch.tensor([[sosToken]], dtype=torch.long)
    
    while decoderInput.item() != eosToken and len(Y)<maxAnswerLength:  
        decoderOutput, hidden, decoderAttentionWeight = decoderRNN(decoderInput, hidden, encoderOutput)   
        topv, topi = decoderOutput.topk(1)
        decoderInput = topi[:,:,0]
        Y.append(decoderInput.item())
    outputSeq = id2seq(id2word, Y)
    return ''.join(outputSeq[:-1])
