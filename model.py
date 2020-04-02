import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import data_loader


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        # embedding_size will be size of 'features' and input to Decoder
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # vocab_size will be length of list of all words in captions PLUS 2 for <start> and <end>
        self.vocab_size = vocab_size
        # tokenize captions by converting words in vocab to tensor of embedding_size
        """
        definition of embedding layer = transform each words in a caption into a vector of a desired, consistent shape.
            - the desired, consistent shape is, of course, the embedding_size
        """
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)
        #self.init_weights()
    
    def forward(self, features, captions):
        # DecoderRNN trained on captions from COCO dataset
        # reshape caption length from 15 to 14 because do not INPUT <end> keyword
        captions = captions[:, :-1]
        # convert captions into list of tokenized words (list of integers)
        embeddings = self.embed(captions)
        # concatenate features + captions = embedding layer input
        embeddings = torch.cat((features.unsqueeze(dim = 1), embeddings), dim = 1)
        
        hiddens, states = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        # outputs is a distribution of most likely next word
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # length of iteration is max_len=20 -- this is length of output sentence
        # input are features output from CNN
        # input features run through LSTM --> create NEW input
        # States are second input for self.lstm(inputs, states)        
        new_list = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            score, index = outputs.max(1)
 
            new_list.append(index.item())
            inputs = self.embed(index)
            inputs = inputs.unsqueeze(1)
        return new_list
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1

        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)