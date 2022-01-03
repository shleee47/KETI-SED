import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer,'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class finetunePANNs(nn.Module):
    def __init__(self,PANNs_pretrain,class_num):
        super(finetunePANNs, self).__init__()
        self.PANNs = PANNs_pretrain
#        self.add_fc1 = nn.Linear(2048,1024, bias=True)
#        self.add_fc2 = nn.Linear(1024,512, bias=True)
#        self.add_fc3 = nn.Linear(512,128, bias=True)
#        self.add_fc4 = nn.Linear(128,class_num, bias=True)
#
#        self.bn0 = nn.BatchNorm1d(1024)
#        self.bn1 = nn.BatchNorm1d(512)
#        self.bn2 = nn.BatchNorm1d(128)

        self.add_fc1 = nn.Linear(527,class_num, bias=True)
        
        self.init_weights()

    def init_weights(self):
        #init_bn(self.bn0)
        #init_bn(self.bn1)
        #init_bn(self.bn2)
        init_layer(self.add_fc1)
        #init_layer(self.add_fc2)
        #init_layer(self.add_fc3)
        #init_layer(self.add_fc4)


    def forward(self, input):
        #pdb.set_trace()
        x=  self.PANNs(input)
        embed = x['embedding']

        #x = F.relu_(self.bn0(self.add_fc1(embed)))
        #x = F.relu_(self.bn1(self.add_fc2(x)))
        #embedding = F.relu_(self.bn2(self.add_fc3(x)))
        #clipwise_output = torch.sigmoid(self.add_fc4(embedding))
        
        clipwise_output = torch.sigmoid(self.add_fc1(embed))

        output_dict = {'clipwise_output': clipwise_output}
        #output_dict = {'clipwise_output': clipwise_output, 'embedding' : embedding}

        return output_dict



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x
