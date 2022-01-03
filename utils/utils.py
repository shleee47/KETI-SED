from collections import namedtuple
import torch
from torch import nn
from utils.resnet import resnet18, resnet50, resnet101
import pdb

Encoder = namedtuple('Encoder', ('model', 'features', 'features_shape'))


def make_encoder(name, input_size=224, input_channels=3, pretrained=True, pretrain_path=None):
    """Make encoder (backbone) with a given name and parameters"""
    
    features_size = input_size // 32
    num_features = 2048
    # if name.startswith('resnet'):
    if name == 'resnet50':
        model = resnet50(pretrained=True)
        features = nn.Sequential(*list(model.children())[:-2])
        # features[0] = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = 512 if int(name[6:]) < 50 else 2048
        
        features_shape = (num_features, features_size, features_size)
        return Encoder(model, features, features_shape)

    elif name == 'resnet101':
        model = resnet101(pretrained=True)
        features = nn.Sequential(*list(model.children())[:-2])
        #num_features = 512 if int(name[6:]) < 50 else 2048
        #features_shape = (num_features, features_size, features_size)
        return features#Encoder(model, features, features_shape) 


    elif name == 'resnet18':
        print('resnet18')
        model = resnet18(pretrained=True)
        features = nn.Sequential(*list(model.children())[:-3])
        # features_shape = (num_features, features_size, features_size)
        return features #Encoder(model, features, features_shape) 


    else:
        raise KeyError("Unknown model name: {}".format(name))


    

def load_from_pretrainedmodels(model_name):
    import pretrainedmodels
    return getattr(pretrainedmodels, model_name)



def squash_dims(tensor, dims):
    assert len(dims) >= 2, "Expected two or more dims to be squashed"

    size = tensor.size()

    squashed_dim = size[dims[0]]
    for i in range(1, len(dims)):
        assert dims[i] == dims[i - 1] + 1, "Squashed dims should be consecutive"
        squashed_dim *= size[dims[i]]

    result_dims = size[:dims[0]] + (squashed_dim,) + size[dims[-1] + 1:]
    return tensor.contiguous().view(*result_dims)


def unsquash_dim(tensor, dim, res_dim):
    size = tensor.size()
    result_dim = size[:dim] + res_dim + size[dim + 1:]
    return tensor.view(*result_dim)


def intermediate_at_measures(encoded_ref,encoded_est):
    tp = (encoded_est + encoded_ref == 2).sum(axis=0)
    fp = (encoded_est - encoded_ref == 1).sum(axis=0)
    fn = (encoded_ref - encoded_est == 1).sum(axis=0)
    tn = (encoded_est + encoded_ref == 0).sum(axis=0)
    return tp,fp,fn,tn


