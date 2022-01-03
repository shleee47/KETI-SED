import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
import argparse
import torch
import torch.nn as nn
import pdb
import yaml 
import numpy as np
from torch.utils.data import DataLoader
import os
import pickle
from pathlib import Path
from trainer import ModelTrainer, ModelTester
from model import pretrained_Gated_CRNN8
from PANNs import ResNet38, MobileNetV2
from fintune import finetunePANNs, Identity
from utils.setup import setup_solver
from utils.loss import create_criterion
from dataloader import Audio_Reader, Audio_Collate,Test_Reader


def tr_val_split(data_path):
    yes_list = np.loadtxt(os.path.join(data_path,"yes.csv"),str)
    no_list = np.loadtxt(os.path.join(data_path,"no.csv"),str)
    yes_train_list =[]
    yes_val_list = []
    no_train_list =[]
    no_val_list = []
    seed = 5
    np.random.seed(seed)
    split_ratio = 0.9

    '''Split yes dataset'''
    data_len = len(yes_list)
    idx = np.arange(data_len)
    np.random.shuffle(idx)
    split_idx = int(split_ratio*data_len)
    yes_train_list.append(yes_list[idx[:split_idx]])
    yes_val_list.append(yes_list[idx[split_idx:]])
    
    '''Split no dataset'''
    data_len = len(no_list)
    idx = np.arange(data_len)
    np.random.shuffle(idx)
    split_idx = int(split_ratio*data_len)
    no_train_list.append(no_list[idx[:split_idx]])
    no_val_list.append(no_list[idx[split_idx:]])

    return np.squeeze(np.array(yes_train_list),axis=0), np.squeeze(np.array(yes_val_list),axis=0), np.squeeze(np.array(no_train_list),axis=0), np.squeeze(np.array(no_val_list),axis=0)

def train(config):
    '''1. Dataset Preparation'''
    yes_train_list, yes_val_list, no_train_list, no_val_list = tr_val_split(config['datasets']['path'])

    '''2. Dataloader'''
    train_dataset = Audio_Reader(yes_train_list, no_train_list)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['dataloader']['train']['batch_size'], shuffle=True, collate_fn=lambda x: Audio_Collate(x), num_workers=config['dataloader']['train']['num_workers'])
    valid_dataset = Audio_Reader(yes_val_list, no_val_list)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config['dataloader']['valid']['batch_size'], shuffle=True, collate_fn=lambda x: Audio_Collate(x), num_workers=config['dataloader']['valid']['num_workers'])
    
    '''3. Model'''
    ## Gated CRNN
    #SED_model = pretrained_Gated_CRNN8(config['MYNET']['n_classes'])

    ## ResNet
    PANNs_model = ResNet38(32000, 1024, 1024//4,64, config['MYNET']['n_classes'],True)
    #pretrained_model = torch.load('/home/sanghoon/fewshot/model/ResNet38_mAP=0.434.pth')

    ## MobileNet
    #PANNs_model = MobileNetV2(32000, 1024, 1024//4,64, 0.0, None, config['MYNET']['n_classes'])
    #pretrained_model = torch.load('/home/sanghoon/fewshot/model/MobileNetV2_mAP=0.383.pth')

    #PANNs_model.load_state_dict(pretrained_model['model'])
    #PANNs_model.fc_audioset = Identity()
    SED_model = finetunePANNs(PANNs_model,config['MYNET']['n_classes'])
    
    '''4. Optimizer / Scheduler'''
    criterion = create_criterion(config['criterion']['name'])
    optimizer, scheduler = setup_solver(SED_model.parameters(), config)
    
    '''5. Trainer'''
    trainer = ModelTrainer(SED_model, train_loader, valid_loader, criterion, optimizer, scheduler, config, **config['trainer'])
    trainer.train()

def test(config):
    '''1. Dataset Preparation'''
    #pdb.set_trace()
    test_csv_path = config['datasets']['test']
    #test_list = np.loadtxt(os.path.join(test_csv_path,"test.csv"),str)
    #test_dataset = Test_Reader(test_list)
    test_list = np.loadtxt(os.path.join(test_csv_path,"test.csv"),str).tolist()
    test_dataset = Test_Reader([test_list])
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory = True, num_workers=40)

    '''2. Model'''
    ## Gated CRNN
    #SED_model = pretrained_Gated_CRNN8(config['MYNET']['n_classes'])

    ## ResNet
    PANNs_model = ResNet38(32000, 1024, 1024//4,64, config['MYNET']['n_classes'],True)

    ## MobileNet
    #PANNs_model = MobileNetV2(32000, 1024, 1024//4,64, 0.0, None, config['MYNET']['n_classes'])
    #pretrained_model = torch.load('/home/sanghoon/fewshot/model/ResNet38_mAP=0.434.pth')

    #PANNs_model.load_state_dict(pretrained_model['model'])
    #PANNs_model.fc_audioset = Identity()
    SED_model = finetunePANNs(PANNs_model,config['MYNET']['n_classes'])
    
    '''3. Tester'''
    tester = ModelTester(SED_model, test_loader, config['tester']['ckpt_path'], config['tester']['device'])
    tester.test()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.', help='Root directory')
    parser.add_argument('-c', '--config', type=str, help='Path to option YAML file.')
    parser.add_argument('-d', '--dataset', type=str, default='config', help='configuration file')
    parser.add_argument('-m', '--mode', type=str, help='Train or Test')
    args = parser.parse_args()

    ## Load Config
    with open(os.path.join(args.config, args.dataset + '.yml'), mode='r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    if args.mode == 'Train':
        print("Mode : Train")
        train(config)

    elif args.mode == 'Test':
        print("Mode : Test")
        test(config)
