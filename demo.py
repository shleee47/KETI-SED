import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
import argparse
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import pdb
import yaml 
import numpy as np
from torch.utils.data import DataLoader
import os
import librosa
import pickle
from pathlib import Path
from model import pretrained_Gated_CRNN8
from PANNs import ResNet38
from fintune import finetunePANNs, Identity

class ModelTester:
    def __init__(self, model, test_loader, ckpt_path, device, target_class):
        #pdb.set_trace()
        self.device = torch.device('cuda:{}'.format(device))
        #self.model = model.to(self.device)
        self.model = model.cuda()
        self.test_loader = test_loader
        self.target_class = target_class
        
        '''Load Trained Model'''
        self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt):
        ckpt = ckpt[self.target_class.lower()]
        print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)


    def test(self):

        self.model.eval()
        output_result = {} 
        batch_size = len(self.test_loader)
        audio_correct_list = []
        cor = 0
        non = 0
        with torch.no_grad():
            for b, batch in enumerate(self.test_loader):
                #pdb.set_trace() 
                inputs, audio_path = batch
                #inputs = torch.unsqueeze(inputs,1)
                #B, C, T, Freq = inputs.size()  
                try:
                    B, T = inputs.size()  
                except:
                    print('No Audio Signal Detected in {}'.format(audio_path[0].split('/')[-1]))
                    continue

                inputs = inputs.cuda()
                if inputs.dtype != torch.float32:
                    inputs = inputs.float()
                outputs = self.model(inputs)
                outputs = outputs['clipwise_output']
                #best_prediction = outputs.max(2)[1].mode()[0]
                best_prediction = outputs.max(1)[1].mode()[0]

                # print(best_prediction.item(),end = ' ') 
                if str(best_prediction.item()) not in output_result.keys():
                    output_result[str(best_prediction.item())] = []
                #     output_result[str(best_prediction.item())] = 0
                #else:
                #    output_result[str(best_prediction.item())] +=1
                output_result[str(best_prediction.item())].append(audio_path[0].split('/')[-1])
                #print(outputs)
                if best_prediction.item() == 0:
                    audio_correct_list.append(audio_path)
                    print('prediction result of {}: {}'.format(audio_path[0].split('/')[-1],self.target_class))
                    cor +=1
                else:
                    print('prediction result of {}: Not-{}'.format(audio_path[0].split('/')[-1],self.target_class))
                    non +=1
        print('correct: {}'.format(cor))
        print('non-correct: {}'.format(non))
        print('audio_correct_list: {}'.format(audio_correct_list))
        pdb.set_trace()
        # print('number of NaN data: {}'.format(self.test_loader.dataset.nan_cnt))
        # #del(output_result['0'])
        # print('output_result: {}'.format(output_result))


class Data_Reader(Dataset):
    def __init__(self, datalist):
        super(Data_Reader, self).__init__()
        self.datalist = datalist
        self.nfft = 512
        self.hopsize = self.nfft // 4
        self.window = 'hann'
        self.nan_cnt = 0

    def __len__(self):
        print('Number of wav data to inference: {}'.format(len(self.datalist)))
        return len(self.datalist)

    def LogMelExtractor(self, sig):
        def logmel(sig):
            D = np.abs(librosa.stft(y=sig,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2        
            SS = librosa.feature.melspectrogram(S=D,sr=16000)
            S = librosa.amplitude_to_db(SS)
            #S[:10, :] = 0.0
            return S

        def transform(audio):
            feature_logmel = logmel(audio)
            return feature_logmel
        
        return transform(sig)

    def __getitem__(self, idx):
        audio_path = self.datalist[idx]
        try:
            audio, _ = librosa.load(audio_path, sr=32000, dtype=np.float32)
        except:
            self.nan_cnt +=1
            print('No Audio Signal Detected in {}'.format(audio_path.split('/')[-1]))
            return torch.zeros(1,1,1), audio_path.split('/')[-1]
        
        if audio.sum() == 0.0:
            self.nan_cnt +=1
            print('No Audio Signal Detected in {}'.format(audio_path.split('/')[-1]))
            return torch.zeros(1,1,1), audio_path.split('/')[-1]

        else:
            #feature = self.LogMelExtractor(audio)
            #return torch.FloatTensor(feature).transpose(0,1), np.array([class_num])
            return audio, audio_path.split('/')[-1]


def inference(config,args):
    '''1. Dataset Preparation'''
    test_csv_path = config['datasets']['demo']
    test_list = np.loadtxt(os.path.join(test_csv_path,"test.csv"),str)
    test_dataset = Data_Reader(test_list)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory = True, num_workers=0)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, pin_memory = True, num_workers=10)

    '''2. Model'''
    PANNs_model = ResNet38(32000, 1024, 1024//4,64, config['MYNET']['n_classes'],True)
    SED_model = finetunePANNs(PANNs_model,config['MYNET']['n_classes'])
    
    '''3. Tester'''
    tester = ModelTester(SED_model, test_loader, config['demo'], config['device'], args.version)
    tester.test()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.', help='Root directory')
    parser.add_argument('-c', '--config', type=str, help='Path to option YAML file.')
    parser.add_argument('-d', '--dataset', type=str, default='config', help='configuration file')
    parser.add_argument('-v', '--version', type=str, default='Boar', help='which event to detect')
    args = parser.parse_args()

    '''Load Config'''
    with open(os.path.join(args.config, args.dataset + '.yml'), mode='r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    '''Inference Mode'''
    print("Mode : Inference Starts")
    inference(config,args)
