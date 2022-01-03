import os
import sys
import time
import numpy as np
import datetime
import pickle as pkl
from pathlib import Path
import torch
import pdb
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import logging
import json
from multiprocessing import Pool
from utils.utils import intermediate_at_measures
import time
import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:

    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, config, epochs, device, save_path, ckpt_path=None, comment=None, fold=2):
        # Essential parts
        #pdb.set_trace()
        self.device = torch.device('cuda:{}'.format(device))
        #self.model = model.to(self.device)
        self.model = model.cuda()
        #self.model = model
        self.n_class = config["MYNET"]["n_classes"]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp_path = Path(os.path.join(save_path, datetime.now().strftime('%d%B_%0l%0M'))) #21November_0430
        print("=== {} ===".format(self.exp_path))
        self.exp_path.mkdir(exist_ok=True, parents=True)

        # Set logger
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        
        #Dump hyper-parameters
        # config_info = {'optim':str(self.optimizer), 'scheduler':str(self.scheduler), 'criterion':str(self.criterion)}
        with open(str(self.exp_path.joinpath('config.json')), 'w') as f:
            json.dump(config, f, indent=2)

        if comment != None:
            self.logger.info(comment)

        self.writter = SummaryWriter(self.exp_path.joinpath('logs'))
        self.epochs = epochs
        self.best_acc = 0.0
        self.best_epoch = 0
        #pdb.set_trace() 
        if ckpt_path != None:
            self.load_checkpoint(ckpt_path)
            #self.optimizer.param_groups[0]['lr'] = 0.0001

    def train(self):
        #pdb.set_trace() 
        for epoch in tqdm(range(self.epochs)):
            start = time.time()
            self.train_loader.dataset.shuffle_dataset()
            train_loss, t_accuracy, t_f1_score, t_tp, t_fp, t_fn, t_tn  = self.train_single_epoch(epoch)
            valid_loss, v_accuracy, v_f1_score, v_tp, v_fp, v_fn, v_tn = self.inference()
            duration = time.time() - start

            if v_accuracy > self.best_acc:
                self.best_acc = v_accuracy
                self.best_epoch = epoch

            #self.scheduler.step(v_accuracy)
            self.logger.info("epoch: {} --- t_loss : {:0.3f}, train_acc = {}%, train_f1 = {}, v_loss: {:0.3f}, val_acc: {}%, val_f1 = {}, best_acc: {}%, best_epoch: {}, time: {:0.2f}s, lr: {}"\
                                                           .format(epoch, train_loss, t_accuracy, t_f1_score, valid_loss, v_accuracy, v_f1_score, self.best_acc, self.best_epoch, duration,self.optimizer.param_groups[0]['lr']))
            self.logger.info("epoch: {} --- train TP:{}, FP:{}, FN:{}, TN:{} ------- valid TP:{}, FP:{}, FN:{}, TN:{}"\
                                                           .format(epoch, t_tp,t_fp,t_fn,t_tn,v_tp,v_fp,v_fn,v_tn))
            self.save_checkpoint(epoch, v_accuracy)

            ## Update Tensorboard 
            self.writter.add_scalar('data/Train/Loss', train_loss, epoch)
            self.writter.add_scalar('data/Valid/Loss', valid_loss, epoch)
            self.writter.add_scalar('data/Train/Accuracy', t_accuracy, epoch)
            self.writter.add_scalar('data/Valid/Accuracy', v_accuracy, epoch)
            #pdb.set_trace()
            self.writter.add_scalar('data/Train_F1-Score/Yes', t_f1_score[0], epoch)
            self.writter.add_scalar('data/Valid_F1-Score/Yes', v_f1_score[0], epoch)
            self.writter.add_scalar('data/Train_F1-Score/No', t_f1_score[1], epoch)
            self.writter.add_scalar('data/Valid_F1-Score/No', v_f1_score[1], epoch)

        self.writter.close()


    def train_single_epoch(self, epoch):
        #pdb.set_trace()
        self.model.train()
        total_loss = 0.0
        accuracy = 0.0
        correct_cnt = 0
        tot_cnt = 0
        batch_size = len(self.train_loader)
        tp = np.zeros(self.n_class)
        tn = np.zeros(self.n_class)
        fp = np.zeros(self.n_class)
        fn = np.zeros(self.n_class)

        for b, batch in (enumerate(self.train_loader)):
            if batch == -1:
                continue
            inputs, labels, data_len = batch
            #B, C, T, Freq = inputs.size()
            B, T = inputs.size()
            #print('B: {}'.format(B))
            inputs = inputs.cuda()
            labels = labels.cuda()
            batch_loss = 0.0
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            #[B, T-3 or T-4, class]
            outputs = outputs['clipwise_output']
            scores = outputs
            #scores = outputs.mean(1)
            #scores = F.softmax(scores,dim=1)
            best_prediction = scores.max(-1)[1]

            for i in range(B):
                if labels[i, best_prediction[i]] == 1.0:
                    correct_cnt += 1
            #print("scores {}\nlabel {}".format(scores[0],labels[0]))
            batch_loss = self.criterion(scores, labels)
            batch_loss.backward()
            total_loss += batch_loss.item()
            self.optimizer.step()
            tot_cnt += B
            print("{}/{}: {}/{}".format(b, batch_size, correct_cnt, tot_cnt), end='\r')

            tp_, fp_, fn_, tn_ = intermediate_at_measures(labels.detach().cpu().numpy(),torch.nn.functional.one_hot(best_prediction,num_classes=self.n_class).float().detach().cpu().numpy())
            tp += tp_
            fp += fp_
            fn += fn_
            tn += tn_

        macro_f_measure = np.zeros(self.n_class)
        mask_f_score = 2 * tp + fp +fn != 0
        macro_f_measure[mask_f_score] = 2 * tp[mask_f_score] / (2 * tp + fp + fn)[mask_f_score]

        mean_loss = total_loss / tot_cnt
        return mean_loss, (correct_cnt/tot_cnt)*100, macro_f_measure, tp, fp, fn, tn


    def inference(self):
        self.model.eval()
        total_loss = 0.0
        accuracy = 0.0
        correct_cnt = 0
        tot_cnt = 0
        batch_size = len(self.valid_loader)
        tp = np.zeros(self.n_class)
        tn = np.zeros(self.n_class)
        fp = np.zeros(self.n_class)
        fn = np.zeros(self.n_class)

        with torch.no_grad():
            for b, batch in enumerate(self.valid_loader):
                if batch == -1:
                    continue
                inputs, labels, data_len = batch
                B, T = inputs.size()  
                #print('B: {}'.format(B))
                #B, C, T, Freq = inputs.size()  
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = self.model(inputs)
                #[B, T-4, class]

                # scores = outputs.mean(1)
                # best_prediction = outputs.max(2)[1].mode()[0]
                # scores = torch.zeros(B, 10).to(self.device)
                # inter_diff = (max(data_len) - outputs.size(1)).item()

                # for i in range(B):
                #     scores[i] = outputs[i][:(data_len[i] - inter_diff),:].mean(0)
                # best_prediction = outputs.max(2)[1].mode()[0]
                #scores = outputs.mean(1)
                outputs = outputs['clipwise_output']
                scores = outputs
                # scores = F.softmax(scores,dim=1)
                best_prediction = scores.max(-1)[1]

                for i in range(B):
                    if labels[i, best_prediction[i]] == 1.0:
                        correct_cnt += 1

                batch_loss = self.criterion(scores, labels)
                total_loss += batch_loss.item()

                tot_cnt += B

                print("{}/{}: {}/{}".format(b, batch_size, correct_cnt, tot_cnt), end='\r')
                tp_, fp_, fn_, tn_ = intermediate_at_measures(labels.detach().cpu().numpy(),torch.nn.functional.one_hot(best_prediction,num_classes=self.n_class).float().detach().cpu().numpy())
                tp += tp_
                fp += fp_
                fn += fn_
                tn += tn_

        macro_f_measure = np.zeros(self.n_class)
        mask_f_score = 2 * tp + fp +fn != 0
        macro_f_measure[mask_f_score] = 2 * tp[mask_f_score] / (2 * tp + fp + fn)[mask_f_score]

        mean_loss = total_loss / tot_cnt
        return mean_loss, (correct_cnt/tot_cnt)*100, macro_f_measure, tp, fp, fn, tn


    def load_checkpoint(self, ckpt):
        self.logger.info("Loading checkpoint from {ckpt}")
        print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])#, strict=False)

    def save_checkpoint(self, epoch, vacc, best=True):
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        self.exp_path.joinpath('ckpt').mkdir(exist_ok=True, parents=True)
        save_path = "{}/ckpt/{}_{:0.4f}.pt".format(self.exp_path, epoch, vacc)
        torch.save(state_dict, save_path)
        
        
class ModelTester:
    def __init__(self, model, test_loader, ckpt_path, device):
        #pdb.set_trace()
        
        self.device = torch.device('cuda:{}'.format(device))
        #self.model = model.to(self.device)
        self.model = model.cuda()
        #self.model = model
        self.test_loader = test_loader
        
        '''Set Logger'''
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(sh)

        self.load_checkpoint(ckpt_path)


    def load_checkpoint(self, ckpt):
        #pdb.set_trace()
        self.logger.info(f"Loading checkpoint from {ckpt}")
        # print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)


    def test(self):

        self.model.eval()
        total_loss = 0.0
        output_result = {} 
        output_list = []
        correct_audio_list = []
        label_list = []
        correct_cnt = 0
        tot_cnt = 0
        batch_size = len(self.test_loader)
        #pdb.set_trace()
        with torch.no_grad():
            for b, batch in enumerate(self.test_loader):
                #pdb.set_trace() 
                #inputs, labels = batch
                inputs, labels, audio_path = batch
                #inputs = torch.unsqueeze(inputs,1)
                #B, C, T, Freq = inputs.size()  
                try:
                    B, T = inputs.size()  
                except:
                    print('stop')
                    pdb.set_trace()
                inputs = inputs.cuda()
                labels = labels.cuda()
                #pdb.set_trace()
                if inputs.dtype != torch.float32:
                    inputs = inputs.float()
                outputs = self.model(inputs)
                outputs = outputs['clipwise_output']
                #print(outputs)
                #best_prediction = outputs.max(2)[1].mode()[0]
                best_prediction = outputs.max(1)[1].mode()[0]

                if best_prediction == labels:
                    correct_cnt += 1
                    correct_audio_list.append(audio_path)
                tot_cnt += 1
                print(best_prediction.item(),end = ' ') 
                if str(best_prediction.item()) not in output_result.keys():
                    #output_result[str(best_prediction.item())] = []
                    output_result[str(best_prediction.item())] = 0
                    #correct_audio_list.append(audio_path)
                #else:
                output_result[str(best_prediction.item())] +=1
                #output_result[str(best_prediction.item())].append(data_name)

        print('number of NaN data: {}'.format(self.test_loader.dataset.nan_cnt))
        print('Accuracy: {} ({}/{})'.format(correct_cnt/tot_cnt,correct_cnt,tot_cnt))
        #del(output_result['0'])
        print('output_result: {}'.format(output_result))
        print('correct_audio_list: {}'.format(correct_audio_list))
    
    
