from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers    #args.e_layers = 2 / args.s_layers = None
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:                       #不実行 args.use_multi_gpu = False / args.use_gpu = True
            model = nn.DataParallel(model, device_ids=self.args.device_ids)     #不実行
            
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]               # Data = Dataset_Custom
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        elif flag=='all':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
            
        data_set = Data(                                 # Data = Dataset_Custom
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))                       # train:3506 ／ val:508 ／ test:1022
        
        data_loader = DataLoader(
            data_set,                                    # data.data_loader.Dataset_Custom object
            batch_size=batch_size,                       # batch_size = 32
            shuffle=shuffle_flag,                        # shuffle = True
            num_workers=args.num_workers,                # num_workers = 0
            drop_last=drop_last)                         # drop_last = True

        return data_set, data_loader                     # train_data, train_loader ／ vali_data, vali_loader ／ test_data, test_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#            pred, true = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred, true, extra = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        
        
        
#//////////////////////////////////////////////////////////////////////////////////////////////////////
#        import requests
#        import pprint

#        print('\n■train■\n')
#        print('\n▼train_data\n')
#        pprint.pprint(vars(train_data))
#        print('\n▼train_loader\n')
#        pprint.pprint(vars(train_loader))

#        print('\n▼stamp\n')
#        print(train_data.data_stamp.shape)
#        print('\n▼x\n')
#        print(train_data.data_x.shape)
#        print('\n▼y\n')
#        print(train_data.data_y.shape)

# ■train_loader
# ┣━ いろいろ
# ┗━■dataset (=train_data)
#   ┣━いろいろ
#   ┣━data_stamp:(3611,5) … vali:517 ／ test:1031
#   ┣━data_x    :(3611,8) … 
#   ┣━data_y    :(3611,8) …
#   ┃
#   ┣━[   0][0] ~ [   0][3] : (96, 8)／(58, 8)／(96, 5)／(58, 5)
#   ┣━ …
#   ┣━[3505][0] ~ [3505][3] : (96, 8)／(58, 8)／(96, 5)／(58, 5)
#   ┣━[3506][0] ~ [3506][3] : (96, 8)／(57, 8)／(96, 5)／(57, 5)
#   ┣━ …
#   ┣━[3516][0] ~ [3562][3] : (95, 8)／(47, 8)／(95, 5)／(47, 5)
#   ┣━ … 
#   ┣━[3562][0] ~ [3562][3] : (49, 8)／( 1, 8)／(49, 5)／( 1, 5)
#   ┣━[3563][0] ~ [3563][3] : (48, 8)／( 0, 8)／(48, 5)／( 0, 5)
#   ┣━ …
#   ┗━[3610][0] ~ [3610][3] : ( 1, 8)／( 0, 8)／( 1, 5)／( 0, 5)
#//////////////////////////////////////////////////////////////////////////////////////////////////////
        
        

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)                                                      # len(train_loader) = 109
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):     #batch_x     :(32,96,8)
                                                                                               #batch_y     :(32,58,8)
                                                                                               #batch_x_mark:(32,96,5)
                                                                                               #batch_y_mark:(32,58,5)
                iter_count += 1
                model_optim.zero_grad()                                                        #テンソルの勾配を0に初期化
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#                pred, true = self._process_one_batch(                                          #pred:(32,10,8)
#                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)                  #true:(32,10,8)
                pred, true, extra = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
                loss = criterion(pred, true)                                                   #loss:スカラー
                
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#            pred, true = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred, true, extra = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        all_data, all_loader = self._get_data(flag='all')
    
        print('▼ここから')
        embedding_list = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(all_loader):
            pred, true, extra = self._process_one_batch(all_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            embedding_list.append(extra)
            print(torch.stack(embedding_list).size())   
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#            pred, true = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred, true, extra = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)                                            # batch_x     :(32,96,8)
        batch_y = batch_y.float()                                                            # batch_y     :(32,58,8)
        batch_x_mark = batch_x_mark.float().to(self.device)                                  # batch_x_mark:(32,96,5)
        batch_y_mark = batch_y_mark.float().to(self.device)                                  # batch_y_mark:(32,58,5)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()      # dec_inp:(32,10,8) pudding=0 のため要素0のテンソルを生成
        elif self.args.padding==1:                                                                        # pudding=0 不実行
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()       # 
            
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)  # dec_inp:(32,58,8) … batch_y(32,0:48,8) + dec_inp:(32,10,8)

        
        
        # encoder - decoder
        if self.args.use_amp:                                                                 # self.args.use_amp=false のため不実行
            with torch.cuda.amp.autocast():                                                   # ×
                if self.args.output_attention:                                                # ×
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]     # ×
                else:                                                                         # ×
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)        # ×
                    
        else:
            if self.args.output_attention:                                                    # self.args.output_attention=false のため不実行
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]         # ×
            else:
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)            # output:(32,10,8)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                enbedding_vec = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[1]
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    

        if self.args.inverse:                                                                 # self.args.inverse=false のため不実行
            outputs = dataset_object.inverse_transform(outputs)                               # ×

        f_dim = -1 if self.args.features=='MS' else 0                                         # f_dim = 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)                      # batch_y:(32,10,8) … batch_y:(32,58,8)の最後10要素を取得

#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#        return outputs, batch_y
        return outputs, batch_y, enbedding_vec
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
