from re import X
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch
import torchvision.transforms as transforms
from copy import deepcopy
parser=argparse.ArgumentParser()
args=parser.parse_args("")
args.exp_name="exp-2022-05-31-normalize"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

## -- Data Loading -- ##
args.batch_size=2048

## MODEL DEFINE ##
args.in_dim=34
args.out_dim=5
args.hid_dim=21
args.n_layers=1
args.act='relu'
args.use_bn=True
args.use_xavier=True   

##-- Regularization --##
args.l2=0.00001
args.dropout=0.3
args.use_bn=True


#-- Optimizer & Training --##
args.optim='Adam'
args.lr=0.0001
args.epoch=2000

class MLP(nn.Module):
    
    def __init__(self,args) -> None:
        super(MLP,self).__init__()
        torch.set_default_dtype(torch.float32)
        self.in_dim=args.in_dim
        self.out_dim = args.out_dim
        self.hid_dim = args.hid_dim
        self.n_layers = args.n_layers
        self.act = args.act
        self.dropout = args.dropout
        self.use_bn = args.use_bn
        self.use_xavier = args.use_xavier

    
        # ====== Create Linear Layers ====== #  
        # self.layernorm=nn.LayerNorm(17)
        self.fc1=nn.Linear(self.in_dim,self.hid_dim)

        self.linears=nn.ModuleList()
        self.bns=nn.ModuleList()
        
        for i in range(self.n_layers-1):
            self.linears.append(nn.Linear(self.hid_dim,self.hid_dim))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.hid_dim))
        
        self.fc2=nn.Linear(self.hid_dim,self.out_dim)
        self.softmax=nn.Softmax(dim=1)

        # ====== Create Activation Function ====== #
        if self.act == 'relu':
            self.act = nn.ReLU()
        elif self.act == 'tanh':
            self.act == nn.Tanh()
        elif self.act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.act == 'leakyrelu':
            self.act=nn.LeakyReLU()
        else:
            raise ValueError('no valid activation function selected!')

        self.dropout=nn.Dropout(self.dropout)
        if self.use_xavier:
            self.__xavier_init__()

    def forward(self,x):

        x=self.act(self.fc1(x))
        for i in range(len(self.linears)):
            x=self.act(self.linears[i](x))
            x=self.bns[i](x)
            x=self.dropout(x)
        x=self.fc2(x)
        x=self.softmax(x)
        
        return x

    def __xavier_init__(self):
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            linear.bias.data.fill_(0.01)
    

#[객체개수 x 17 x 3 ]의 형태를 [객체개수 x 34]의 형태로 반환한다
##이때 3은 x,y,확률로 되어있으며 확률은 버린다
#x0,y0,x1,y1...의 배치구조
#정규화는 x끼리 y끼리 이루어진다
def MyNomalizer(keypoints:torch.tensor,device:str)->torch.tensor:
    keypoint_list=[]
   
    for keypoint in keypoints:
        result_key_data=torch.empty(34,dtype=torch.float32).to(device)
        x_tensor=torch.empty(17,dtype=torch.float32).to(device)
        y_tensor=torch.empty(17,dtype=torch.float32).to(device)
        for idx,data in enumerate(keypoint):
            x,y,prob=data
            x_tensor[idx]=x
            y_tensor[idx]=y
        x_tensor=F.normalize(x_tensor,dim=0)
        y_tensor=F.normalize(y_tensor,dim=0)
        #print(x_tensor)
        
        #print(y_tensor)
        for idx in range(17):
            result_key_data[2*idx]=x_tensor[idx]
            result_key_data[2*idx+1]=y_tensor[idx]
        #print(result_key_data.shape)
        keypoint_list.append(result_key_data)
        #print(result_key_data)
    result=torch.stack(keypoint_list,dim=0)
    return result


#[객체개수 x 17 x 3 ]의 형태를 [객체개수 x 34]의 형태로 반환한다
##이때 3은 x,y,확률로 되어있으며 확률은 버린다
#x0,y0,x1,y1...의 배치구조
#무게중심 이동은 x끼리 y끼리 이루어진다
def center_of_position_nomalizer(keypoints:torch.tensor,device:str)->torch.tensor:
    keypoint_list=[]
    x_cm=0.0
    y_cm=0.0
    for keypoint in keypoints:
        result_key_data=torch.empty(34,dtype=torch.float32).to(device)
        x_tensor=torch.empty(17,dtype=torch.float32).to(device)
        y_tensor=torch.empty(17,dtype=torch.float32).to(device)
        for idx,data in enumerate(keypoint):
            x,y,prob=data
            x_tensor[idx]=x
            y_tensor[idx]=y
            x_cm+=x
            y_cm+=y
        x_cm/=17
        y_cm/=17
        for idx in range(17):
            x_tensor[idx]=x_tensor[idx]-x_cm
            y_tensor[idx]=y_tensor[idx]-y_cm
            result_key_data[2*idx]=x_tensor[idx]
            result_key_data[2*idx+1]=y_tensor[idx]

        keypoint_list.append(result_key_data)
    
    result=torch.stack(keypoint_list,dim=0)
    return result

#[객체개수 x 17 x 3 ]의 형태를 [객체개수 x 34]의 형태로 반환한다
##이때 3은 x,y,확률로 되어있으며 확률은 버린다
#x0,y0,x1,y1...의 배치구조
#무게중심 이동은 x끼리 y끼리 이루어진다
#정규화를 먼저 하고 무게중심을 기준으로 이동한다.
def center_of_position_nomalizer_with_normalize(keypoints:torch.tensor,device:str)->torch.tensor:
    keypoint_list=[]
    x_cm=0.0
    y_cm=0.0
    for keypoint in keypoints:
        result_key_data=torch.empty(34,dtype=torch.float32).to(device)
        x_tensor=torch.empty(17,dtype=torch.float32).to(device)
        y_tensor=torch.empty(17,dtype=torch.float32).to(device)
        for idx,data in enumerate(keypoint):
            x,y,prob=data
            x_tensor[idx]=x
            y_tensor[idx]=y
            
        x_tensor=F.normalize(x_tensor,dim=0)
        y_tensor=F.normalize(y_tensor,dim=0)

        for x in x_tensor:
            x_cm+=x
        for y in y_tensor:
            y_cm+=y
        x_cm/=17
        y_cm/=17
        for idx in range(17):
            x_tensor[idx]=x_tensor[idx]-x_cm
            y_tensor[idx]=y_tensor[idx]-y_cm
            result_key_data[2*idx]=x_tensor[idx]
            result_key_data[2*idx+1]=y_tensor[idx]

        keypoint_list.append(result_key_data)
    
    result=torch.stack(keypoint_list,dim=0)
    return result
        