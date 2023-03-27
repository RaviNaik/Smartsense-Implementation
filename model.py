import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import logging
from data.kr.dictionary import device_control_dict

# logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.basicConfig(filename='train.log', filemode='w', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   level=logging.INFO)


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        
        with open(path, "rb") as f:
            self.train_data = pickle.load(f)
            
        self.X = self.train_data[:,:,[0,1,2,4]]
        self.y = self.train_data[:,9,4]
        self.X = torch.LongTensor(self.X)
        self.y = nn.functional.one_hot(torch.LongTensor(self.y), num_classes=272)
        self.y = self.y.type(torch.float)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
class QueryAttention(nn.Module):
    def __init__(self, embedding_dim=50, query_embedding_dim=50) -> None:
        super().__init__()
        
        self.fc = nn.Linear(in_features=embedding_dim, 
                            out_features=query_embedding_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, H, q):
        fch = self.tanh(self.fc(H))
        # beta = torch.einsum("k,...k->...", q, fch)
        if H.dim() == 3:
            beta = torch.einsum("k,ijk->ij", q, fch)
            alpha = torch.softmax(beta, dim=-1)
            out = torch.einsum("ij,ijk->ik", alpha, H)
        else:
            beta = torch.einsum("k,ik->i", q, fch)
            alpha = torch.softmax(beta, dim=-1)
            out = torch.einsum("i,ik->k", alpha, H)
        
        return out
        # out -> d dim
            
class QTEModule(nn.Module):
    def __init__(self, 
                 embedding_dim,
                 query_embedding_dim,
                 nhead=2,
                 nlayers=2,
                 dropout=0.1) -> None:

        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                        nhead=nhead,
                                                        dropout=dropout)

        self.attention_module = nn.ModuleList([
            nn.TransformerEncoder(self.encoder_layer,
                                 num_layers=nlayers),
            
            nn.TransformerEncoder(self.encoder_layer,
                                 num_layers=nlayers),
            ])

        self.query_attention_module = QueryAttention(embedding_dim=embedding_dim,
                                                     query_embedding_dim=query_embedding_dim)
        
    def forward(self, x, q):
        # if x.dim() == 1:
        #     x = x.unsqueeze(0)
        # q = q.unsqueeze(0)
        for encoder in self.attention_module:
            x = encoder(x)
        # x -> 4xd dim action encoder | (t-1) x d sequence encoder
    
        h = self.query_attention_module(x, q)
        return h
    

class SmartSenseModel(nn.Module):
    def __init__(self, 
                 num_sequences,
                 device_control_values,
                 nheads=2,
                 embedding_dim=50,
                 query_embedding_dim_act_enc=200,
                 query_embedding_dim_seq_enc=100,
                 len_day=8,
                 len_hour=9,
                 len_device=40,
                 len_device_ctrl=268) -> None:
        super().__init__()
        
        # self.d_model = len_day//2 + len_hour//2 + len_device//2 + len_device_ctrl//2
        # self.d_model_seq = num_sequences * self.d_model
        self.num_sequences = num_sequences
        self.embedding_dim = embedding_dim
        self.query_embedding_dim_act_enc = query_embedding_dim_act_enc
        self.query_embedding_dim_seq_enc = query_embedding_dim_seq_enc
        self.device_control_values = device_control_values
        
        self.day_embeds = nn.Embedding(num_embeddings=len_day, 
                                       embedding_dim=self.embedding_dim)
        
        self.hour_embeds = nn.Embedding(num_embeddings=len_hour, 
                                        embedding_dim=self.embedding_dim)
        
        self.device_embeds = nn.Embedding(num_embeddings=len_device, 
                                          embedding_dim=self.embedding_dim)
        
        self.device_ctrl_embeds = nn.Embedding(num_embeddings=len_device_ctrl, 
                                               embedding_dim=self.embedding_dim)
        
        self.action_encoder = QTEModule(embedding_dim=self.embedding_dim,
                                        query_embedding_dim=self.query_embedding_dim_act_enc, 
                                        nhead=nheads)
        
        self.sequence_encoder = QTEModule(embedding_dim=self.embedding_dim,
                                          query_embedding_dim=self.query_embedding_dim_seq_enc,
                                          nhead=nheads)
        
        self.pe = self.pos_embedding_sinusoidal(max_seq_len=1,
                                                embedding_dim=self.embedding_dim, 
                                                is_cuda=False)
        
        self.pos_emb = nn.Embedding(9, self.embedding_dim)
        
        # self.td = TimeDistributed(self.action_encoder, True)
        
    def generate_global_query(self, x):
        day_of_week = x[:,0]
        hour_of_day = x[:,1]
        device_name = x[:,2]
        device_action = x[:,-1]
        
        day_embeddings = self.day_embeds(day_of_week).mean(dim=0)
        hour_embeddings = self.hour_embeds(hour_of_day).mean(dim=0)
        device_name_embeddings = self.device_embeds(device_name).mean(dim=0)
        device_action_embeddings = self.device_ctrl_embeds(device_action).mean(dim=0)
        
        return torch.concatenate([
            device_name_embeddings,
            device_action_embeddings,
            day_embeddings,
            hour_embeddings
        ])
        
    # def initialize(self, device_control_values):
    #     self.e = self.device_ctrl_embeds(device_control_values, requires_grad=False)
        # e -> Nd X d -> (No of device ctrl/action X embed dim) -> 268 x 50
        
    def forward(self, x):
        # out = torch.zeros((X.shape[0], self.e.shape[0]))
        # for ind_, x in enumerate(X):
        cc = x[9,[0,1]].T
        x = x[:9,:]
        q = self.generate_global_query(x)
        
        # sequence_ips = []
        # for sequence in x:
        day_of_week = x[:,0]
        hour_of_day = x[:,1]
        device_name = x[:,2]
        device_action = x[:,-1]
        
        day_embeddings = self.day_embeds(day_of_week)
        hour_embeddings = self.hour_embeds(hour_of_day)
        device_name_embeddings = self.device_embeds(device_name)
        device_action_embeddings = self.device_ctrl_embeds(device_action)
        
        ip = torch.stack([
            device_name_embeddings,
            device_action_embeddings,
            day_embeddings,
            hour_embeddings,], dim=1)
        # ip -> 4xd dim

        hs = self.action_encoder(ip, q)
        # hs -> d dim
        
        positions = torch.arange(hs.shape[0]).to(device)
        hspe = hs + self.pos_emb(positions)
            # sequence_ips.append(hspe)
            
        # sequence_ips = torch.concatenate(sequence_ips, dim=0)
        # sequence_ips -> (t-1) x d -> No samples X embed dim
        # 9 X 50
        cc_embeddings = torch.concatenate([self.day_embeds(cc[0]), 
                            self.hour_embeds(cc[1])])
        # cc_embeddings -> (2*d) dim -> 100
        
        s = self.sequence_encoder(hspe, cc_embeddings)
        # s -> d dim -> 50
        
        E = self.device_ctrl_embeds(device_control_values)
        yhat = torch.softmax(torch.matmul(E, s), dim=0)
        # out[ind_,:] = yhat
        return yhat
    
    @staticmethod
    def pos_embedding_sinusoidal(max_seq_len, embedding_dim, is_cuda):
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.stack((torch.sin(emb), torch.cos(emb)), dim=0).view(
            max_seq_len, -1).t().contiguous().view(max_seq_len, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(max_seq_len, 1)], dim=1)
        if is_cuda:
            return emb.cuda()
        return emb
    
def get_training_data():
    with open("data/kr/trn_instance_10.pkl", "rb") as f:
        train_data = pickle.load(f)
        
    return train_data

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
writer = SummaryWriter("/runs/initial")

num_sequences=9
nheads=2
embedding_dim=50
query_embedding_dim_act_enc = 200
query_embedding_dim_seq_enc = 100
len_day=8
len_hour=9
len_device=40
len_device_ctrl=272
pe_embed_dim=130
device_control_values = torch.LongTensor(list(device_control_dict.values())).to(device)


model = SmartSenseModel(num_sequences,
                device_control_values,
                nheads,
                embedding_dim,
                query_embedding_dim_act_enc,
                query_embedding_dim_seq_enc,
                len_day,
                len_hour,
                len_device,
                len_device_ctrl
                )

model.to(device)
# model.initialize(device_control_values)
# td = TimeDistributed(model, True)
train_data = get_training_data()
# X = train_data[:2,:,[0,1,2,4]]
# X = torch.LongTensor(X)

# cc = train_data[0,9,[0,1]].T
# cc = torch.LongTensor(cc)

# y = train_data[:2,9,4]
# y = torch.LongTensor(y)

# outputs = model(X)

# print(torch.argmax(outputs, dim=1))

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(params=model.parameters(), 
                             lr=learning_rate)

train_dataset = SeqDataset(path="data/kr/trn_instance_10.pkl")
train_dataloader = DataLoader(train_dataset, 1024)

epochs = 10
losses = []
step=0
for epoch in range(epochs):
    for batchX,batchy in train_dataloader:
        batchX = batchX.to(device)
        batchy = batchy.to(device)
        yhat = []
        optimizer.zero_grad()
            
        model.train()
        for X in batchX:            
            outputs = model(X)
            yhat.append(outputs)
        
        # yhat = td(batchX)
        
        loss = loss_fn(torch.vstack(yhat), batchy)
        loss.backward()


        optimizer.step()
        print(f"LOSS: {loss.item()}")
        writer.add_scalar("Training Loss", loss.item(), global_step=step)
        logging.info(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item()}")
        step+=1
                
    losses.append(loss.item())
    print(f"Epoch: {epoch} | Loss: {loss.item()}")
    writer.add_scalar("Epoch Loss", loss.item(), global_step=epoch)
    logging.info(f"Epoch: {epoch} | Loss: {loss.item()}")
