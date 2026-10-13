import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from random import sample
from einops import rearrange
import numpy as np
# from transformer import *
from VAE import vae_enc_mlp
class encoder(nn.Module):
    def __init__(self, n_dim, dims, emb_dim):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])
        self.enc_2 = Linear(dims[0], dims[1])
        self.enc_3 = Linear(dims[1], dims[2])
        self.z_layer = Linear(dims[2], emb_dim)
        self.z_b0 = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        # print(enc_h3.shape)
        z = self.z_b0(self.z_layer(enc_h3))
        return z


class decoder(nn.Module):
    def __init__(self, n_dim, dims, emb_dim):
        super(decoder, self).__init__()
        self.dec_0 = Linear(emb_dim, emb_dim)
        self.dec_1 = Linear(emb_dim, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar
class decoder2(nn.Module):
    def __init__(self, n_dim, dims, emb_dim):
        super(decoder2, self).__init__()
        self.dec_0 = Linear(emb_dim, emb_dim)
        self.dec_1 = Linear(emb_dim, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar
def information_fusion(y_view,inc_ind_V):
    bool_we=inc_ind_V.bool()
    
    # y_v0=-y_view[0] *torch.log(y_view[0])
    # y_v1=-y_view[1] *torch.log(y_view[1])
    # y_v2=-y_view[2] *torch.log(y_view[2])
    # y_v3=-y_view[3] * torch.log(y_view[3])
    # y_v4=-y_view[4] * torch.log(y_view[4])
    # y_v5=-y_view[5] * torch.log(y_view[5])
    
    # y_v0=y_view[0] * (1-y_view[0])
    # y_v1=y_view[1] * (1-y_view[1])
    # y_v2=y_view[2] * (1-y_view[2])
    # y_v3=y_view[3] * (1-y_view[3])
    # y_v4=y_view[4] * (1-y_view[4])
    # y_v5=y_view[5] * (1-y_view[5])
    
    mean_vectors = []
    for i in range(len(y_view)):
        # y_v=-y_view[i] *torch.log(y_view[i])
        y_v=1-y_view[i] * (1-y_view[i])
        # 沿着第一个维度（行）计算均值，keepdim=True 保持维度
        mean_vector = torch.mean(y_v, dim=1, keepdim=True)
        # 将均值向量添加到列表中
        mean_vectors.append(mean_vector)   
    
    concatenated_tensor = torch.cat(mean_vectors, dim=1)
    concatenated_tensor=concatenated_tensor*inc_ind_V
    
    row_sums = torch.sum(concatenated_tensor, dim=1, keepdim=True)
    weight_m= concatenated_tensor/ row_sums
    
    # weight_m=torch.softmax(concatenated_tensor,1)
    return weight_m

class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,dropout_rate=0.,final_act=True,final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        # init layers
        self.mlps =[]
        layers = []
        
        if len(hidden_dim)>0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            # layers.append(nn.Dropout(dropout_rate))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                # layers.append(nn.Dropout(dropout_rate))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            # layers.append(nn.Dropout(dropout_rate))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
            # x = x + y
        return x

# class MLP(nn.Module):
#     def __init__(self,input_dim,hid_dimen1,hid_dimen3):
#         super(MLP,self).__init__()
#         self.Mlp =  nn.Sequential(
#         nn.Linear(input_dim,hid_dimen1),
#         nn.Tanh(),
#         nn.Linear(hid_dimen1, hid_dimen3),
#         nn.Tanh(),
#         )

#     def forward(self, x):
#         mlp = self.Mlp(x)
#         return mlp

def batch_perm(batch_data,masks):
    n = batch_data.shape[1]
    rotated_flat_matrices =torch.zeros_like(batch_data)
    for i, data in enumerate(batch_data):
        rotation_list = generate_rotation_list(n)
        index_row,index_col = create_index_matrix(n,rotation_list,masks[i])
        new_data = data[index_row,index_col]
        rotated_flat_matrices[i] = new_data
    return rotated_flat_matrices
def rotate_matrix_using_indexes(matrix, index_matrix):
    """使用索引矩阵重构矩阵"""
    flat_matrix = matrix.flatten(0,1)
    rotated_flat_matrix = flat_matrix[index_matrix.flatten(0,1)]
    return rotated_flat_matrix.reshape(matrix.shape)
def generate_rotation_list(n):
    """生成一个从 0 到 N-1 的随机排序列表"""
    rotation_list = np.arange(n)
    np.random.shuffle(rotation_list)
    return rotation_list
def create_index_matrix(n, rotation_list, mask):
    """创建一个索引矩阵，用于重构旋转后的矩阵"""
    index_row = torch.arange(n).view(-1,1).repeat(1,n) #[[0,0,0,0] [1,1,1,1] [2,2,2,2] [3,3,3,3]]
    index_col = torch.arange(n).view(1,-1).repeat(n,1) #[[0,1,2,3] [0,1,2,3] [0,1,2,3] [0,1,2,3]]
    for col in range(n):
        rotation_amount = rotation_list[col]
        # 计算旋转后的索引，考虑屏蔽
        for i in range(n):
            step = rotation_amount
            if mask[i] == 1:  # 只考虑未被屏蔽的行
                new_position = i 
                while step>0:
                    new_position = (new_position + 1) % n
                    step = step - 1 if mask[new_position] != 0 else step
                index_row[new_position, col] = i
                # index_col[new_position, col] = col 
    return index_row,index_col
def poe_aggregate( mu, var, mask=None, eps=1e-5):
    #mu [b m d] var [b m d] mask [b m]
    if mask is None:
        mask_matrix = torch.ones_like(mu).to(mu.device)
    else:
        mask_matrix = mask.unsqueeze(-1)
    # mask_matrix = torch.stack(mask, dim=0)
    mask_matrix_new = torch.cat([torch.ones_like(mask_matrix)[:,:1].cuda(),mask_matrix],dim=1)
    p_z_mu = torch.zeros_like(mu)[:,:1].cuda()
    p_z_var = torch.zeros_like(var)[:,:1].cuda()
    mu_new = torch.cat([p_z_mu,mu],dim=1)
    var_new = torch.cat([p_z_var,var],dim=1)
    exist_mu = mu_new * mask_matrix_new
    T = 1. / (var_new+eps)
    if torch.sum(torch.isnan(exist_mu)).item()>0:
        print('.')
    if torch.sum(torch.isinf(T)).item()>0:
        print('.')
    exist_T = T * mask_matrix_new
    aggregate_T = torch.sum(exist_T, dim=1)
    aggregate_var = 1. / (aggregate_T + eps)
    # if torch.sum(torch.isnan(aggregate_var)).item()>0:
    #     print('.')
    aggregate_mu = torch.sum(exist_mu * exist_T, dim=1) / (aggregate_T + eps)
    if torch.sum(torch.isnan(aggregate_mu)).item()>0:
        print(',')
    return aggregate_mu, aggregate_var

def combination( mu, var, mask=None, eps=1e-5):

    var_star = 1/torch.sum(1/(var + eps), dim=2)
    mu_star = var_star * torch.sum(mu/(var + eps), dim=2)
    return mu_star, var_star

def reparameterize( mu: torch.tensor, var: torch.tensor, times: int=2) -> torch.tensor:
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    """
    std = torch.sqrt(var+1e-8)
    assert torch.sum(std<0).item()==0
    res = torch.zeros_like(mu).to(mu.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + mu
    return res/times

class net(nn.Module):

    def __init__(self, n_stacks, d_list, emb_dim, class_num):
        super(net, self).__init__()


        dims = []
        for n_dim in d_list:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)
        self.encoder_list = nn.ModuleList([vae_enc_mlp(d_list[i],emb_dim,hidden_dim=[512,512,512]) for i in range(len(d_list))])
        # self.encoder2_list = nn.ModuleList([MLP(emb_dim,emb_dim,hidden_dim=[512,512,512]) for i in range(len(d_list))])
        self.decoder_list = nn.ModuleList([MLP(emb_dim,d_list[i],hidden_dim=[512],final_act=False,final_norm=False) for i in range(len(d_list))])
        # self.encoder_list = nn.ModuleList([encoder(d_list[i], dims[i], emb_dim) for i in range(len(d_list))])
        # self.decoder_list = nn.ModuleList([decoder(d_list[i], dims[i], emb_dim) for i in range(len(d_list))])
        self.regression = Linear(1*emb_dim, class_num)
        self.regression_s = Linear(1*emb_dim, class_num)
        self.regression_v = Linear(1*emb_dim, class_num)
        self.act = nn.Sigmoid()
        self.class_num = class_num
        self.BN = nn.BatchNorm1d(emb_dim)
        self.emb_dim=emb_dim
        self.num_views=len(d_list)
        self.crossM_enc_matrix = nn.ModuleDict()
        for i in range(self.num_views):
            for j in range(self.num_views):
                self.crossM_enc_matrix["enc%d%d" % (i, j)] = vae_enc_mlp(d_list[i],self.emb_dim)
                # self.crossM_enc_matrix["enc%d%d" % (i, j)] = MLP(self.emb_dim, self.emb_dim,[512,512,512])
         
        self.classifier_view=Linear(emb_dim, self.num_views)#MLP(self.nz,512,self.num_views)

        # self.encoder0=Linear(d_list[0], 1024)
        # self.encoder1=Linear(d_list[1], 1024)
        # self.encoder2=Linear(d_list[2], 1024)
        # self.encoder3=Linear(d_list[3], 1024)
        # self.encoder4=Linear(d_list[4], 1024)
        # self.encoder5=Linear(d_list[5], 1024)
        # self.encoderi=[self.encoder0,self.encoder1,self.encoder2,self.encoder3,self.encoder4,self.encoder5]
        # dimsC=[int(round(1024* 0.8)),int(round(1024 * 0.8)),1500]
        # self.encoder=encoder(1024, dimsC, emb_dim)
       
    def forward(self, mul_X, inc_ind_V,mode,sigma):
        # dep_graph = torch.eye(self.class_num,device=we.device).float()
        batch_size = mul_X[0].shape[0]
        num_views = inc_ind_V.shape[1]
        summ = 0
        prop = sigma
        views_embs = []
        # if mode =='train':
            # for i,X in enumerate(mul_X):
            #     mask_len = int(prop*X.size(-1))
            #     mask = torch.ones_like(X)
            #     # for j in range(mask.shape[0]):
            #     #     zero_indices = torch.randperm(mask.shape[1])[:mask_len]
            #     #     mask[j, zero_indices] = 0
            #     mul_X[i] = mul_X[i].mul(mask)
        
        
        # for enc_i, enc in enumerate(self.encoder_list):        
        #     missing_indices = torch.nonzero(inc_ind_V[:,enc_i]== 0).squeeze() #先编码后填充
        #     z_mu,z_var = enc(mul_X[enc_i])     #[128, 512]     
        #     # mu,var =  self.crossM_enc_matrix["enc%d%d"%(enc_i,enc_i)](z_i)       
        #     z_i = reparameterize(z_mu,z_var)
        #     # z_i = self.encoder2_list[enc_i](z_i)  
        
        #     # z_i[missing_indices]=0
        #     views_embs.append(z_i)
            
        cross_enc_mu = torch.zeros(batch_size, self.num_views, self.num_views, self.emb_dim).to(mul_X[0].device)
        cross_enc_var = torch.zeros(batch_size, self.num_views, self.num_views, self.emb_dim).to(mul_X[0].device)
        # for i in range(self.num_views):
        #     for j in range(self.num_views):
        #         cross_enc_mu[:,i,j,:], cross_enc_var[:,i,j,:] = self.crossM_enc_matrix["enc%d%d"%(i,j)](mul_X[i])
        # cross_enc_mu_perm  =  batch_perm(cross_enc_mu,inc_ind_V) #[128, 6, 512]
        # cross_enc_var_perm =  batch_perm(cross_enc_var,inc_ind_V) #[128, 6, 512]
        
        # enc_mu_fusion = cross_enc_mu.mean(dim=2)
        # enc_mu_perm_fusion = enc_mu_perm_fusion.mean(dim=2)
        # mu_comb_views, var_comb_views = combination(cross_enc_mu,cross_enc_var)
        # mu_comb_views_perm, var_comb_views_perm = combination(cross_enc_mu_perm,cross_enc_var_perm)
        
        
        for i in range(self.num_views):
            cross_enc_mu[:,i,i,:], cross_enc_var[:,i,i,:] = self.crossM_enc_matrix["enc%d%d"%(i,i)](mul_X[i])
        mu_comb_views = cross_enc_mu.diagonal(dim1=1, dim2=2).permute(0, 2, 1) #[B M D]
        var_comb_views = cross_enc_var.diagonal(dim1=1, dim2=2).permute(0, 2, 1)
        # enc_mu_perm_fusion = cross_enc_mu_perm.diagonal(dim1=1, dim2=2)
        # enc_var_perm_fusion = cross_enc_var_perm.diagonal(dim1=1, dim2=2)
        
        cross_enc_sampling = reparameterize(mu_comb_views,var_comb_views) #torch.Size([128, 6, 512]) 
        # cross_enc_sampling_perm = reparameterize(enc_mu_perm_fusion,enc_var_perm_fusion)#torch.Size([128, 6, 512])
        # print(cross_enc_sampling.shape,cross_enc_sampling_perm.shape)
        #     summ += torch.diag(we[:, enc_i]).mm(z_i)
        # wei = 1 / torch.sum(we, 1)
        # s_z = torch.diag(wei).mm(summ)    
        
        
        
        
        y_s=[]
        # for i in range(num_views):
        #     y1=self.act(self.regression_s(F.relu(views_embs[i])))
        #     y_s.append(y1)
        
        ####### Fusion Multiview  ########
        
        enc_mu_poe, enc_var_poe = poe_aggregate(mu_comb_views, var_comb_views, inc_ind_V)
        # enc_mu_perm_poe, enc_var_perm_poe = poe_aggregate(mu_comb_views_perm, var_comb_views_perm, inc_ind_V)
        poe_sampling = reparameterize(enc_mu_poe,enc_var_poe) #torch.Size([128, 512]) 
        
        x_bar_list = []
        for dec_i, dec in enumerate(self.decoder_list):
            x_bar_list.append(dec(poe_sampling))
            # x_bar_list.append(dec(enc_mu_perm_fusion[:,dec_i]))
            # x_bar_list_c.append(dec(z))
            
        # weight_c=information_fusion(y_s,inc_ind_V)
        # f_emb=torch.zeros_like(enc_mu_fusion[:,0])
        # for i in range(num_views):
        #     # 提取权重矩阵 W 的第一列
        #     # first_column_of_W = weight_c[:, i]
        #     first_column_of_W = inc_ind_V[:,i]
        #     # 将矩阵 A 的每一行与第一列的值相乘
        #     result = enc_mu_fusion[:,i] * first_column_of_W.view(-1, 1)
        #     f_emb = f_emb + result
        
        
        
        # summvz = 0
        # viewsp_zs = []
        # for enc_i, enc in enumerate(self.encoder2_list):
            
        #     missing_indices = torch.nonzero(we[:,enc_i]== 0).squeeze() #先编码后填充
        #     z_i = enc(mul_X[enc_i])                        
        #     z_i[missing_indices]=0
        #     viewsp_zs.append(z_i)
        
        z = poe_sampling
        # z = f_emb.mul(f_emb.sigmoid()) #使用自身作激活
        # z = self.BN(z)
        # z = F.relu(z)
        # z = s_z+v_z

        # x_bar_list = []
        # for dec_i, dec in enumerate(self.decoder_list):
        #     x_bar_list.append(dec(views_embs[dec_i]))
        #     # x_bar_list.append(dec(torch.cat((views_emb[dec_i],viewsp_zs[dec_i]),-1)))
        
        

        logi = self.regression(z) #[n c]     
        yLable = self.act(logi)
        
        return x_bar_list,yLable, z, views_embs,[mu_comb_views,var_comb_views],[enc_mu_poe, enc_var_poe]
def get_model(n_stacks,d_list,emb_dim,class_num,device):
    model = net(n_stacks=n_stacks,d_list=d_list,emb_dim=emb_dim,class_num=class_num).to(device)
    return model