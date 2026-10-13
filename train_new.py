import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model_acti import get_model,gaussian_reparameterization_var 
import evaluation
import torch
import numpy as np
import copy
from myloss_new import Loss
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torchstat import stat
# from constructGraph import getMvKNNGraph, getIncMvKNNGraph

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
def view_mixup(data_list,label,v_mask,l_mask):
    new_data = []
    new_labels = label.unsqueeze(0).repeat(len(data_list),1,1)
    num = data_list[0].shape[0]
    for i,view_data in enumerate(data_list):
        indices = torch.randperm(num).cuda()
        new_data.append(torch.index_select(view_data, 0, indices))
        new_labels[i] = torch.index_select(new_labels[i],0,indices)
        v_mask[:,i] = torch.index_select(v_mask[:,i],0,indices)
        
    label = new_labels.sum(dim=0)
    label = torch.masked_fill(label,label>0,1)
    l_mask = torch.masked_fill(l_mask,label>0,1)
    return new_data,label,v_mask,l_mask


def build_bce_driven_weights(preds_views, label, inc_L_ind, inc_V_ind, loss_model):
    per_view_losses = []
    for v in range(preds_views.shape[0]):
        view_label_mask = inc_L_ind * inc_V_ind[:, v:v + 1]
        view_loss = loss_model.weighted_BCE_loss(preds_views[v].detach(), label, view_label_mask, reduction='none')
        per_view_losses.append(view_loss)

    per_view_losses = torch.stack(per_view_losses, dim=-1)  # [B, C, M]
    view_mask = inc_V_ind.unsqueeze(1)
    target_logits = (-per_view_losses).masked_fill(view_mask <= 0, -1e9)
    bce_driven_weights = torch.softmax(target_logits, dim=-1) * view_mask
    bce_driven_weights = bce_driven_weights / (bce_driven_weights.sum(dim=-1, keepdim=True) + 1e-8)
    return bce_driven_weights.detach()


def masked_mse_loss(prediction, target, weight):
    diff = (prediction - target) ** 2
    diff = diff * weight
    return diff.sum() / (weight.sum() + 1e-8)


def train(loader, model, loss_model, opt, sche, epoch,last_preds,logger):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    mce= nn.MultiLabelSoftMarginLoss()
    model.train()
    end = time.time()
    All_preds = torch.tensor([]).cuda()
    for i, (data, label, inc_V_ind, inc_L_ind, row_index) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.clone().to('cuda:0') for v_data in data]
        # sim_based_label = getMvKNNGraph(label)
        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        row_index = row_index.float().to('cuda:0')
        # print(label.shape,label.sum())
        # data,label,inc_V_ind,inc_L_ind = view_mixup(data,label,inc_V_ind,inc_L_ind)

        # pred, label_embedding,  p_pre = model(data,mask=torch.ones_like(inc_V_ind))
        if last_preds is None:
            last_p = label*inc_L_ind
        else:
            last_p = last_preds[i*args.batch_size:i*args.batch_size+label.shape[0]].detach()
        latefus_weight = args.omega
        z_sample, views_dis_mu, views_dis_var, common_dis_mu, common_dis_var, xr_list,\
        pred, label_emb_sample, label_emb,label_emb_var, cross_enc_mu, cross_enc_var, pred_diag, preds_views, perception_weights = model(data,row_index,mask=inc_V_ind, latefus_weight=latefus_weight)

        # if i == 0 : 
        #     compute_and_plot_pairwise_kl(fig_dir="./kl_heatmaps",means=cross_enc_mu[0][:4].diagonal(dim1=1, dim2=2).transpose(1,2),variances = cross_enc_var[0][:4].diagonal(dim1=1, dim2=2).transpose(1,2),
        #                                  epoch=epoch,sample_ids=[str(i) for i in range(4)])
        mode = 0 if epoch<args.pre_epochs else 1 # 
        # if epoch<args.pre_epochs:
        #     loss_mse_viewspec = 0
        

        #     loss_CL_views = 0
        #     loss_list=[]

        #     loss = loss_CL_views
        #     assert torch.sum(torch.isnan(loss)).item() == 0
        #     assert torch.sum(torch.isinf(loss)).item() == 0
        
        All_preds = torch.cat([All_preds,pred[0]],dim=0)
        loss_CL = loss_model.weighted_BCE_loss(pred[0],label,inc_L_ind)
        loss_CL_views = torch.tensor(0.0, device=label.device)
        valid_view_num = 0
        for v in range(preds_views[0].shape[0]):
            view_label_mask = inc_L_ind * inc_V_ind[:, v:v+1]
            if torch.sum(view_label_mask).item() == 0:
                continue
            loss_CL_views += loss_model.weighted_BCE_loss(preds_views[0][v], label, view_label_mask)
            valid_view_num += 1
        if valid_view_num > 0:
            loss_CL_views = loss_CL_views / valid_view_num

        loss_perception = torch.tensor(0.0, device=label.device)
        if epoch >= args.pre_epochs:
            bce_driven_weights = build_bce_driven_weights(preds_views[0], label, inc_L_ind, inc_V_ind, loss_model)
            perception_supervision_mask = inc_L_ind.unsqueeze(-1) * inc_V_ind.unsqueeze(1)
            loss_perception = masked_mse_loss(perception_weights[0], bce_driven_weights, perception_supervision_mask)
        # posterior_kl_loss,c_loss = loss_model.vae_loss(fusion_z_mu, fusion_z_sca, label, label_emb,label_emb_var,label,inc_L_ind)
        # z_c_loss = loss_model.z_c_loss_new(z_sample, label, label_emb_sample,inc_L_ind)
        
        corh_loss = 0
        corh_loss = loss_model.corherent_loss(views_dis_mu[0], views_dis_var[0],common_dis_mu[0], common_dis_var[0],mask=inc_V_ind)
        if epoch>=args.pre_epochs:
            corh_loss += loss_model.corherent_loss(views_dis_mu[1], views_dis_var[1],common_dis_mu[1], common_dis_var[1],mask=inc_V_ind)
            corh_loss = corh_loss / 2
        
        UC_corh_loss = 0
        UC_corh_loss = loss_model.uni_comb_corh_loss(cross_enc_mu[0],cross_enc_var[0],views_dis_mu[0],views_dis_var[0],inc_V_ind)
        if epoch>=args.pre_epochs:
            
            UC_corh_loss += loss_model.uni_comb_corh_loss(cross_enc_mu[1],cross_enc_var[1],views_dis_mu[1],views_dis_var[1],inc_V_ind)
            UC_corh_loss = UC_corh_loss / 2
        
        loss_mse = 0
        for v in range(len(data)):
            loss_mse1 = loss_model.weighted_wmse_loss(xr_list[0][v],data[v],inc_V_ind[:,v],reduction='mean')
            if epoch>=args.pre_epochs:
                loss_mse2 = loss_model.weighted_wmse_loss(xr_list[1][v],data[v],inc_V_ind[:,v],reduction='mean')
                loss_mse += (loss_mse1 + loss_mse2)/2
            else: loss_mse += loss_mse1
            
        loss_vsp_KL = 0
        loss_vsp_KL = loss_model.view_specific_perm_KL(cross_enc_mu[0],cross_enc_var[0],cross_enc_mu[1],cross_enc_var[1],inc_V_ind)
        if epoch>=args.pre_epochs:
            loss_vsp_KL += loss_model.view_specific_perm_KL(cross_enc_mu[1],cross_enc_var[1],cross_enc_mu[0],cross_enc_var[0],inc_V_ind)
            loss_vsp_KL = loss_vsp_KL / 2
        
        loss_views_KL = 0
        loss_views_KL += loss_model.comb_views_perm_KL(views_dis_mu[0],views_dis_var[0],views_dis_mu[1],views_dis_var[1],inc_V_ind)
        if epoch>=args.pre_epochs:
            loss_views_KL += loss_model.comb_views_perm_KL(views_dis_mu[1],views_dis_var[1],views_dis_mu[0],views_dis_var[0],inc_V_ind)
            loss_views_KL = loss_views_KL / 2
        
        contras_loss = 0
        contras_loss = loss_model.contrastive_loss(views_dis_mu[0], inc_V_ind,temperature=0.1)
        
        # loss_comm_KL = loss_model.common_perm_KL(common_dis_mu[0],common_dis_var[0],common_dis_mu[1],common_dis_var[1])
        # if epoch>=args.pre_epochs:
        #     loss_comm_KL += loss_model.common_perm_KL(common_dis_mu[1],common_dis_var[1],common_dis_mu[0],common_dis_var[0])
        #     loss_comm_KL = loss_comm_KL / 2

        # loss_mse_2 = 0
        # for v in range(len(data)):
        #     loss_mse_2 += loss_model.weighted_wmse_loss(data[v],p_xr_list[v],1-inc_V_ind[:,v],reduction='mean')
        # assert torch.sum(torch.isnan(c_loss)).item() == 0
        # assert torch.sum(torch.isnan(posterior_kl_loss)).item() == 0
        
        assert torch.sum(torch.isnan(loss_mse)).item() == 0
        loss = loss_CL + loss_CL_views + loss_perception * args.theta + loss_mse * args.alpha  + loss_vsp_KL *args.beta + contras_loss * args.gamma + UC_corh_loss * args.sigma 
        # loss = loss_CL
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        # print("loss_mse",loss_mse,"loss_vsp_KL",loss_vsp_KL)
        opt.step()
        # print(model.classifier.parameters().grad)
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time() 
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses,model,All_preds,label_emb_sample

def test(loader, model, loss_model, epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind, row_index) in enumerate(loader):
        # data_time.update(time.time() - end)
        data=[v_data.clone().to('cuda:0') for v_data in data]
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        row_index = row_index.float().to('cuda:0')
        # pred,_,_ = model(data,mask=torch.ones_like(inc_V_ind).to('cuda:0'))
        latefus_weight = args.omega
        z_sample, views_dis_mu, views_dis_var, common_dis_mu, common_dis_var, xr_list, pred, label_emb_sample, label_emb,label_emb_var,_,_,_,_,_ = model(data,row_index,mask=inc_V_ind, latefus_weight=latefus_weight)
        # qc_x = vade_trick(fusion_z_mu, model.mix_prior, model.mix_mu, model.mix_sca)
        mode = 0 if epoch<args.pre_epochs else 1
        P = pred[0].cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,P.detach().numpy()),axis=0) if len(total_preds)>0 else P.detach().numpy()
        
        loss=loss_model.weighted_BCE_loss(P,label,inc_L_ind)
        
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        losses = losses,
                        ap=evaluation_results[0], 
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results

def seed_torch(seed=1029):
# 	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def main(args,file_path):
    seed_torch(42)
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' + 
                                str(args.training_sample_ratio) + '.mat')
    
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' + 
                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    device = torch.device('cuda:0')
    label_Inp_list = []
    for fold_idx in range(folds_num):
        fold_idx=fold_idx
        train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = False,num_workers=4)
        test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=4)
        val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=4)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num     
        labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')
        dep_graph = torch.matmul(labels.T,labels)
        dep_graph = dep_graph/(torch.diag(dep_graph).unsqueeze(1)+1e-10)
        # dep_graph[dep_graph<=args.sigma]=0.
        dep_graph.fill_diagonal_(fill_value=0.)
        # pri_c = train_dataset.cur_labels.sum(axis=0)/train_dataset.cur_labels.shape[0]
        # pri_c = torch.tensor(pri_c).cuda()
        model=get_model(d_list,num_classes=classes_num,z_dim=args.z_dim,adj=dep_graph,rand_seed=0)
        # print(model)
        # loss_model = Loss()
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters (M): {total_params / 1e6:.3f}M")
        print(f"Trainable parameters (M): {trainable_params / 1e6:.3f}M")
        loss_model = Loss()
        # crit = nn.BCELoss()
        # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
        optimizer = Adam(model.parameters(), lr=args.lr)
        # optimizer = Adam([{"params": model.VAE.parameters(), 'lr': args.lr},
                                                    # {"params": model.mix_mu, 'lr': args.lr}, 
                                                    # {"params": model.mix_sca, 'lr': args.lr},
                                                    # {"params": model.mix_prior, 'lr': args.lr},
                                                    # ])
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.85)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
        scheduler = None
        
        logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))
        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch=0
        best_model_dict = {'model':model.state_dict(),'epoch':0}
        for epoch in range(args.epochs):
            # tt=time.time()
            if epoch==0:
                All_preds = None
            train_losses,model,All_preds,label_emb_sample = train(train_dataloder,model,loss_model,optimizer,scheduler,epoch,All_preds,logger)
            # print("traintime:",time.time()-tt)
            label_InP = label_emb_sample.mm(label_emb_sample.t())
            # test_results = test(test_dataloder,model,loss_model,epoch,dep_graph,logger)
            # tt=time.time()
            
            # if epoch>=args.pre_epochs:
            val_results = test(val_dataloder,model,loss_model,epoch,logger)
            # print("testtime:",time.time()-tt)
            # for i,re in enumerate(epoch_results):
                # re.update(test_results[i])
            
            if val_results[0]*0.5+val_results[3]*0.5>=static_res:
                static_res = val_results[0]*0.5+val_results[3]*0.5
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch=epoch
            train_losses_last = train_losses
            total_losses.update(train_losses.sum)
        model.load_state_dict(best_model_dict['model'])
        print("epoch",best_model_dict['epoch'])
        test_results = test(test_dataloder,model,loss_model,epoch,logger)
        label_Inp_list.append(label_InP.cpu().detach().numpy())
        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,best_epoch,test_results[0],test_results[1],
            test_results[2],test_results[3]))
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Peak GPU memory allocated: {peak_memory:.2f} MB")   
        for i in range(9):
            folds_results[i].update(test_results[i])
        if args.save_curve:
            np.save(osp.join(args.curve_dir,args.dataset+'_V_'+str(args.mask_view_ratio)+'_L_'+str(args.mask_label_ratio))+'_'+str(fold_idx)+'.npy', np.array(list(zip(epoch_results[0].vals,train_losses.vals))))
    # np.save(f"mid_res/label_InP_{args.dataset}.npy",np.array(label_Inp_list))
    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP 1-HL 1-RL AUCme 1-oneE 1-Cov macAUC macro_f1 micro_f1 lr alpha beta gamma sigma theta omega\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg,3))+'+'+str(round(res.std,3)) for res in folds_results]
    res_list.extend([str(args.lr),str(args.alpha),str(args.beta),str(args.gamma),str(args.sigma),str(args.theta),str(args.omega)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()
        

def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'final_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH', 
                        default='./data/')
    parser.add_argument('--dataset', type=str, default='')#mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--name', type=str, default='10B_final_')
    # Optimization args 
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100) #200 for corel5k, iaprtc12, and espgame, 50 for pascal07 and mirflickr
    parser.add_argument('--pre_epochs', type=int, default=0)
    # Training args
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--sigma', type=float, default=0.)
    parser.add_argument('--theta', type=float, default=1e-1)


    args = parser.parse_args()
    
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [1e-3]
    alpha_list = [1e0]# reconstruction loss. 1e0 for all
    beta_list = [1e0]# consistency loss. 1e-1 for pascal07 and mirflickr，1e0 for others
    gamma_list = [1e-1] # inter loss 1e-1 for all
    sigma_list = [1e-1]# intra loss. 1e-1 for all
    theta_list = [1e0] # perception loss.  1e0 for all
    omega_list = [0.5] # late fusion weight. 0.6 for mirflickr; 0.5 for others
    if args.lr >= 0.01: 
        args.momentumkl = 0.90
    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for sigma in sigma_list:
                        args.sigma = sigma
                        for theta in theta_list:
                            args.theta = theta
                            for omega in omega_list:
                                args.omega = omega
                                for dataset in args.datasets:
                                    args.dataset = dataset
                                    file_path = osp.join(args.records_dir,args.name+args.dataset+'_VM_' + str(
                                                    args.mask_view_ratio) + '_LM_' +
                                                    str(args.mask_label_ratio) + '_T_' + 
                                                    str(args.training_sample_ratio) + '.txt')
                                    args.file_path = file_path
                                    existed_params = filterparam(file_path,[-7,-6,-5,-4,-3,-2,-1])
                                    if [args.lr,args.alpha,args.beta,args.gamma,args.sigma,args.theta,args.omega] in existed_params:
                                        print('existed param! beta:{}'.format(args.beta))
                                        # continue
                                    main(args,file_path)