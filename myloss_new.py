import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var)
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def W_distance(p_mu,p_v,q_mu,q_v,eps=1e-9):
    p_var = p_v+eps
    q_var = q_v+eps
    res = torch.sum((p_mu-q_mu)**2,dim=-1)+p_var.sum(-1)+q_var.sum(-1)- 2*((p_var*q_var)**0.5).sum(-1)
    return res
def gaussian_log_var(x, mu, var, eps=1e-14):
    return -0.5 * (torch.log(torch.tensor(2.0 * np.pi)) + torch.log(var) + torch.pow(x - mu, 2) / var)

def kl_div_var(q_mu, q_var, p_mu, p_var, eps=1e-12):
    
    return 0.5 * (torch.log((p_var+eps) / (q_var+eps)) + (q_var+eps) / (p_var+eps) + torch.pow(q_mu - p_mu, 2) / (p_var+eps) - 1)

def cosdis(x1,x2):
    return (1-torch.cosine_similarity(x1,x2,dim=-1))/2

def vade_trick(mc_sample, gmm_pi, gmm_mu, gmm_sca):
    
        log_pz_c = torch.mean(gaussian_log_var(mc_sample.unsqueeze(1), gmm_mu.unsqueeze(0), gmm_sca.unsqueeze(0)), dim=-1)

        log_pc = torch.log(gmm_pi.unsqueeze(0))
        log_pc_z = log_pc + log_pz_c
        pc_z = torch.exp(log_pc_z) + 1e-10
        normalize_pc_z = pc_z / torch.sum(pc_z, dim=1, keepdim=True)
        return normalize_pc_z

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    
    def compact_loss(self,u_list,var_list,mask): 
        loss = 0
        view_num = len(u_list)
        for i in range(0,len(u_list)):
            for j in range(i+1,len(u_list)):
                
                # kl =  kl_div_var(u_list[i].detach(),var_list[i].detach(),u_list[j],var_list[j])
                # assert torch.sum(torch.isinf(kl)).item() == 0
                # kl_loss = kl * var_list[j]
                # kl_loss = kl * (mask[:,i]*mask[:,j]).unsqueeze(-1)
                
                kl = (u_list[i].detach()-u_list[j])**2
                kl = kl*var_list[j]
                kl_loss = kl*(mask[:,i]*mask[:,j]).unsqueeze(-1)
                if torch.sum(torch.isnan(kl_loss)).item() > 0:
                    pass
                # loss += kl_loss.sum(dim=-1).mean()
                loss += kl_loss.mean()
        assert torch.sum(torch.isnan(loss)).item() == 0
        loss = loss/view_num**2

        return loss
    def vae_loss_new(self,z_mu,z_sca, qc_x,  c_mu,c_sca,p_c,inc_L_ind):
        
        z_sam = gaussian_reparameterization_var(z_mu,z_sca)
        c_sam = gaussian_reparameterization_var(c_mu,c_sca)
        logsoft_qz_x = (z_sam.softmax(-1)+1e-5).log().detach()
        soft_qz_c = c_sam.softmax(-1)+1e-5
        # assert torch.sum(torch.isnan(kl_xc)).item() == 0
    

        kl_xc = torch.abs(F.kl_div(logsoft_qz_x.unsqueeze(1),soft_qz_c.unsqueeze(0),reduction='none'))
        if torch.sum(torch.isnan(kl_xc)).item() > 0:
            print('error')
        assert torch.sum(torch.isnan(kl_xc)).item() == 0
        
        z_kl_loss = torch.mean((qc_x.unsqueeze(-1) * kl_xc).sum((1,2)))


        # print((qc_x<0).sum())
        assert torch.sum(torch.isnan(z_kl_loss)).item() == 0
        if len(p_c.shape)==1: 
            p_c = p_c.unsqueeze(0)
        c_kl_loss = torch.mean((qc_x * torch.log((qc_x / (p_c+1e-5)+1e-5))).abs())
        
        assert torch.sum(torch.isnan(c_kl_loss)).item() == 0
        # # print((c_kl_loss<0).sum())
        # c_kl_loss_mean = torch.mean(c_kl_loss)
        return z_kl_loss,c_kl_loss
    def z_c_loss(self,z_mu, label,  c_mu,inc_L_ind):
        loss = ((z_mu.unsqueeze(1)-c_mu.unsqueeze(0))**2).mean(dim=-1)

        loss = (loss*(label*inc_L_ind)).sum(-1)/((label*inc_L_ind).sum(-1)+1e-9)
        return loss.mean()

        
    
    def z_c_loss2(self,z_mu,z_sca, label,  c_mu,c_sca,inc_L_ind):
        # loss = (z_mu-(label*inc_L_ind).matmul(c_mu))**2
        loss = W_distance(z_mu.unsqueeze(1),z_sca.unsqueeze(1),c_mu.unsqueeze(0),c_sca.unsqueeze(0))
        loss = loss*(label*inc_L_ind)
        return loss.mean()
    def z_c_loss3(self,z_mu, label,  c_mu,inc_L_ind):
        # loss = (z_mu-(label*inc_L_ind).matmul(c_mu))**2
        loss = torch.abs(z_mu.unsqueeze(1)-c_mu.unsqueeze(0)).sum(-1)
        loss = (loss*(label*inc_L_ind)).sum(-1)/((label*inc_L_ind).sum(-1)+1e-9)
        return loss.sum()/z_mu.shape[0]
    
    def vae_loss(self,z_mu,z_sca, qc_x,  c_mu,c_sca,p_c,inc_L_ind):
        kl_xc = kl_div_var(z_mu.unsqueeze(1), z_sca.unsqueeze(1), c_mu.unsqueeze(0), c_sca.unsqueeze(0))
        # z_kl_loss = torch.mean(qc_x * torch.mean(kl, dim=-1),
        #                     dim=1)
        # z_kl_loss = torch.mean(z_kl_loss)

        # print(qc_x.log().sigmoid().unsqueeze(-1).shape,qz_c.log().sigmoid().unsqueeze(0).shape)
        # if torch.sum(torch.isnan(kl_xc)).item() > 0 or torch.sum(torch.isnan(kl_xc)).item()>0:
        #     assert torch.sum(torch.isnan(kl_xc)).item() == 0
        #     assert torch.sum(torch.isnan(kl_xc)).item() == 0

        # assert torch.sum(torch.isnan(kl_xc)).item() == 0

        if torch.sum(torch.isnan(kl_xc)).item() > 0:
            print('error')
        assert torch.sum(torch.isnan(kl_xc)).item() == 0
        
        z_kl_loss = torch.mean((qc_x.unsqueeze(-1) * kl_xc).sum((1,2)))


        # print((qc_x<0).sum())
        assert torch.sum(torch.isnan(z_kl_loss)).item() == 0
        if len(p_c.shape)==1: 
            p_c = p_c.unsqueeze(0)
        c_kl_loss = torch.mean((qc_x * torch.log((qc_x / (p_c+1e-5)+1e-5))).abs())
        
        assert torch.sum(torch.isnan(c_kl_loss)).item() == 0
        # # print((c_kl_loss<0).sum())
        # c_kl_loss_mean = torch.mean(c_kl_loss)
        return z_kl_loss,c_kl_loss


    def tc_kl_loss(self,uniview_dist_mu, uniview_dist_sca, aggregate_mu, aggregate_sca, mask=None):
        z_tc_loss = []
        for v in range(len(uniview_dist_mu)):
            zv_tc_loss = torch.sum(kl_div_var(aggregate_mu, aggregate_sca, uniview_dist_mu[v], uniview_dist_sca[v]),
                                dim=1)
            exist_loss = zv_tc_loss * mask[v].squeeze()
            zv_tc_loss_mean = torch.sum(exist_loss)
            z_tc_loss.append(zv_tc_loss_mean)
        tc_term_loss = sum(z_tc_loss) / aggregate_mu.shape[0]

        return tc_term_loss

    def corherent_loss2(self,uniview_dist_mu, uniview_dist_sca, aggregate_mu, aggregate_sca, mask=None):
        if mask is None:
            mask = torch.ones_like((aggregate_mu.shape[0],len(uniview_dist_mu))).to(aggregate_mu.device)

        z_tc_loss = []
        # mask_stack = torch.concatenate(mask, dim=1)
        norm = torch.sum(mask, dim=1)
        for v in range(len(uniview_dist_mu)):
            zv_tc_loss = torch.mean((aggregate_mu - aggregate_sca)**2,dim=1)
            exist_loss = zv_tc_loss * mask[:,v]
            z_tc_loss.append(exist_loss)
        z_tc_loss = torch.stack(z_tc_loss,dim=1)

        sample_ave_tc_term_loss = torch.sum(z_tc_loss) / mask.sum()

        return sample_ave_tc_term_loss
    def corherent_loss(self,uniview_dist_mu, uniview_dist_sca, aggregate_mu, aggregate_sca, mask=None):
        if mask is None:
            mask = torch.ones_like((aggregate_mu.shape[0],len(uniview_dist_mu))).to(aggregate_mu.device)

        z_tc_loss = []
        # mask_stack = torch.concatenate(mask, dim=1)
        # norm = torch.sum(mask, dim=1)
        # weight = F.softmax(torch.stack(uniview_dist_sca,dim=1),dim=1)
        for v in range(len(uniview_dist_mu)):
            # zv_tc_loss = torch.mean(kl_div_var(aggregate_mu, aggregate_sca, uniview_dist_mu[v], uniview_dist_sca[v])*weight[:,v,:],
            #                     dim=1)
            zv_tc_loss = torch.mean(kl_div_var(aggregate_mu, aggregate_sca, uniview_dist_mu[v], uniview_dist_sca[v]),
                                dim=1)
            exist_loss = zv_tc_loss * mask[:,v]
            z_tc_loss.append(exist_loss)
        z_tc_loss = torch.stack(z_tc_loss,dim=1)

        sample_ave_tc_term_loss = torch.sum(z_tc_loss) / mask.sum()
        return sample_ave_tc_term_loss
    
    def uni_comb_corh_loss(self,cross_enc_mu,cross_enc_var,mu_comb_views,var_comb_views,mask):
        #cross_enc_mu [b m m d]
        #mu_comb_views [m b d]
        #mask [b m]
        num_views = mu_comb_views.shape[0]
        batch_size = mu_comb_views.shape[1]
        dim = mu_comb_views.shape[2]
        mu0 = cross_enc_mu.view(-1,cross_enc_mu.shape[-1])
        var0 = cross_enc_var.view(-1,cross_enc_mu.shape[-1])

        mu1 = mu_comb_views.permute(1,0,2).unsqueeze(2).repeat(1,1,num_views,1).view(-1,mu_comb_views.shape[-1])
        var1 = var_comb_views.permute(1,0,2).unsqueeze(2).repeat(1,1,num_views,1).view(-1,var_comb_views.shape[-1])
        KL = kl_div_var(mu1,var1,mu0,var0).view(batch_size, num_views, num_views, dim) #[b m m d]
        KL = torch.mean(KL,dim=(2,3))
        KL = KL * mask
        return KL.sum()/mask.sum()
    def contrastive_loss(self, input_mu, mask_v, temperature=None):
        if temperature is None:
            temperature = self.temperature
        shared_mu = torch.transpose(input_mu,0,1)
        batch_size, n_view, d_emb = shared_mu.shape
        total_loss = 0
        valid_pairs = 0

        for i in range(n_view):
            for j in range(i + 1, n_view):
                valid_mask = (mask_v[:, i] > 0) & (mask_v[:, j] > 0)

                valid_indices = torch.where(valid_mask)[0]

                if len(valid_indices) < 2:
                    continue

                anchor = shared_mu[valid_indices, i, :]
                positive = shared_mu[valid_indices, j, :]
                # weights_i = weights[valid_indices, i, :]
                # weights_j = weights[valid_indices, j, :]
                # print("weights:",weights.min(),weights.max())
                anchor = F.normalize(anchor, p=2, dim=1)
                positive = F.normalize(positive, p=2, dim=1)
                # losses = ((anchor-positive)**2).sum(-1)
                
                all_sim = torch.mm(anchor, positive.t()) / temperature
                # all_sim *= torch.mm(weights_i, weights_j.t())

                pos_sim = torch.diag(all_sim)
                log_sum_exp = torch.logsumexp(all_sim, dim=1)
                losses = -pos_sim + log_sum_exp
                # losses = -pos_sim
                
                total_loss += torch.sum(losses)

                valid_pairs += len(valid_indices)
        
        return total_loss / max(valid_pairs, 1) if valid_pairs > 0 else torch.tensor(0.0, device=shared_mu.device)
        
    def view_specific_perm_KL(self, p_mu, p_var,q_mu, q_var, inc_ind_V):
        z_dim = q_mu.shape[-1]
        mu1, mu0 = q_mu.view(-1, z_dim), p_mu.view(-1, z_dim)
        var1, var0 = q_var.view(-1, z_dim), p_var.view(-1, z_dim)
        KLV = torch.mean(kl_div_var(mu1, var1, mu0, var0), dim=-1)

        KLV = KLV.view(q_mu.shape[0],q_mu.shape[1],q_mu.shape[1]).mean(dim=-1)# [b m] 
        mask = (inc_ind_V/inc_ind_V.sum(dim=-1,keepdim=True))
        KLV = KLV * mask
        return KLV.sum()/mask.shape[0]
    def common_perm_KL(self, p_mu, p_var, q_mu, q_var):
        # z_dim = q_mu.shape[-1]
        # mu0, mu1 = q_mu.view(-1, z_dim), p_mu.view(-1, z_dim)
        # var0, var1 = q_var.view(-1, z_dim), p_var.view(-1, z_dim)
        KLV = torch.mean(kl_div_var(q_mu, q_var, p_mu, p_var))

        return KLV
    
    def comb_views_perm_KL(self, p_mu, p_var, q_mu, q_var, inc_ind_V):
        # p_mu [m b d]
        # inc_ind_V [b m]
        z_dim = q_mu.shape[-1]
        mask = inc_ind_V.T.contiguous().view(-1)
        mu1, mu0 = q_mu.view(-1, z_dim), p_mu.view(-1, z_dim)
        var1, var0 = q_var.view(-1, z_dim), p_var.view(-1, z_dim)
        KLV = torch.mean(kl_div_var(mu1, var1, mu0, var0), dim=-1)
        KLV = KLV * mask
        return KLV.sum()/inc_ind_V.sum()
    
    def weighted_BCE_loss(self,pred,label,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(pred))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - pred + 1e-5))).item() == 0
        res=torch.abs((label.mul(torch.log(pred + 1e-5)) \
                                                + (1-label).mul(torch.log(1 - pred + 1e-5))).mul(inc_L_ind))
        assert torch.sum(torch.isnan(res)).item() == 0
        assert torch.sum(torch.isinf(res)).item() == 0

        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res
            
    def weighted_wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        if torch.sum(torch.isnan(ret)).item()>0:
            print(ret)
        if reduction == 'mean':
            return torch.mean(ret)
        elif reduction=='sum':
            return torch.sum(ret)
        elif reduction=='none':
            return ret


    
    