import torch
import torch.nn as nn
import random
import copy
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

from model_VAE_new import VAE


def gaussian_reparameterization_std(means, std, times=1):
    std = std.abs()
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res / times


def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var + 1e-8)
    assert torch.sum(std < 0).item() == 0
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res / times


def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def fill_with_label(label_embedding, label, x_embedding, inc_V_ind):
    fea = label.matmul(label_embedding)
    new_x = x_embedding * inc_V_ind.unsqueeze(-1) + fea.unsqueeze(1)
    return new_x


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim: list = [512, 1024, 1024, 1024, 512],
        act=nn.GELU,
        norm=nn.BatchNorm1d,
        final_act=True,
        final_norm=True,
    ):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm

        self.mlps = []
        layers = []

        if len(hidden_dim) > 0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []

            for i in range(len(hidden_dim) - 1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
                layers.append(self.norm(hidden_dim[i + 1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []

            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
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
        return x


class Qc_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[1024]):
        super(Qc_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim, hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())

    def forward(self, x):
        assert torch.sum(torch.isnan(x)).item() == 0
        hidden_features = self.mlp(x)
        c_mu = self.z_loc(hidden_features)
        c_sca = self.z_sca(hidden_features)
        assert torch.sum(torch.isinf(c_mu)).item() == 0
        return c_mu, c_sca


class ActivePerception(nn.Module):
    def __init__(self, num_views, hidden_dim=None):
        super(ActivePerception, self).__init__()
        hidden_dim = hidden_dim or max(8, num_views * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(num_views),
            nn.Linear(num_views, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_views),
        )

    def forward(self, logits_views, mask):
        perception_logits = self.net(logits_views)
        view_mask = mask.float().unsqueeze(1)
        perception_logits = perception_logits.masked_fill(view_mask <= 0, -1e9)
        perception_weights = torch.softmax(perception_logits, dim=-1) * view_mask
        perception_weights = perception_weights / (perception_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return perception_logits, perception_weights


class Net(nn.Module):
    def __init__(self, d_list, num_classes, z_dim, adj, rand_seed=0):
        super(Net, self).__init__()
        self.rand_seed = rand_seed

        self.label_embedding_u = nn.Parameter(torch.eye(num_classes), requires_grad=True)
        self.label_embedding_std = nn.Parameter(torch.ones(num_classes), requires_grad=True)
        self.label_adj = nn.Parameter(torch.eye(num_classes), requires_grad=True)

        self.adj = adj
        self.z_dim = z_dim
        self.k = num_classes

        self.label_mlp = Qc_inference_mlp(num_classes, z_dim)
        self.VAE = VAE(d_list=d_list, z_dim=z_dim, class_num=num_classes)

        self.num_views = len(d_list)
        self.view_cls = nn.ModuleList([nn.Linear(z_dim, num_classes) for _ in range(self.num_views)])
        self.active_perception = ActivePerception(self.num_views)

        self.cls_conv = nn.Conv1d(num_classes, num_classes, z_dim * 2, groups=num_classes)
        self.cls = nn.Linear(z_dim, num_classes)

        self.mix_prior = None
        self.mix_mu = None
        self.mix_sca = None
        self.set_prior()
        self.cuda()

    def set_prior(self):
        self.mix_prior = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        self.mix_mu = nn.Parameter(torch.rand((self.k, self.z_dim)), requires_grad=True)
        self.mix_sca = nn.Parameter(torch.rand((self.k, self.z_dim)), requires_grad=True)

    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        nn.init.normal_(self.label_adj)
        self.cls_conv.reset_parameters()

    def poe_two(self, z_mu, z_var, c_mu, c_var, eps=1e-5):
        z_mu = z_mu.unsqueeze(1)
        z_var = z_var.unsqueeze(1)
        c_mu = c_mu.unsqueeze(0)
        c_var = c_var.unsqueeze(0)

        s_mu = torch.zeros([z_mu.shape[0], c_mu.shape[1], z_mu.shape[2]]).cuda()
        s_var = torch.ones([z_mu.shape[0], c_mu.shape[1], z_mu.shape[2]]).cuda()

        T_x = 1.0 / (z_var + eps)
        T_c = 1.0 / (c_var + eps)
        T_s = 1.0 / (s_var + eps)

        T_sum = T_x + T_c + T_s
        aggregate_mu = (z_mu * T_x + c_mu * T_c + s_mu * T_s) / T_sum
        aggregate_var = 1.0 / T_sum

        assert torch.sum(torch.isnan(aggregate_mu)).item() == 0
        assert torch.sum(torch.isinf(aggregate_mu)).item() == 0
        assert torch.sum(torch.isnan(aggregate_var)).item() == 0
        assert torch.sum(torch.isinf(aggregate_var)).item() == 0
        return aggregate_mu, aggregate_var

    def masked_late_fusion_logits(self, mu_views, var_views, mask):
        z_views = gaussian_reparameterization_var(mu_views, var_views, times=1)

        logits_views = []
        for v in range(self.num_views):
            logits_views.append(self.view_cls[v](z_views[v]))
        logits_views = torch.stack(logits_views, dim=-1)  # [B, C, M]
        preds_views = torch.sigmoid(logits_views)

        perception_logits, perception_weights = self.active_perception(logits_views, mask)
        late_logits = (logits_views * perception_weights).sum(dim=-1)

        return late_logits, preds_views, perception_weights, perception_logits

    def forward(self, x_list, row_index, mask, latefus_weight=0.2):
        label_embedding = self.label_embedding_u
        label_embedding_sample = self.label_embedding_u
        label_embedding_var = self.label_embedding_u

        prop = 0.25
        if self.training is True:
            for i, X in enumerate(x_list):
                bit_mask_len = int(prop * X.size(-1))
                bit_mask = torch.ones_like(X)
                for j in range(mask.shape[0]):
                    zero_indices = torch.randperm(bit_mask.shape[1])[:bit_mask_len]
                    bit_mask[j, zero_indices] = 0
                x_list[i] = x_list[i].mul(bit_mask)

        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, cross_enc_mu, cross_enc_var = self.VAE(
            x_list, row_index, mask
        )

        pred_mid_logits = self.cls(z_sample[0])
        pred_mid_perm_logits = self.cls(z_sample[1])

        late_logits, preds_views, perception_weights, _ = self.masked_late_fusion_logits(uniview_mu_list[0], uniview_sca_list[0], mask)
        late_perm_logits, preds_views_perm, perception_weights_perm, _ = self.masked_late_fusion_logits(uniview_mu_list[1], uniview_sca_list[1], mask)
        midfus_weight = 1 - latefus_weight
        
        pred = torch.sigmoid(pred_mid_logits*midfus_weight  + late_logits*latefus_weight)
        pred_perm = torch.sigmoid(pred_mid_perm_logits*midfus_weight + late_perm_logits*latefus_weight)

        vis_sampling = gaussian_reparameterization_var(
            cross_enc_mu[0].diagonal(dim1=1, dim2=2),
            cross_enc_var[0].diagonal(dim1=1, dim2=2),
        ).transpose(1, 2)
        pred_diag = self.cls(vis_sampling)

        return (
            z_sample,
            uniview_mu_list,
            uniview_sca_list,
            fusion_z_mu,
            fusion_z_sca,
            xr_list,
            [pred, pred_perm],
            label_embedding_sample,
            label_embedding,
            label_embedding_var,
            cross_enc_mu,
            cross_enc_var,
            pred_diag,
            [preds_views.permute(2, 0, 1), preds_views_perm.permute(2, 0, 1)],
            [perception_weights, perception_weights_perm],
        )


def get_model(d_list, num_classes, z_dim, adj, rand_seed=0):
    model = Net(d_list, num_classes=num_classes, z_dim=z_dim, adj=adj, rand_seed=rand_seed)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


if __name__ == "__main__":
    from MLdataset import getIncDataloader

    dataloder, dataset = getIncDataloader(
        "/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat",
        "/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0_LabelMaskRatio_0_TraindataRatio_0.7.mat",
        training_ratio=0.7,
        mode="train",
        batch_size=3,
        num_workers=2,
    )
    input = next(iter(dataloder))[0]
    model = get_model(d_list=dataset.d_list, num_classes=dataset.classes_num, z_dim=512, adj=torch.eye(dataset.classes_num).cuda(), rand_seed=0)
    print(model)
    input = [v_data.to("cuda:0") for v_data in input]
    out = model(input, row_index=torch.zeros(input[0].shape[0], len(input), len(input)).cuda(), mask=torch.ones(input[0].shape[0], len(input)).cuda())
    print(out[6][0].shape)