import torch
import wandb
import math

import numpy as np
import torch.nn as nn

from torch import linalg as LA
from torch.distributions.multivariate_normal import MultivariateNormal
from model_disentgnn import mean_tensors
from torchmetrics.regression import PearsonCorrCoef



def l2_loss(anchors, positives):
    """
    Calculates L2 norm regularization loss
    :param anchors: A torch.Tensor, (n, embedding_size)
    :param positives: A torch.Tensor, (n, embedding_size)
    :return: A scalar
    """
    return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]


def pair_loss(s_pred, s_pred_label_change, s_pred_label_unchange):
    s_pred_unsqueeze = torch.unsqueeze(s_pred, dim=1)
    s_pred_label_change_unsqueeze = torch.unsqueeze(s_pred_label_change, dim=1)
    s_pred_label_unchange_unsqueeze = torch.unsqueeze(s_pred_label_unchange, dim=1)
    res = torch.matmul(s_pred_unsqueeze, (s_pred_label_unchange_unsqueeze - s_pred_label_change_unsqueeze).transpose(1, 2))
    res = torch.sum(torch.exp(res), 2)
    loss = torch.mean(torch.log(torch.tensor(1) + res))
    return loss


def pair_loss_cos(s_pred, s_pred_label_change, s_pred_label_unchange):
    temperature = 0.05
    cos_similarity = nn.CosineSimilarity(dim=1)
    cos_distance_close = cos_similarity(s_pred, s_pred_label_change) / temperature
    cos_distance_far = cos_similarity(s_pred, s_pred_label_unchange) / temperature
    res = torch.exp(cos_distance_far - cos_distance_close)
    loss = torch.mean(torch.log(torch.tensor(1) + res))
    return loss


def pair_loss_triplet(s_pred, s_pred_label_change, s_pred_label_unchange):
    # triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity())

    loss = triplet_loss(s_pred, s_pred_label_change, s_pred_label_unchange)
    return loss


def pearson_loss(s_pred, s_pred_label_change, s_pred_label_unchange):
    pearson_loss = PearsonCorrCoef(num_outputs=s_pred.shape[1]).to(s_pred.get_device())
    loss_close = pearson_loss(s_pred, s_pred_label_change).mean()
    loss_far = pearson_loss(s_pred, s_pred_label_unchange).maan()
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=pearson_loss)

    loss = triplet_loss(s_pred, s_pred_label_change, s_pred_label_unchange)
    return loss


class VAECriterion(nn.Module):
    def __init__(self, lambda_e, lambda_od, gamma_e, gamma_od, step_size):
        super(VAECriterion, self).__init__()
        self.lambda_e = lambda_e
        self.lambda_od = lambda_od
        self.gamma_e = gamma_e
        self.gamma_od = gamma_od
        self.step_size = step_size

        self.cross = nn.BCEWithLogitsLoss()
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

    def orth_kld(self, mean_t, mean_s, log_std_t, log_std_s):
        mean_1, mean_2 = mean_tensors(np.ones(16), np.zeros(16), 13)

        m_t = MultivariateNormal(torch.zeros(16).to(self.device), torch.eye(16).to(self.device))
        m_s = MultivariateNormal(torch.zeros(16).to(self.device), torch.eye(16).to(self.device))

        # Loss_e = L_e(s_zt)
        prior_t = []
        prior_s = []
        enc_dis_t = []
        enc_dis_s = []

        for i in range(mean_t.shape[0]):
            prior_t.append(m_t.sample())
            prior_s.append(m_s.sample())
            n_t = MultivariateNormal(mean_t[i], torch.diag(torch.exp(log_std_t[i]) + torch.tensor(1e-15).to(self.device)))
            n_s = MultivariateNormal(mean_s[i], torch.diag(torch.exp(log_std_s[i]) + torch.tensor(1e-15).to(self.device)))
            enc_dis_t.append(n_t.sample())
            enc_dis_s.append(n_s.sample())

        prior_t = torch.stack(prior_t).to(self.device)
        prior_s = torch.stack(prior_s).to(self.device)
        enc_dis_t = torch.stack(enc_dis_t).to(self.device)
        enc_dis_s = torch.stack(enc_dis_s).to(self.device)

        L_zt = self.kld(torch.log_softmax(prior_t, dim=1), torch.softmax(enc_dis_t, dim=1))
        L_zs = self.kld(torch.log_softmax(prior_s, dim=1), torch.softmax(enc_dis_s, dim=1))
        return L_zt + L_zs


    def orth_kld_z(self, mean_t, mean_s, z1, z2):
        mean_1, mean_2 = mean_tensors(np.ones(16), np.zeros(16), 13)
        m_t = MultivariateNormal(mean_1.to(self.device), torch.eye(16).to(self.device))
        m_s = MultivariateNormal(mean_2.to(self.device), torch.eye(16).to(self.device))

        # Loss_e = L_e(s_zt)
        prior_t = []
        prior_s = []
        # enc_dis_t = []
        # enc_dis_s = []

        for i in range(mean_t.shape[0]):
            prior_t.append(m_t.sample())
            prior_s.append(m_s.sample())

        prior_t = torch.stack(prior_t).to(self.device)
        prior_s = torch.stack(prior_s).to(self.device)

        L_zt = self.kld(torch.log_softmax(prior_t, dim=1), torch.softmax(z1, dim=1))
        L_zs = self.kld(torch.log_softmax(prior_s, dim=1), torch.softmax(z2, dim=1))
        # inner product of z1 and z2
        # learned mean std from encoder
        # change vae prior to normal distribution
        return L_zt + L_zs

    def forward(self, model_output, target):
        # mean_t, mean_s, log_std_t, log_std_s = model_output[0]
        # y_zt, s_zt, s_zs = model_output[1]
        #
        # z1, z2 = model_output[2]

        # y_pred, s_pred, s_pred_label_change, s_pred_label_unchange, mean_t_list, mean_s_list, \
        #     log_var_t_list, log_var_s_list, z_1_list, z_2_list = model_output

        y_pred, s_pred, s_pred_label_change, s_pred_label_unchange = model_output

        # ignore nan targets (unlabeled) when computing training loss.
        is_labeled = target == target
        loss_target = self.cross(y_pred.to(self.device)[is_labeled], target.to(self.device)[is_labeled])
        loss_close_cos = 10 * self.cos_loss(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                            torch.ones(len(s_pred)).to(self.device))
        loss_far_cos = 10 * self.cos_loss(s_pred.to(self.device), s_pred_label_unchange.to(self.device),
                                          torch.ones(len(s_pred)).to(self.device))


        Loss = loss_target + loss_close_cos + loss_far_cos

        # wandb.log({'Loss target': L_t, 'Loss e': L_t, 'KL loss zt': L_zt, 'KL loss zs': L_zs})
        wandb.log({'Loss target': loss_target, 'Loss close cos': loss_close_cos, 'Loss far cos': loss_far_cos})

        return Loss


class DisentCriterion(nn.Module):
    def __init__(self, criterion_name, beta_set):
        super(DisentCriterion, self).__init__()

        self.cross = nn.BCEWithLogitsLoss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion_name = criterion_name
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.beta1, self.beta2, self.beta3 = beta_set

    def reconstruction_loss_max(self, z_1_ori, z_2_ori, z_1_label_change, z_2_label_change,
                            z_1_label_unchange, z_2_label_unchange):
        c_matrix = torch.cat((z_1_ori, z_1_label_change, z_1_label_unchange), dim=0)
        s_matrix = torch.cat((z_2_ori, z_2_label_change, z_2_label_unchange), dim=0)

        c_temp = torch.matmul(c_matrix.transpose(0, 1), c_matrix)
        c_temp_inv = torch.linalg.inv(c_temp)
        res = torch.matmul(c_matrix, c_temp_inv)
        res = torch.matmul(res, c_matrix.transpose(0, 1))
        res = torch.matmul(res, s_matrix)
        res = LA.matrix_norm(res - s_matrix)

        res = 1 / torch.square(res + 1e-10)

        # res = 1 / (res + 1e-10)
        # res = - res

        return res


    def forward(self, model_output, target, epoch):
        y_pred, s_pred, s_pred_label_change, s_pred_label_unchange, z_1_ori, z_2_ori, z_1_label_change, \
            z_2_label_change, z_1_label_unchange, z_2_label_unchange, y_pred_label_unchange = model_output

        # for baseline gnn
        # y_pred = model_output


        # ignore nan targets (unlabeled) when computing training loss.
        is_labeled = target == target
        # print("target: ", target)
        # print("y_pred: ", y_pred.shape)
        loss_target = self.cross(y_pred.to(self.device)[is_labeled], target.to(self.device)[is_labeled])
        # loss_target_label_unchange = 0
        loss_target_label_unchange = self.cross(y_pred_label_unchange.to(self.device)[is_labeled], target.to(self.device)[is_labeled])

        loss_target_label_unchange_coef = 0.15

        loss_target_total = self.beta2 * loss_target + loss_target_label_unchange_coef * loss_target_label_unchange

        reconstruction_coef_max = self.beta3

        loss_reconstruction_1 = reconstruction_coef_max * self.reconstruction_loss_max(z_1_ori, z_2_ori, z_1_label_change, z_2_label_change,
                                z_1_label_unchange, z_2_label_unchange)

        loss_reconstruction = loss_reconstruction_1

        # wandb.log({'Loss reconstruction max': loss_reconstruction_1, 'Loss reconstruction min': loss_reconstruction_2})
        wandb.log({'Loss reconstruction': loss_reconstruction})

        if self.criterion_name == "pair_loss_triplet":

            loss_pair = 0.05 * pair_loss_triplet(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                                 s_pred_label_unchange.to(self.device))
            Loss = loss_target_total + loss_pair
            wandb.log({'Loss target': loss_target_total, 'Loss pair': loss_pair, 'Loss KLD': kld_loss})

        elif self.criterion_name == "pair_loss":
            loss_pair = 0.1 * pair_loss(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                        s_pred_label_unchange.to(self.device))
            loss_l2 = 0.01 * l2_loss(s_pred.to(self.device), s_pred_label_change.to(self.device))
            Loss = loss_target_total + loss_pair + loss_l2
            wandb.log({'Loss target': loss_target_total, 'Loss pair': loss_pair, 'Loss l2': loss_l2})

        elif self.criterion_name == "pair_loss_cos":

            coef = self.beta1

            loss_pair_cos = coef * pair_loss_cos(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                                 s_pred_label_unchange.to(self.device))

            Loss = loss_target_total + loss_pair_cos + loss_reconstruction
            # Loss = loss_target
            wandb.log({'Loss target': loss_target_total, 'Loss pair cos': loss_pair_cos})

        elif self.criterion_name == "cos_loss":
            loss_close_cos = 1 * self.cos_loss(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                               torch.ones(len(s_pred)).to(self.device))
            loss_far_cos = 1 * self.cos_loss(s_pred.to(self.device), s_pred_label_unchange.to(self.device),
                                             (torch.zeros(len(s_pred)) - 1).to(self.device))
            Loss = loss_target_total + loss_close_cos + loss_far_cos

            wandb.log({'Loss target': loss_target_total, 'Loss close cos': loss_close_cos, 'Loss far cos': loss_far_cos})

        elif self.criterion_name == "pearson_loss":
            loss_pearson = pearson_loss(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                        s_pred_label_unchange.to(self.device))
            Loss = loss_target_total + loss_pearson
            wandb.log({'Loss pearson': loss_pearson, 'Loss target': loss_target_total})

        elif self.criterion_name == "none":
            Loss = loss_target_total
            wandb.log({'Loss target': loss_target_total})
        else:
            raise NotImplementedError("Criterion not implemented!")
        # wandb.log({'Loss target': L_t, 'Loss e': L_t, 'KL loss zt': L_zt, 'KL loss zs': L_zs})
        return Loss

