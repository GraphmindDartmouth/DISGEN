import torch
import wandb
import math

import numpy as np
import torch.nn as nn

from torch import linalg as LA
from torch.distributions.multivariate_normal import MultivariateNormal
from quick_test_vae_model import mean_tensors
from torchmetrics.regression import PearsonCorrCoef



def L_e(sen_dis_out):
    # L_e = -torch.sum(torch.softmax(sen_dis_out, dim=1) * torch.log_softmax(sen_dis_out, dim=1)) / sen_dis_out.shape[0]
    L_e = torch.sum(torch.softmax(sen_dis_out, dim=1) * torch.log_softmax(sen_dis_out, dim=1)) / sen_dis_out.shape[0]

    return L_e


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
        # print("Mean1: ", mean_1)
        # print("Mean2: ", mean_2)

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

        # loss_kld_ori = self.orth_kld_z(mean_t_list[0], mean_s_list[0], z_1_list[0], z_2_list[0])
        # loss_kld_ori = self.orth_kld(mean_t_list[0], mean_s_list[0], log_var_t_list[0], log_var_s_list[0])
        # loss_kld_label_change = self.orth_kld(mean_t_list[1], mean_s_list[1], log_var_t_list[1], log_var_s_list[1])
        # loss_kld_label_unchange = self.orth_kld(mean_t_list[2], mean_s_list[2], log_var_t_list[2], log_var_s_list[2])
        loss_kld_label_change = 0
        loss_kld_label_unchange = 0

        # loss_kld = 0
        # loss_kld = 0.05 * loss_kld_ori + loss_kld_label_change + loss_kld_label_unchange
        # mean_1, mean_2 = mean_tensors(np.ones(16), np.zeros(16), 13)
        # m_t = MultivariateNormal(mean_1.to(self.device), torch.eye(16).to(self.device))
        # m_s = MultivariateNormal(mean_2.to(self.device), torch.eye(16).to(self.device))

        # Loss_e = L_e(s_zt)
        # prior_t = []
        # prior_s = []
        # enc_dis_t = []
        # enc_dis_s = []
        #
        # for i in range(z1.shape[0]):
        #     prior_t.append(m_t.sample())
        #     prior_s.append(m_s.sample())
        #     n_t = MultivariateNormal(mean_t[i], torch.diag(torch.exp(log_std_t[i])))
        #     n_s = MultivariateNormal(mean_s[i], torch.diag(torch.exp(log_std_s[i])))
        #     enc_dis_t.append(n_t.sample())
        #     enc_dis_s.append(n_s.sample())
        #
        # prior_t = torch.stack(prior_t).to(self.device)
        # prior_s = torch.stack(prior_s).to(self.device)
        # enc_dis_t = torch.stack(enc_dis_t).to(self.device)
        # enc_dis_s = torch.stack(enc_dis_s).to(self.device)

        # L_zt = self.kld(torch.log_softmax(prior_t, dim=1), torch.softmax(enc_dis_t, dim=1))
        # L_zs = self.kld(torch.log_softmax(prior_s, dim=1), torch.softmax(enc_dis_s, dim=1))

        # lambda_e = self.lambda_e * self.gamma_e ** (current_step / self.step_size)
        # lambda_od = self.lambda_od * self.gamma_od ** (current_step / self.step_size)

        # Loss = L_t + self.lambda_e * Loss_e + self.lambda_od * (L_zt + L_zs)
        # Loss = loss_target + loss_close_cos + loss_far_cos + loss_kld
        Loss = loss_target + loss_close_cos + loss_far_cos

        # wandb.log({'Loss target': L_t, 'Loss e': L_t, 'KL loss zt': L_zt, 'KL loss zs': L_zs})
        wandb.log({'Loss target': loss_target, 'Loss close cos': loss_close_cos, 'Loss far cos': loss_far_cos})

        return Loss


class DisentCriterion(nn.Module):
    def __init__(self, criterion_name):
        super(DisentCriterion, self).__init__()

        self.cross = nn.BCEWithLogitsLoss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion_name = criterion_name
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def kld_uniform_loss(self, y_pred_z2_ori, epoch, feed_sensitive_info):
        # if feed_sensitive_info:
        #     prior_uniform = torch.zeros((y_pred_z2_ori.shape[0], y_pred_z2_ori.shape[1])).to(self.device).squeeze()
        # else:
        #     prior_uniform = torch.rand((y_pred_z2_ori.shape[0], y_pred_z2_ori.shape[1])).to(self.device).squeeze()
        #     prior_uniform = (prior_uniform - 1/2) / math.sqrt(1/12)

        prior_uniform = torch.rand((y_pred_z2_ori.shape[0], y_pred_z2_ori.shape[1])).to(self.device).squeeze()
        # prior_uniform = torch.ones((y_pred_z2_ori.shape[0], y_pred_z2_ori.shape[1])).to(self.device).squeeze() / 2
        # prior_uniform = torch.zeros((y_pred_z2_ori.shape[0], y_pred_z2_ori.shape[1])).to(self.device).squeeze()

        # if epoch == 1 or epoch == 90:
        #     print("Prior: ", prior_uniform)
        #     print("Input: ", y_pred_z2_ori)
        # prior_uniform = (torch.ones((y_pred_z2_ori.shape[0], y_pred_z2_ori.shape[1])) / 2).to(self.device).squeeze()

        kld_loss = self.kld(torch.log_softmax(prior_uniform, dim=0), torch.softmax(y_pred_z2_ori.squeeze(), dim=0))
        return kld_loss


    def reconstruction_loss_max(self, z_1_ori, z_2_ori, z_1_label_change, z_2_label_change,
                            z_1_label_unchange, z_2_label_unchange):
        c_matrix = torch.cat((z_1_ori, z_1_label_change, z_1_label_unchange), dim=0)
        s_matrix = torch.cat((z_2_ori, z_2_label_change, z_2_label_unchange), dim=0)

        c_temp = torch.matmul(c_matrix.transpose(0, 1), c_matrix)
        c_temp_inv = torch.linalg.inv(c_temp)
        res = torch.matmul(c_matrix, c_temp_inv)
        res = torch.matmul(res, s_matrix.transpose(0, 1))
        res = torch.matmul(res, c_matrix)
        res = LA.matrix_norm(res - s_matrix)

        # gin graphsst2
        res = 1 / torch.square(res + 1e-10)

        # other datasets
        # res = 1 / (res + 1e-10)
        # res = - res

        return res

    # def reconstruction_loss_min(self, z_1_ori, z_1_label_unchange):
    #     c_matrix = torch.cat((z_1_ori, z_1_ori, z_1_ori), dim=0)
    #     s_matrix = torch.cat((z_1_label_unchange, z_1_label_unchange, z_1_label_unchange), dim=0)
    #
    #     c_temp = torch.matmul(c_matrix.transpose(0, 1), c_matrix)
    #     c_temp_inv = torch.linalg.inv(c_temp)
    #     res = torch.matmul(c_matrix, c_temp_inv)
    #     res = torch.matmul(res, s_matrix.transpose(0, 1))
    #     res = torch.matmul(res, c_matrix)
    #     res = LA.matrix_norm(res - s_matrix)
    #
    #     return res

    def forward(self, model_output, target, epoch):
        # for multiview gnn
        # y_pred, s_pred, s_pred_label_change, s_pred_label_unchange, y_pred_z2_ori_nn1, y_pred_z1_ori_nn2 = model_output
        # y_pred = model_output
        # for multiview gnn with reconstruction loss
        y_pred, s_pred, s_pred_label_change, s_pred_label_unchange, z_1_ori, z_2_ori, z_1_label_change, \
            z_2_label_change, z_1_label_unchange, z_2_label_unchange, y_pred_label_unchange = model_output

        # for baseline gnn
        # y_pred = model_output

        # if epoch == 1 or epoch % 10 == 0:
        #     print("Z1 ori: ", z_1_ori)
        #     print("Z2 ori: ", z_2_ori)

        # ignore nan targets (unlabeled) when computing training loss.
        is_labeled = target == target
        # print("target: ", target)
        # print("y_pred: ", y_pred.shape)
        loss_target = self.cross(y_pred.to(self.device)[is_labeled], target.to(self.device)[is_labeled])
        # loss_target_label_unchange = 0
        loss_target_label_unchange = self.cross(y_pred_label_unchange.to(self.device)[is_labeled], target.to(self.device)[is_labeled])

        # for graphsst2 gin
        loss_target_label_unchange_coef = 0.06

        # for nci109 gt
        # loss_target_label_unchange_coef = 0.2

        # regular coef
        # loss_target_label_unchange_coef = 0.5
        loss_target_total = loss_target + loss_target_label_unchange_coef * loss_target_label_unchange

        # cos_similarity = nn.CosineSimilarity(dim=1)
        # z1_cos_distance_close_coef = 1
        # z1_cos_distance_close_loss = z1_cos_distance_close_coef * torch.mean((1 - cos_similarity(z_1_ori, z_1_label_unchange)))
        # # print("Cos distance close loss: ", z1_cos_distance_close_loss)
        # wandb.log({'Loss z1_cos_distance': z1_cos_distance_close_loss.item()})


        # reconstruction loss
        # for proteins
        # reconstruction_coef_max = 1e10


        # for graphsst gin
        reconstruction_coef_max = 1e11


        # for proteins gin
        # reconstruction_coef_max = 5e4

        # # for proteins gt
        # reconstruction_coef_max = 1e4

        # for bbbp gt
        # reconstruction_coef_max = 5e9
        # print("Reconstruction coef max: ", reconstruction_coef_max)

        # for nci1 gt
        # reconstruction_coef_max = 1e8

        # reconstruction_coef_max = 5e8


        # reconstruction_coef_max = 5e-9
        # reconstruction_coef_min = 1e-9

        loss_reconstruction_1 = reconstruction_coef_max * self.reconstruction_loss_max(z_1_ori, z_2_ori, z_1_label_change, z_2_label_change,
                                z_1_label_unchange, z_2_label_unchange)

        # loss_reconstruction_2 = reconstruction_coef_min * self.reconstruction_loss_min(z_1_ori, z_1_label_unchange)
        # loss_reconstruction_2 = 0
        # print("Reconstruction loss max: ", loss_reconstruction_1)
        # print("Reconstruction loss min: ", loss_reconstruction_2)
        # loss_reconstruction = loss_reconstruction_1 + loss_reconstruction_2
        loss_reconstruction = loss_reconstruction_1

        # wandb.log({'Loss reconstruction max': loss_reconstruction_1, 'Loss reconstruction min': loss_reconstruction_2})
        wandb.log({'Loss reconstruction': loss_reconstruction})

        # print("Reconstruction loss: ", loss_reconstruction)
        # coef for gt nci109
        kld_coef = 0.1

        # coef for gt nci1
        # kld_coef = 500

        # coef for gt proteins
        # kld_coef = 0.05

        # coef for gt bbbp
        # kld_coef = 5
        # coef pair_loss_cos
        # kld_coef = 0.5

        # for base data
        # kld_coef = 0.05

        # for gin bbbp
        # kld_coef = 10

        # for graphsst2 data
        # kld_coef = 0.5

        # for nci109 data
        # kld_coef = 0.5

        # coef pair_loss_cos for normalized kld
        # kld_coef = 0.05

        # coef pair_loss_cos with layer norm
        # kld_coef = 5

        # coef cos_loss
        # kld_coef = 1

        # kld_loss = kld_coef * (self.kld_uniform_loss(y_pred_z2_ori_nn1, epoch, True) + self.kld_uniform_loss(y_pred_z1_ori_nn2, epoch, False))
        # kld_loss = kld_coef * self.kld_uniform_loss(y_pred_z1_ori_nn2)

        kld_loss = 0
        if self.criterion_name == "pair_loss_triplet":

            loss_pair = 0.05 * pair_loss_triplet(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                                 s_pred_label_unchange.to(self.device))
            Loss = loss_target_total + loss_pair + kld_loss
            wandb.log({'Loss target': loss_target_total, 'Loss pair': loss_pair, 'Loss KLD': kld_loss})

        elif self.criterion_name == "pair_loss":
            loss_pair = 0.1 * pair_loss(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                        s_pred_label_unchange.to(self.device))
            loss_l2 = 0.01 * l2_loss(s_pred.to(self.device), s_pred_label_change.to(self.device))
            Loss = loss_target_total + loss_pair + loss_l2
            wandb.log({'Loss target': loss_target_total, 'Loss pair': loss_pair, 'Loss l2': loss_l2})

        elif self.criterion_name == "pair_loss_cos":
            # loss_pair_cos = 0.2 * pair_loss_cos(s_pred.to(self.device), s_pred_label_change.to(self.device),
            #                                     s_pred_label_unchange.to(self.device))

            # for bace data
            # if epoch == 1 or epoch == 90:
            #     print("s_pred: ", s_pred)
            #     print("s_pred_label_change: ", s_pred_label_change)
            #     print("s_pred_label_unchange: ", s_pred_label_unchange)

            # for graphsst data gt
            coef = 0.15

            # # for graphsst data gin
            # coef = 0.15

            # for nci109 data gt
            # coef = 0.5

            # for nci109 data gin
            # coef = 0.01

            # for proteins data gin
            # coef = 0.05

            # for proteins data gt
            # coef = 0.5

            # for bbbp data gt
            # coef = 0.1
            # for nci109 data gt
            # coef = 0.1
            # bbbp gnn
            # coef = 0.5
            loss_pair_cos = coef * pair_loss_cos(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                                 s_pred_label_unchange.to(self.device))

            # loss_l2 = 0.0001 * l2_loss(s_pred.to(self.device), s_pred_label_change.to(self.device))

            # loss_pair_cos = 0
            Loss = loss_target_total + loss_pair_cos + loss_reconstruction
            # Loss = loss_target
            wandb.log({'Loss target': loss_target_total, 'Loss pair cos': loss_pair_cos})

        elif self.criterion_name == "cos_loss":
            loss_close_cos = 1 * self.cos_loss(s_pred.to(self.device), s_pred_label_change.to(self.device),
                                               torch.ones(len(s_pred)).to(self.device))
            loss_far_cos = 1 * self.cos_loss(s_pred.to(self.device), s_pred_label_unchange.to(self.device),
                                             (torch.zeros(len(s_pred)) - 1).to(self.device))
            Loss = loss_target_total + loss_close_cos + loss_far_cos + kld_loss

            wandb.log({'Loss target': loss_target_total, 'Loss close cos': loss_close_cos, 'Loss far cos': loss_far_cos,
                       'Loss KLD': kld_loss})

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

