import torch
import wandb

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from fedavg_util import set_average_weights_as_main_model, send_main_model_to_nodes
from quick_test_vae_util import VAECriterion, DisentCriterion


def eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''

    rocauc_list = []

    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            # print(y_pred[is_labeled, i])
            # print(y_true[is_labeled, i])
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


def eval_step(loader, model, cls_criterion, device, compute_loss=False):
    model.eval()

    y_true = []
    y_pred = []
    loss = 0
    all_count = 0
    L = 0
    for data in loader:
        data = data.to(device)

        if data.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                x = data.x
                edge_index = data.edge_index
                pred = model(x, edge_index, data)
            if compute_loss:
                data.y = data.y.view(data.y.shape[0], -1)

                is_labeled = data.y == data.y
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
                L += loss.item()
                all_count += 1
            y_true.append(data.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    if compute_loss:
        loss = L/all_count

    y_true = torch.cat(y_true, dim=0).numpy()
    soft_y_pred = torch.cat(y_pred, dim=0).numpy()
    hard_y_pred = (torch.cat(y_pred, dim=0).sigmoid() > 0.5).int().numpy()
    # # print(y_true)
    # print(hard_y_pred)
    # exit()
    return eval_rocauc(y_true, soft_y_pred), loss, f1_score(y_true, hard_y_pred), accuracy_score(y_true, hard_y_pred)


def train_step(tr_loader, model, optimizer, cls_criterion, device):
    model.train()
    model.to(device)
    all_count = 0
    L = 0
    for data in tr_loader:
        data = data.to(device)

        if data.x.shape[0] == 1:
            pass
        else:
            x = data.x
            edge_index = data.edge_index
            pred = model(x, edge_index, data)
            optimizer.zero_grad()

            data.y = data.y.view(data.y.shape[0], -1)
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = data.y == data.y
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            L += loss.item()
            all_count += 1
    # print("all_count: ", all_count)
    return L/all_count


def train_data(train_loader, val_loader, test_loader_small, test_loader_large, model, optimizer, device, pth=""):

    cls_criterion = torch.nn.BCEWithLogitsLoss()

    # bval = 10
    bval = 100

    btest_1 = 0
    btest_2 = 0
    btest_1_f1 = 0
    btest_2_f1 = 0
    btest_1_acc = 0
    btest_2_acc = 0
    bval_f1 = 0
    bval_roc = 0
    bval_acc = 0
    patience = 50
    count = 0
    for epoch in range(1, bval + 1):
        train_loss = train_step(train_loader, model, optimizer, cls_criterion, device)
        train_roc, _, train_f1, train_acc = eval_step(train_loader, model, cls_criterion, device, compute_loss=False)
        val_roc, val_loss, val_f1, val_acc = eval_step(val_loader, model, cls_criterion, device, compute_loss=True)
        test_roc_1, test_loss_1, test_f1_small, test_acc_1 = eval_step(test_loader_small, model,
                                                                       cls_criterion, device, compute_loss=True)
        test_roc_2, test_loss_2, test_f1_large, test_acc_2 = eval_step(test_loader_large, model,
                                                                       cls_criterion, device, compute_loss=True)
        wandb.log({'train accuracy': train_acc, 'train loss': train_loss, 'val accuracy' : val_acc,
                   'val loss': val_loss, 'test small accuracy': test_acc_1, 'test small loss': test_loss_1})

        if bval > val_loss:
            bval = val_loss
            bval_f1 = val_f1
            bval_roc = val_roc
            bval_acc = val_acc
            btest_1 = test_roc_1
            btest_2 = test_roc_2
            btest_1_f1 = test_f1_small
            btest_2_f1 = test_f1_large
            btest_1_acc = test_acc_1
            btest_2_acc = test_acc_2
            count = 0
            # if pth:
            #     torch.save(model.state_dict(), pth)
        else:
            count += 1
            if count == patience:
                break
        print('Epoch: {:02d}, trloss: {:.4f}, trroc: {:.4f}, trf1: {:.4f}, tracc: {:.4f}, valloss: {:.4f}, '
              'valroc: {:.4f}, valf1: {:.4f}, valacc: {:.4f}, testloss_1: {:.4f}, testroc_1: {:.4f}, testf1_1: {:.4f}, '
              'testacc_1: {:.4f}, testloss_2: {:.4f}, testroc_2: {:.4f}, testf1_2: {:.4f}, '
              'testacc_2: {:.4f}.'.format(epoch,train_loss, train_roc, train_f1, train_acc, val_loss, val_roc,
                                          val_f1, val_acc, test_loss_1, test_roc_1, test_f1_small, test_acc_1,
                                          test_loss_2, test_roc_2, test_f1_large, test_acc_2))
    print("Best test roc/f1: test_roc_1:{:.4f}, test_f1_1:{:.4f}, test_acc_1:{:.4f}, "
          "test_roc_2:{:.4f}, test_f1_2:{:.4f}, test_acc_2:{:.4f}, val_f1:{:.6f}, val_roc:{:.4f}, "
          "val_acc:{:.4f}".format(btest_1, btest_1_f1, btest_1_acc, btest_2, btest_2_f1,
                                  btest_2_acc, bval_f1, bval_roc, bval_acc))

    return btest_1, btest_2, btest_1_f1, btest_2_f1, btest_1_acc, btest_2_acc, bval_f1, bval_roc, bval_acc



def train_data_fedavg(main_model, train_loader_list, val_loader_list, model_list, optimizer_list, criterion_list,
                      test_loader_small, val_server_loader, test_loader_large, server_epochs, model_name, device, pth=""):

    cls_criterion = torch.nn.BCEWithLogitsLoss()

    bval = 10

    bval_f1 = 0
    bval_roc = 0
    bval_acc = 0
    # patience = 50
    # count = 0
    btest_1 = 0
    btest_2 = 0
    btest_1_f1 = 0
    btest_2_f1 = 0
    btest_1_acc = 0
    btest_2_acc = 0
    best_server_epoch = 0
    val_server_loss_temp = 10

    for server_epoch in range(1, server_epochs + 1):
        model_list = send_main_model_to_nodes(main_model, model_list, model_name)
        # print("After sending main model to silo: ", model_list[0].conv1.lin.weight[0, 0:5])
        # print("Main model: ", main_model.conv1.lin.weight[0])
        for silo_idx in range(len(model_list)):
            print("Training silo {} ...".format(silo_idx))
            cur_model = model_list[silo_idx]
            cur_train_loader = train_loader_list[silo_idx]
            cur_val_loader = val_loader_list[silo_idx]
            cur_optimizer = optimizer_list[silo_idx]
            cur_criterion = criterion_list[silo_idx]
            for epoch in range(1, 10 + 1):
                train_loss = train_step(cur_train_loader, cur_model, cur_optimizer, cur_criterion, device)
                train_roc, _, train_f1, train_acc = eval_step(cur_train_loader, cur_model, cur_criterion, device, compute_loss=False)
                val_roc, val_loss, val_f1, val_acc = eval_step(cur_val_loader, cur_model, cur_criterion, device, compute_loss=True)

                if val_loss < bval:
                    bval = val_loss
                    bval_f1 = val_f1
                    bval_roc = val_roc
                    bval_acc = val_acc
                    # btest_1=test_roc_1
                    # btest_2=test_roc_2
                    # btest_1_f1 = test_f1_small
                    # btest_2_f1 = test_f1_large
                    # btest_1_acc = test_acc_1
                    # btest_2_acc = test_acc_2
                    model_list[silo_idx] = cur_model
                    # print("Silo {} model updated".format(silo_idx))
                print('Epoch: {:02d}, trloss: {:.4f}, trroc: {:.4f}, trf1: {:.4f}, tracc: {:.4f}, valloss: {:.4f}, '
                      'valroc: {:.4f}, valf1: {:.4f}, valacc: {:.4f}.'.format(epoch, train_loss, train_roc, train_f1,
                                                                              train_acc, val_loss, val_roc, val_f1, val_acc))
            print("Best val roc/f1: val_roc:{:.4f}, val_f1_1:{:.4f}, val_acc_:{:.4f}. ".format(bval_roc, bval_f1, bval_acc))

        # print("After training, silo model: ", model_list[0].conv1.lin.weight[0, 0:5])
        # print("Main model: ", main_model.conv1.lin.weight[0])
        main_model = set_average_weights_as_main_model(main_model, model_list, model_name, device)

        # print("After aggregating, silo model: ", model_list[0].conv1.lin.weight[0, 0:5])
        # print("Main model: ", main_model.conv1.lin.weight[0])
        # print("After fedavg agg, main model: ", main_model.conv1.lin.weight[0, 0:5])

        val_server_roc, val_server_loss, val_server_f1, val_server_acc = eval_step(val_server_loader, main_model,
                                                                       cls_criterion, device, compute_loss=True)
        test_roc_1, test_loss_1, test_f1_small, test_acc_1 = eval_step(test_loader_small, main_model,
                                                                       cls_criterion, device, compute_loss=True)
        test_roc_2, test_loss_2, test_f1_large, test_acc_2 = eval_step(test_loader_large, main_model,
                                                                       cls_criterion, device, compute_loss=True)

        print('Server Epoch: {:02d}, valloss: {:.4f}, valroc: {:.4f}, valf1: {:.4f}, '
              'valacc: {:.4f}, testloss_1: {:.4f}, testroc_1: {:.4f}, testf1_1: {:.4f}, '
              'testacc_1: {:.4f}, testloss_2: {:.4f}, testroc_2: {:.4f}, testf1_2: {:.4f}, '
              'testacc_2: {:.4f}.'.format(server_epoch,  val_server_loss, val_server_roc, val_server_f1, val_server_acc,
                                          test_loss_1, test_roc_1, test_f1_small, test_acc_1, test_loss_2,
                                          test_roc_2, test_f1_large, test_acc_2))

        if val_server_loss < val_server_loss_temp:
            val_server_loss_temp = val_server_loss
            btest_1 = test_roc_1
            btest_1_f1 = test_f1_small
            btest_1_acc = test_acc_1
            btest_2 = test_roc_2
            btest_2_f1 = test_f1_large
            btest_2_acc = test_acc_2
            best_server_epoch = server_epoch

    print("Best test roc/f1: test_roc_1:{:.4f}, test_f1_1:{:.4f}, test_acc_1:{:.4f}, test_roc_2:{:.4f}, "
          "test_f1_2:{:.4f}, test_acc_2:{:.4f} on server epoch: {}.".format(btest_1, btest_1_f1, btest_1_acc, btest_2,
                                                                           btest_2_f1, btest_2_acc, best_server_epoch))

    return model_list


def eval_step_gnn_vae(loader, model, cls_criterion, device, epoch, compute_loss=False):
    model.eval()
    y_true = []
    y_pred = []
    loss = 0
    all_count = 0
    L = 0
    for data in loader:
        data = data.to(device)

        if data.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                x = data.x
                edge_index = data.edge_index
                # batch = data.batch
                pred = model(x, edge_index, data)
            if compute_loss:
                data.y = data.y.view(data.y.shape[0], -1)

                is_labeled = data.y == data.y
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
                L += loss.item()
                all_count += 1
            y_true.append(data.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    if compute_loss:
        loss = L/all_count

    y_true = torch.cat(y_true, dim=0).numpy()
    soft_y_pred = torch.cat(y_pred, dim=0).numpy()
    hard_y_pred = (torch.cat(y_pred, dim=0).sigmoid() > 0.5).int().numpy()
    # if epoch == 1 or epoch == 90:
    #     print("Ground truth: ", y_true)
    #     print("Y pred: ", hard_y_pred)
    # exit()
    return eval_rocauc(y_true, soft_y_pred), loss, f1_score(y_true, hard_y_pred), accuracy_score(y_true, hard_y_pred)


def cal_size_label(data):
    size_len_list = []
    size = int(data.batch.max().item() + 1)
    batch = data.batch
    for i in range(size):
        mask = torch.eq(batch, i)
        input_tensor = data.x[mask]
        size_len_list.append(len(input_tensor))

    size_len_list = torch.tensor(size_len_list).view(len(size_len_list), -1)

    size_len_list_permuted = torch.vstack((size_len_list[-1], size_len_list[0:-1]))
    compare_result = torch.gt(size_len_list, size_len_list_permuted)
    # compare_result_2 = torch.ge(size_len_list, size_len_list_permuted)
    # print(compare_result)
    # print(compare_result_2)
    # print(compare_result == compare_result_2)
    # print(torch.sum((compare_result == compare_result_2).clone().detach()))
    # size_len_list_concated = torch.hstack((out_2, out_2_permuted))

    return compare_result


def train_step_gnn_vae(tr_loader, model, optimizer_1, optimizer_2, epoch, device):
    model.train()
    model.to(device)

    # hyperparameter for vae criterion
    # lambda_e = 1.0
    lambda_e = 0.3

    lambda_od = 0.7
    # lambda_od = 0.063

    gamma_e = 1
    # gamma_od = 1.7
    gamma_od = 0.7
    step_size = 30
    # bce_criterion = torch.nn.BCEWithLogitsLoss()

    vae_criterion = VAECriterion(lambda_e, lambda_od, gamma_e, gamma_od, step_size)
    all_count = 0
    L = 0

    for data in tr_loader:
        data = data.to(device)
        if data.x.shape[0] == 1:
            pass
        else:
            x = data.x
            edge_index = data.edge_index
            model_output = model(x, edge_index, data)
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            # size_compare_label = cal_size_label(data).to(torch.float32).to(device)
            # size_compare_pred = model_output[1][2].to(torch.float32)

            # loss_sensitive = bce_criterion(size_compare_pred.to(torch.float32), size_compare_label.to(torch.float32))

            # for param in model.vae.encoder.mlp_block.parameters():
            #     param.requires_grad = False
            # loss_sensitive.backward(retain_graph=True)
            #
            # for param in model.vae.encoder.mlp_block.parameters():
            #     param.requires_grad = True

            data.y = data.y.view(data.y.shape[0], -1)
            loss = vae_criterion(model_output, data.y.to(torch.float32), epoch)
            loss.backward()

            optimizer_1.step()
            optimizer_2.step()
            # L += (loss.item() + loss_sensitive.item())
            L += loss.item()

            all_count += 1
            # wandb.log({'loss sensitive': loss_sensitive, 'vae total loss': loss})
            wandb.log({'vae total loss': loss})

    # print("all_count: ", all_count)

    # print("Size label: ", size_compare_label)
    # print("Size pred: ", size_compare_pred)

    return L/all_count


def train_data_gnn_vae(train_loader, val_loader, test_loader_small, test_loader_large, model, optimizer_1, optimizer_2,
                       device, pth=""):

    cls_criterion = torch.nn.BCEWithLogitsLoss()

    bval = 200
    btest_1 = 0
    btest_2 = 0
    btest_1_f1 = 0
    btest_2_f1 = 0
    btest_1_acc = 0
    btest_2_acc = 0
    bval_f1 = 0
    bval_roc = 0
    bval_acc = 0
    patience = 50
    count = 0
    for epoch in range(1, bval + 1):
        train_loss = train_step_gnn_vae(train_loader, model, optimizer_1, optimizer_2, epoch, device)
        train_roc, _, train_f1, train_acc = eval_step_gnn_vae(train_loader, model, cls_criterion, device, compute_loss=False)
        val_roc, val_loss, val_f1, val_acc = eval_step_gnn_vae(val_loader, model, cls_criterion, device, compute_loss=True)
        test_roc_1, test_loss_1, test_f1_small, test_acc_1 = eval_step_gnn_vae(test_loader_small, model,
                                                                       cls_criterion, device, compute_loss=True)
        test_roc_2, test_loss_2, test_f1_large, test_acc_2 = eval_step_gnn_vae(test_loader_large, model,
                                                                       cls_criterion, device, compute_loss=True)

        if bval > val_loss:
            bval = val_loss
            bval_f1 = val_f1
            bval_roc = val_roc
            bval_acc = val_acc
            btest_1 = test_roc_1
            btest_2 = test_roc_2
            btest_1_f1 = test_f1_small
            btest_2_f1 = test_f1_large
            btest_1_acc = test_acc_1
            btest_2_acc = test_acc_2
            count = 0
            # if pth:
            #     torch.save(model.state_dict(), pth)
        else:
            count += 1
            if count == patience:
                break
        print('Epoch: {:02d}, trloss: {:.4f}, trroc: {:.4f}, trf1: {:.4f}, tracc: {:.4f}, valloss: {:.4f}, '
              'valroc: {:.4f}, valf1: {:.4f}, valacc: {:.4f}, testloss_1: {:.4f}, testroc_1: {:.4f}, testf1_1: {:.4f}, '
              'testacc_1: {:.4f}, testloss_2: {:.4f}, testroc_2: {:.4f}, testf1_2: {:.4f}, '
              'testacc_2: {:.4f}.'.format(epoch, train_loss, train_roc, train_f1, train_acc, val_loss, val_roc,
                                          val_f1, val_acc, test_loss_1, test_roc_1, test_f1_small, test_acc_1,
                                          test_loss_2, test_roc_2, test_f1_large, test_acc_2))

        wandb.log({'train accuracy': train_acc, 'train loss': train_loss, 'val accuracy': val_acc,
                   'val loss': val_loss, 'test small accuracy': test_acc_1, 'test small loss': test_loss_1,
                   'test small roc': test_roc_1, 'test small f1': test_f1_small, 'test large accuracy': test_acc_2,
                   'test large loss': test_loss_2, 'test large roc': test_roc_2, 'test large f1': test_f1_large})

    print("Best test roc/f1: test_roc_1:{:.4f}, test_f1_1:{:.4f}, test_acc_1:{:.4f}, "
          "test_roc_2:{:.4f}, test_f1_2:{:.4f}, test_acc_2:{:.4f}, val_f1:{:.6f}, val_roc:{:.4f}, "
          "val_acc:{:.4f}".format(btest_1, btest_1_f1, btest_1_acc, btest_2, btest_2_f1,
                                  btest_2_acc, bval_f1, bval_roc, bval_acc))

    return btest_1, btest_2, btest_1_f1, btest_2_f1, btest_1_acc, btest_2_acc, bval_f1, bval_roc, bval_acc


def train_step_disentgnn(tr_loader, model, optimizer_1, epoch, device, criterion_name):
    model.train()
    model.to(device)

    disent_criterion = DisentCriterion(criterion_name)
    all_count = 0
    L = 0

    for data in tr_loader:
        data = data.to(device)
        if data.x.shape[0] == 1:
            pass
        else:
            x = data.x
            edge_index = data.edge_index
            model_output = model(x, edge_index, data)
            optimizer_1.zero_grad()

            # size_compare_label = cal_size_label(data).to(torch.float32).to(device)
            # size_compare_pred = model_output[1][2].to(torch.float32)

            # loss_sensitive = bce_criterion(size_compare_pred.to(torch.float32), size_compare_label.to(torch.float32))

            # for param in model.vae.encoder.mlp_block.parameters():
            #     param.requires_grad = False
            # loss_sensitive.backward(retain_graph=True)
            #
            # for param in model.vae.encoder.mlp_block.parameters():
            #     param.requires_grad = True

            data.y = data.y.view(data.y.shape[0], -1)
            loss = disent_criterion(model_output, data.y.to(torch.float32), epoch)
            loss.backward()

            optimizer_1.step()
            # L += (loss.item() + loss_sensitive.item())
            L += loss.item()

            all_count += 1
            # wandb.log({'loss sensitive': loss_sensitive, 'vae total loss': loss})
            wandb.log({'disent total loss': loss})

    # print("all_count: ", all_count)

    # print("Size label: ", size_compare_label)
    # print("Size pred: ", size_compare_pred)

    return L/all_count


def train_data_disentgnn(train_loader, val_loader, test_loader_small, test_loader_large, model, optimizer_1,
                         device, criterion_name, pth=""):

    cls_criterion = torch.nn.BCEWithLogitsLoss()

    bval = 100
    btest_1 = 0
    btest_2 = 0
    btest_1_f1 = 0
    btest_2_f1 = 0
    btest_1_acc = 0
    btest_2_acc = 0
    bval_f1 = 0
    bval_roc = 0
    bval_acc = 0
    patience = 50
    count = 0
    for epoch in range(1, bval + 1):
        train_loss = train_step_disentgnn(train_loader, model, optimizer_1, epoch, device, criterion_name)
        train_roc, _, train_f1, train_acc = eval_step_gnn_vae(train_loader, model, cls_criterion, device, epoch, compute_loss=False)
        val_roc, val_loss, val_f1, val_acc = eval_step_gnn_vae(val_loader, model, cls_criterion, device, epoch, compute_loss=True)
        test_roc_1, test_loss_1, test_f1_small, test_acc_1 = eval_step_gnn_vae(test_loader_small, model,
                                                                       cls_criterion, device, epoch, compute_loss=True)
        test_roc_2, test_loss_2, test_f1_large, test_acc_2 = eval_step_gnn_vae(test_loader_large, model, cls_criterion,
                                                                               device, epoch, compute_loss=True)

        if bval > val_loss:
            bval = val_loss
            bval_f1 = val_f1
            bval_roc = val_roc
            bval_acc = val_acc
            btest_1 = test_roc_1
            btest_2 = test_roc_2
            btest_1_f1 = test_f1_small
            btest_2_f1 = test_f1_large
            btest_1_acc = test_acc_1
            btest_2_acc = test_acc_2
            count = 0
            # if pth:
            #     torch.save(model.state_dict(), pth)
        else:
            count += 1
            if count == patience:
                break
        print('Epoch: {:02d}, trloss: {:.4f}, trroc: {:.4f}, trf1: {:.4f}, tracc: {:.4f}, valloss: {:.4f}, '
              'valroc: {:.4f}, valf1: {:.4f}, valacc: {:.4f}, testloss_1: {:.4f}, testroc_1: {:.4f}, testf1_1: {:.4f}, '
              'testacc_1: {:.4f}, testloss_2: {:.4f}, testroc_2: {:.4f}, testf1_2: {:.4f}, '
              'testacc_2: {:.4f}.'.format(epoch, train_loss, train_roc, train_f1, train_acc, val_loss, val_roc,
                                          val_f1, val_acc, test_loss_1, test_roc_1, test_f1_small, test_acc_1,
                                          test_loss_2, test_roc_2, test_f1_large, test_acc_2))

        wandb.log({'train accuracy': train_acc, 'train loss': train_loss, 'val accuracy': val_acc,
                   'val loss': val_loss, 'test small accuracy': test_acc_1, 'test small loss': test_loss_1,
                   'test small roc': test_roc_1, 'test small f1': test_f1_small, 'test large accuracy': test_acc_2,
                   'test large loss': test_loss_2, 'test large roc': test_roc_2, 'test large f1': test_f1_large})

    print("Best test roc/f1: test_roc_1:{:.4f}, test_f1_1:{:.4f}, test_acc_1:{:.4f}, "
          "test_roc_2:{:.4f}, test_f1_2:{:.4f}, test_acc_2:{:.4f}, val_f1:{:.6f}, val_roc:{:.4f}, "
          "val_acc:{:.4f}".format(btest_1, btest_1_f1, btest_1_acc, btest_2, btest_2_f1,
                                  btest_2_acc, bval_f1, bval_roc, bval_acc))

    return btest_1, btest_2, btest_1_f1, btest_2_f1, btest_1_acc, btest_2_acc, bval_f1, bval_roc, bval_acc


