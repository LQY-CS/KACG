import copy
import torch
import torch
import torch.nn.functional as F
from utils import *
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def evaluate(model, g, feats):
    model.eval()
    with torch.no_grad():
        logits = model(g, feats)
        out = logits.log_softmax(dim=1)
    return logits, out

def evaluate_mini_batch(model, feats):
    model.eval()
    with torch.no_grad():
        logits = model(feats)
        logits = model.cls(logits)
        out = logits.log_softmax(dim=1)
    return logits, out

def train_our_batch(model, feats, hg, hgm, criterion_t, optimizer, param):
    model.train()
    z = model(feats)
    logits = model.cls(z) 
    loss_g = criterion_t((logits/param['tau_0']).log_softmax(dim=1), (hg/param['tau_0']).log_softmax(dim=1)) 
    topk = param['topk']
    _, topk_indices = torch.topk(hgm, k=topk, dim=1)
    candidate_label = torch.zeros_like(hgm).scatter_(1, topk_indices, 1)
    complementary_label = 1 - candidate_label
    weighted_hgm = (hgm ** param['tau_1']) * candidate_label 
    normalized_hgm = weighted_hgm / (weighted_hgm.sum(dim=1, keepdim=True) + 1e-6) 
    loss_pl = -torch.sum(normalized_hgm * logits.log_softmax(dim=1), dim=1).mean()
    loss_nl = -torch.mean(torch.sum(complementary_label * torch.log(1.0000001 - F.softmax(logits, dim=1)), dim=1))
    loss = loss_g  + param['l_alpha'] * loss_nl + param['l_beta'] * loss_pl 
    with torch.no_grad():
        hgm = param['beta'] * F.softmax(logits, dim=1) + (1 - param['beta']) * hgm
        hgm = hgm / (hgm.sum(dim=1, keepdim=True) + 1e-6)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss_g.item(), loss_nl.item() + loss_pl.item(), hgm

def train_our_student(param, model, feats, labels, out_t_all, our_t_self_all, 
                      indices, criterion_t, evaluator,  optimizer):

    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices

    es, val_best, test_val, test_best = 0, 0, 0, 0
    prob_out_t = F.softmax(out_t_all, dim=1)
    prob_our_t_self = F.softmax(our_t_self_all, dim=1)
    kl_div = F.kl_div(prob_out_t.log(), prob_our_t_self, reduction='none').sum(dim=1)
    alpha = torch.exp(-kl_div).unsqueeze(1) 
    min_alpha = param['alpha']
    alpha = torch.clamp(torch.exp(-kl_div), min=min_alpha).unsqueeze(1) 
    hgm = alpha * prob_out_t + (1 - alpha) * prob_our_t_self
    for epoch in range(1, param["max_epoch"] + 1):
        if param['exp_setting'] == 'tran':
            loss_l, loss_t, new_hgm= train_our_batch(model, feats, out_t_all, hgm, criterion_t, optimizer, param)
            if epoch > param['warm_up'] :
                hgm = new_hgm
            _, out = evaluate_mini_batch(model, feats)
            train_acc = evaluator(out[idx_train], labels[idx_train])
            val_acc = evaluator(out[idx_val], labels[idx_val])
            test_acc = evaluator(out[idx_test], labels[idx_test])
        if epoch % 10 == 0:
            print("\033[0;30;43m [{}] loss_l: {:.5f}, loss_t: {:.5f}, Total: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}\033[0m".format(
                                        epoch, loss_l, loss_t, loss_l + loss_t, train_acc, val_acc, test_acc, val_best, test_val, test_best))

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best =  val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == 50 and epoch > 100:
            print("Early stopping!")
            break
    
    model.eval()
    if param['exp_setting'] == 'tran':
        out, _ = evaluate_mini_batch(model, feats)
        mode_test_acc = evaluator(out[idx_test], labels[idx_test])
    return test_acc, test_val, test_best

