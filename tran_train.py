import nni
import argparse
import warnings
import numpy as np
import torch
import torch.optim as optim

from utils import *
from models import *
from train_and_eval import *
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

import pickle
import os
def load_indices(seed, dataset_path):
    seed_path = os.path.join(dataset_path, str(seed))
    idx_file_path = os.path.join(seed_path, 'indices.pkl')
    with open(idx_file_path, 'rb') as f:
        idx_data = pickle.load(f)
    print(f"Indices loaded for seed {seed} from {seed_path}")
    return idx_data['idx_train'], idx_data['idx_val'], idx_data['idx_test']

def load_graph_and_labels(dataset_path):
    graph_file_path = os.path.join(dataset_path, 'graph.pkl')
    labels_file_path = os.path.join(dataset_path, 'labels.pt')
    with open(graph_file_path, 'rb') as f:
        g = pickle.load(f)
    labels = torch.load(labels_file_path)
    print(f"Graph and labels loaded from {dataset_path}")
    return g, labels

def load_teacher_outputs(filepath):
    data = torch.load(filepath)
    return data['out_t'], data['out_self_loop'], data['test_teacher'], data['state']



def save_to_pkl(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_from_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def main(seed, param):

    g_path = f'data/{param["dataset"]}/{param["exp_setting"]}_{seed}.pkl'
    if os.path.exists(g_path):
        data_dict = load_from_pkl(g_path)
        g, labels, idx_train, idx_val, idx_test = data_dict['g'], data_dict['labels'], data_dict['idx_train'], data_dict['idx_val'], data_dict['idx_test']
    feats = g.ndata["feat"].to(device)
    labels = labels.to(device)
    param['feat_dim'] = g.ndata["feat"].shape[1]
    param['label_dim'] = labels.int().max().item() + 1
    if param['exp_setting'] == "tran":
        indices = (idx_train, idx_val, idx_test)
    elif param['exp_setting'] == "ind":
        indices = graph_split(idx_train, idx_val, idx_test, labels, param)
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    evaluator = get_evaluator(param["dataset"])
    model_t = Model(param, model_type='teacher').to(device)
    teacher_filepath = f'teacher/{param["dataset"]}/{param["exp_setting"]}_{param["teacher"]}_{seed}_teacher_outputs.pth'
    out_t, out_self_loop, test_teacher, state_t = load_teacher_outputs(teacher_filepath)
    out_t, out_self_loop = out_t.to(device), out_self_loop.to(device)
    model_t.load_state_dict(state_t)
    g = g.to(device)
    model_s = MLP(param['num_layers'], param['feat_dim'], param['hidden_dim'], int(labels.max() + 1), param['dropout_s']).to(device)
    optimizer_s = optim.Adam(list(model_s.parameters()), lr=float(param["learning_rate"]), weight_decay=float(param["weight_decay"]))
    test_acc, test_val, test_best = train_our_student(param, model_s, 
                                                            feats, labels, out_t, out_self_loop, indices, criterion_t, 
                                                            evaluator, optimizer_s) 
    return test_teacher, test_acc, test_val, test_best


def use_weight(param, seed):
    g_path = f'data/{param["dataset"]}/{param["exp_setting"]}_{seed}.pkl'
    if os.path.exists(g_path):
        data_dict = load_from_pkl(g_path)
        g, labels, idx_train, idx_val, idx_test = data_dict['g'], data_dict['labels'], data_dict['idx_train'], data_dict['idx_val'], data_dict['idx_test']
    feats = g.ndata["feat"].to(device)
    labels = labels.to(device)
    param['feat_dim'] = g.ndata["feat"].shape[1]
    param['label_dim'] = labels.int().max().item() + 1
    evaluator = get_evaluator(param["dataset"])
    model_s = MLP(param['num_layers'], param['feat_dim'], param['hidden_dim'], int(labels.max() + 1), param['dropout_s']).to(device)
    model_s.load_state_dict(torch.load(f'save_weight/{param["dataset"]}/{param["exp_setting"]}_{seed}.pt'))
    model_s.eval()
    evaluator = get_evaluator(param["dataset"])
    out, _ = evaluate_mini_batch(model_s, feats)
    mode_test_acc = evaluator(out[idx_test], labels[idx_test])
    return mode_test_acc




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--teacher", type=str, default="GCN")
    parser.add_argument("--student", type=str, default="MLP")
    parser.add_argument("--exp_setting", type=int, default=0)
    parser.add_argument("--split_rate", type=float, default=0.1)

    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout_t", type=float, default=0.3)
    parser.add_argument("--dropout_s", type=float, default=0.4)

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=500)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_mode", type=int, default=0)

    parser.add_argument("--tau_0", type=float, default=0.5)
    parser.add_argument("--tau_1", type=float, default=20)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--l_alpha", type=float, default=0.5)
    parser.add_argument("--l_beta", type=float, default=0.5)
    parser.add_argument("--l_gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--warm_up", type=int, default=50)
    
    
    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())

    if param['data_mode'] == 0:
        param['dataset'] = 'cora'
    if param['data_mode'] == 1:
        param['dataset'] = 'citeseer'
    if param['data_mode'] == 2:
        param['dataset'] = 'pubmed'
    if param['data_mode'] == 3:
        param['dataset'] = 'c-cs'
    if param['data_mode'] == 4:
        param['dataset'] = 'c-phy'
    if param['data_mode'] == 5:
        param['dataset'] = 'a-photo'
    if param['data_mode'] == 6:
        param['dataset'] = 'a-computer'
    if param['data_mode'] == 7:
        param['dataset'] = 'ogbn-arxiv'

    if param['data_mode'] == 7:
        param['norm_type'] = 'batch'
    else:
        param['norm_type'] = 'none'
    if param['exp_setting'] == 0:
        param['exp_setting'] = 'tran'
    else:
        param['exp_setting'] = 'ind'

    print("最终参数:", param) 

    test_acc_list = []
    test_val_list = []
    test_best_list = []
    test_teacher_list = []
    new_test_teacher_list = []
    teacher_time_list, student_time_list = [], []
    model_test_accs = []
    for seed in range(10):
        model_test_accs.append(use_weight(param, seed))
        set_seed(param['seed'] + seed)
        test_teacher, test_acc, test_val, test_best= main(seed, param)
        test_acc_list.append(test_acc)
        test_val_list.append(test_val)
        test_best_list.append(test_best)
        test_teacher_list.append(test_teacher)
        nni.report_intermediate_result(test_acc)
        
        
    nni.report_final_result(np.mean(test_acc_list))
    
    final_acc = np.mean(test_acc_list)
    if len(test_acc_list) != 10:
        final_acc = 0
    print(f'test_acc_list = {test_acc_list}')
    print(f'final_acc = {final_acc}')
    print(f'model_test_accs = {model_test_accs}')
    print(f'final_model_acc = {np.mean(model_test_accs)}')
        
