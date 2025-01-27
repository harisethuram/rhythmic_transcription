import torch
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.RhythmLSTM import RhythmLSTM
from src.preprocess.collate import preprocess_data
from const_tokens import *

def topk_accuracy(model: RhythmLSTM, val_dataloader: DataLoader, token_to_id: dict, topk: int=5):
    """
    Compute the top-k accuracy of the model on the given dataloader.
    """
    
    # set the model to eval mode
    model.eval()
    
    # initialize the number of correct predictions
    num_correct = 0
    total = 0
    
    # iterate over the dataloader
    for _, data in val_dataloader:
        # get the data and the labels
        input_data = data[..., :-1]
        target = data[..., 1:]
        
        # get the model predictions
        output, _ = model(input_data)
        
        # get the top k predictions
        _, topk_preds = torch.topk(output, topk, dim=-1)
        
        # get the number of correct predictions
        num_correct += (torch.any(topk_preds == target.unsqueeze(-1), dim=-1) & torch.logical_not(target == token_to_id[PADDING_TOKEN])).sum().item()
        total += torch.logical_not(target == token_to_id[PADDING_TOKEN]).sum().item()
        
    return num_correct / total

def topk_f1(model: RhythmLSTM, val_dataloader: DataLoader, token_to_id: dict, topk: int=5):
    """
    Compute the top-k F1 score of the model on the given dataloader.
    """
    
    # set the model to eval mode
    model.eval()
    
    tp = {i : 0 for i in token_to_id.values()}
    fp = {i : 0 for i in token_to_id.values()}
    fn = {i : 0 for i in token_to_id.values()}
    f1s = {i : 0 for i in token_to_id.values()}
    precisions = {i : 0 for i in token_to_id.values()}
    recalls = {i : 0 for i in token_to_id.values()}
    
    # iterate over the dataloader
    for token_id in token_to_id.values():
        # print("****")
        # print(token_id, token_to_id[token_id])
        
        for _, data in val_dataloader:
            # get the data and labels
            padding_mask = torch.logical_not(data[..., 1:] == token_to_id[PADDING_TOKEN])
            input_data = data[..., :-1]
            target = (data[..., 1:] == token_id).long()
            # print(input_data, target, token_id)
            # input()
            output, _ = model(input_data)
            
            _, topk_preds = torch.topk(output, topk, dim=-1)
            
            # print(topk_preds)
            topk_preds = torch.any(topk_preds == token_id, dim=-1).long()
            
            # topk_preds = (torch.Tensor([1, 1, 1, 1, 2]) == 1).long()
            # target = (torch.Tensor([0, 1, 0, 1, 1]) == 1).long()
            
            tp[token_id] += torch.logical_and(target * topk_preds, padding_mask).sum().item()
            fp[token_id] += torch.logical_and((1 - target) * topk_preds, padding_mask).sum().item()
            fn[token_id] += torch.logical_and(target * (1 - topk_preds), padding_mask).sum().item()
            
            
        precisions[token_id] = tp[token_id] / (tp[token_id] + fp[token_id] + 1e-8)
        recalls[token_id] = tp[token_id] / (tp[token_id] + fn[token_id] + 1e-8)
        
        f1s[token_id] = 2 * precisions[token_id] * recalls[token_id] / (precisions[token_id] + recalls[token_id] + 1e-8)
        
    return precisions, recalls, f1s  

def gen_heatmap(data_dict, path, title, id_to_token, ks):
    data = torch.Tensor([list(d.values()) for d in data_dict]).T
    
    sns.heatmap(data, xticklabels=ks, yticklabels=id_to_token.keys(), cmap="Blues")
    plt.xlabel("k")
    plt.ylabel("Token ID")
    plt.title(title)
    
    plt.savefig(path)
    plt.clf()
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/bach_fugues/pretrain_hparam_search/lr_1e-3/b_size_64/emb_64/hid_256/model.pth"
    data_path = "processed_data/bach_fugues/"
    
    model = torch.load(model_path)
    model.to(device)
    batch_size = 64
    
    _, val_dataloader, token_to_id, id_to_token = preprocess_data(data_path, batch_size=batch_size, device=device)
    accuracy_val = []
    precision_val = []
    average_precision_val = []
    recall_val = []
    average_recall_val = []
    f1_val = []
    average_f1_val = []
    
    train_loader, _, token_to_id, _ = preprocess_data(data_path, batch_size=batch_size, device=device)
    accuracy_train = []
    precision_train = []
    average_precision_train = []
    recall_train = []
    average_recall_train = []
    f1_train = []
    average_f1_train = []
    
    ks = [k for k in range(1, 1 + len(token_to_id.keys()))]
    print("Validation results:")
    for k in tqdm(ks):
        topk_val_acc = topk_accuracy(model, val_dataloader, token_to_id, topk=k)
        accuracy_val.append(topk_val_acc)
        
        topk_train_acc = topk_accuracy(model, train_loader, token_to_id, topk=k)
        accuracy_train.append(topk_train_acc)
        print(f"Top-{k} val accuracy: {topk_val_acc}, Top-{k} train accuracy: {topk_train_acc}")
        
        topk_val_prec, topk_val_recall, topk_val_f1 = topk_f1(model, val_dataloader, token_to_id, topk=k)
        precision_val.append(topk_val_prec)
        recall_val.append(topk_val_recall)
        f1_val.append(topk_val_f1)
        
        topk_train_prec, topk_train_recall, topk_train_f1 = topk_f1(model, train_loader, token_to_id, topk=k)
        precision_train.append(topk_train_prec)
        recall_train.append(topk_train_recall)
        f1_train.append(topk_train_f1)
        
    # plot top-k accuracy for each k
    
    plt.plot(ks, accuracy_val, label="Validation")
    plt.plot(ks, accuracy_train, label="Train")
    plt.xlabel("k")
    plt.legend()
    plt.ylabel("Top-k accuracy")
    plt.title("Top-k accuracy of model")
    plt.savefig("topk_accuracy.png")
    plt.clf()
    
    # plot f1, precision and recall averaged over all tokens for each k
    plt.plot(ks, [sum(f1.values()) / len(token_to_id.keys()) for f1 in f1_val], label="F1 val", color="maroon")
    plt.plot(ks, [sum(f1.values()) / len(token_to_id.keys()) for f1 in f1_train], label="F1 train", color="lightcoral")
    
    plt.plot(ks, [sum(prec.values()) / len(token_to_id.keys()) for prec in precision_val], label="Precision val", color="darkgreen")
    plt.plot(ks, [sum(prec.values()) / len(token_to_id.keys()) for prec in precision_train], label="Precision train", color="lime")
    
    plt.plot(ks, [sum(rec.values()) / len(token_to_id.keys()) for rec in recall_val], label="Recall val", color="darkblue")
    plt.plot(ks, [sum(rec.values()) / len(token_to_id.keys()) for rec in recall_train], label="Recall train", color="lightblue")
    
    plt.xlabel("k")
    plt.legend()
    plt.ylabel("Score")
    plt.title("Top-k F1, precision and recall of model")
    plt.savefig("topk_f1_precision_recall.png")
    plt.clf()
    
    # plot the heatmap of f1, y axis is token id, x axis is k
    # f1_val = torch.Tensor([list(f1.values()) for f1 in f1_val]).T
    # f1_train = torch.Tensor([list(f1.values()) for f1 in f1_train]).T
    
    gen_heatmap(f1_val, "f1_val_heatmap.png", "F1 score of model on validation set", id_to_token, ks)
    gen_heatmap(f1_train, "f1_train_heatmap.png", "F1 score of model on training set", id_to_token, ks)
    gen_heatmap(precision_val, "precision_val_heatmap.png", "Precision of model on validation set", id_to_token, ks)
    gen_heatmap(precision_train, "precision_train_heatmap.png", "Precision of model on training set", id_to_token, ks)
    gen_heatmap(recall_val, "recall_val_heatmap.png", "Recall of model on validation set", id_to_token, ks)
    gen_heatmap(recall_train, "recall_train_heatmap.png", "Recall of model on training set", id_to_token, ks)
    
    # sns.heatmap(f1_val, xticklabels=ks, yticklabels=id_to_token.keys(), cmap="Blues")
    # plt.xlabel("k")
    # plt.ylabel("Token ID")
    # plt.title("F1 score of model on validation set")
    # plt.savefig("f1_val_heatmap.png")
    # plt.clf()
    
    