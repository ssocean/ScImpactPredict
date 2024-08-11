import torch
import numpy as np
from sklearn.metrics import ndcg_score

def average_rank_error(pred_indices, true_indices):
    """计算平均排名误差"""
    errors = np.abs(np.array(pred_indices) - np.array(true_indices))
    return np.mean(errors)

def dcg(scores):
    """计算 Discounted Cumulative Gain (DCG)"""
    scores = np.array(scores)   
    return np.sum((2**scores - 1) / np.log2(np.arange(2, len(scores) + 2)))

def ndcg(pred_scores, true_scores):
    """计算 Normalized Discounted Cumulative Gain (NDCG)"""
    pred_scores = np.array(pred_scores)   
    true_scores = np.array(true_scores)   
    actual_dcg = dcg(pred_scores)
    ideal_dcg = dcg(sorted(true_scores, reverse=True))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


 



def NDCG_20(predictions, labels,k=20):
    print(print(predictions.shape, labels.shape))
    predictions = predictions.squeeze().detach().cpu().numpy()
    labels = labels.squeeze().detach().cpu().numpy()
    if len(predictions) < k:
        return -1

     
    num_groups = len(predictions) // 20

     
    ndcg_values = []

    for i in range(num_groups):
         
        group_pred = predictions[i * 20: (i + 1) * 20]
        group_label = labels[i * 20: (i + 1) * 20]
        
         
        current_ndcg = ndcg_score([group_label], [group_pred], k=k)
        ndcg_values.append(current_ndcg)

     
    average_ndcg = sum(ndcg_values) / len(ndcg_values)
    print("Average NDCG:", average_ndcg)
    return average_ndcg

def NDCG_k(predictions, labels,k=20):
    predictions = predictions.squeeze().detach().cpu().numpy()
    labels = labels.squeeze().detach().cpu().numpy()
    if len(predictions) < k:
        return -1
    ndcg = ndcg_score([labels], [predictions], k=k)
    return ndcg