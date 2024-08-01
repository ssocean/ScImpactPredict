# from scipy.stats import kendalltau, spearmanr

# # 假设有五篇文章
# input_seq = ['e1', 'e2', 'e3', 'e4', 'e5']
# GT_seq = ['e1', 'e3', 'e2', 'e4', 'e5']

# # 将这些序列转换为排名
# input_indices = [GT_seq.index(e) for e in input_seq]
# GT_indices = list(range(len(GT_seq)))

# # 计算 Kendall's Tau
# tau, _ = kendalltau(input_indices, GT_indices)
# # 计算 Spearman's Rank Correlation
# rho, _ = spearmanr(input_indices, GT_indices)

# print(tau, rho)
# import torch
# from scipy.stats import kendalltau, spearmanr

# predictions = torch.tensor([0.3740, 0.6719, 0.7295, 0.6670, 0.3157, 0.4370, 0.3186, 0.1973, 0.4268,
#          0.4021, 0.3049, 0.4221, 0.3308, 0.2876, 0.7212, 0.6899, 0.6099, 0.7397,
#          0.5444, 0.4407], dtype=torch.float16)
# labels = torch.tensor([0.4658, 0.6317, 0.7254, 0.8656, 0.3894, 0.5137, 0.3535, 0.5224, 0.6084,
#          0.4363, 0.1559, 0.3021, 0.1494, 0.1836, 0.8819, 0.9975, 0.5475, 0.9608,
#          0.6842, 0.6089])

# # 对预测值和实际值进行排序，获取索引
# _, pred_indices = predictions.sort()
# _, label_indices = labels.sort()
# print(pred_indices,label_indices)
# # 将tensor转换为列表，便于后续处理
# pred_indices = pred_indices.tolist()
# label_indices = label_indices.tolist()

# # 计算 Kendall's Tau
# tau, _ = kendalltau(pred_indices, label_indices)
# # 计算 Spearman's Rank Correlation
# rho, _ = spearmanr(pred_indices, label_indices)

# print("Kendall's Tau:", tau, "Spearman's Rank Correlation:", rho)
import numpy as np



import torch
import numpy as np

def average_rank_error(pred_indices, true_indices):
    """计算平均排名误差"""
    errors = np.abs(np.array(pred_indices) - np.array(true_indices))
    return np.mean(errors)

def dcg(scores):
    """计算 Discounted Cumulative Gain (DCG)"""
    scores = np.array(scores)  # 确保scores是numpy数组
    return np.sum((2**scores - 1) / np.log2(np.arange(2, len(scores) + 2)))

def ndcg(pred_scores, true_scores):
    """计算 Normalized Discounted Cumulative Gain (NDCG)"""
    pred_scores = np.array(pred_scores)  # 确保pred_scores是numpy数组
    true_scores = np.array(true_scores)  # 确保true_scores是numpy数组
    actual_dcg = dcg(pred_scores)
    ideal_dcg = dcg(sorted(true_scores, reverse=True))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


 

# def NDCG_20(predictions,labels):
#     # 排序并获取索引
#     _, pred_indices = torch.sort(predictions)
#     _, true_indices = torch.sort(labels)

#     # 将tensor转换为numpy数组
#     pred_indices = pred_indices.numpy()
#     true_indices = true_indices.numpy()

#     # # 计算平均排名误差
#     # average_error = average_rank_error(pred_indices, true_indices)
#     # print("Average Ranking Error:", average_error)

#     # 计算NDCG，我们需要分数，这里使用排名作为分数
#     pred_scores = [predictions[i].item() for i in pred_indices]
#     true_scores = [labels[i].item() for i in true_indices]

#     # 计算NDCG
#     ndcg_value = ndcg(pred_scores, true_scores)
#     print("NDCG:", ndcg_value)
#     return ndcg_value
import numpy as np



import torch
import numpy as np


from sklearn.metrics import ndcg_score
def NDCG_20(predictions, labels,k=20):
    # predictions = torch.cat(tuple(predictions), dim=0).squeeze().detach().cpu().numpy()
    # labels = torch.cat(tuple(labels), dim=0).squeeze().detach().cpu().numpy()
    print(print(predictions.shape, labels.shape))
    predictions = predictions.squeeze().detach().cpu().numpy()
    labels = labels.squeeze().detach().cpu().numpy()
    if len(predictions) < k:
        return -1

    # 计算总组数（只保留完整的组）
    num_groups = len(predictions) // 20

    # 存储每组的NDCG值
    ndcg_values = []

    for i in range(num_groups):
        # 对每组分别计算NDCG
        group_pred = predictions[i * 20: (i + 1) * 20]
        group_label = labels[i * 20: (i + 1) * 20]
        
        # 计算当前组的NDCG[]
        current_ndcg = ndcg_score([group_label], [group_pred], k=k)
        ndcg_values.append(current_ndcg)

    # 计算所有完整组的平均NDCG值
    average_ndcg = sum(ndcg_values) / len(ndcg_values)
    print("Average NDCG:", average_ndcg)
    return average_ndcg

def NDCG_k(predictions, labels,k=20):

    # print(print(predictions.shape, labels.shape))
    predictions = predictions.squeeze().detach().cpu().numpy()
    labels = labels.squeeze().detach().cpu().numpy()
    if len(predictions) < k:
        return -1

    # 计算当前组的NDCG[]
    ndcg = ndcg_score([labels], [predictions], k=k)



    # print("Average NDCG:", ndcg)
    return ndcg