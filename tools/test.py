# import datetime
# import time
# from tools.Reference import Ref, get_paper_info_from_REST
# from tools.ref_utils import *
#
# from tqdm import tqdm
#
# ref_anchor = []
# # get_paper_info_from_REST('Cervical Cancer Detection Using SVM Based Feature Screening')
# # print('done')
# # Ref('Cervical Cancer Detection Using SVM Based Feature Screening',fill_param=True)
#
# from scholarly import ProxyGenerator
# from scholarly import scholarly
# from scholarly import scholarly, ProxyGenerator
#
# # pg = ProxyGenerator()
# # success = pg.FreeProxies()
# # scholarly.use_proxy(pg)
# # print('Proxy Enabled!')
# # author = next(scholarly.search_author('Steven A Cholewiak'))
# # scholarly.pprint(author)
# # search = get_arxiv('all:cervical cancer image segmentation', 2000)
# # rst = filter_arxiv(search, 'cervical cancer cell segmentation',end_time=datetime.datetime(2019, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc))
# # ref_eval = arxiv2ref(rst)
#
#
# ref_eval = get_google('cervical cancer image segmentation', year_high=2018, topk=10)
#
# avg_cite = get_avg_cite_num(ref_eval, year_high=2018)
#
# ref_text = extract_ref(r"C:\Users\Ocean\Downloads\1811.00849.pdf")
# ref_list = get_ref_list(ref_text)
# print(ref_list)
# import random
#
# for ref_text in tqdm(ref_list):
#     if ref_text == '':
#         continue
#
#     ref = Ref(ref_text, fill_param=True)
#
#     # time.sleep(2)
#     ref_anchor.append(ref)
#     #
#     # time.sleep(random.randint(2,5))
# # Algorithms for screening of Cervical Cancer
# # ref_eval = [Ref('Automated Segmentation of Cervical Nuclei in Pap Smear Images using Deformable Multi-path Ensemble Model',extract_title=False),
# # Ref('DeepPap: Deep Convolutional Networks for Cervical Cell Classification',extract_title=False),
# # Ref('Comparison-Based Convolutional Neural Networks for Cervical Cell/Clumps Detection in the Limited Data Scenario',extract_title=False),
# # Ref('Algorithms for screening of Cervical Cancer: A chronological review',extract_title=False),
# # Ref('Genaue modellbasierte Identifikation von gynäkologischen Katheterpfaden für die MRT-bildgestützte Brachytherapie',extract_title=False),]
#
#
# cocite_ratio = cal_cocite_rate(ref_anchor, ref_eval)
# print(cocite_ratio)
#
# weighted_cite_ratio = cal_weighted_cocite_rate(ref_anchor, ref_eval, avg_cite)
# print(weighted_cite_ratio)


import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv('data_for_model_balanced_2.csv')

# 绘制 TNCSI 的直方图
plt.figure(figsize=(10, 6))
plt.hist(df['TNCSI'], bins=20, color='blue', alpha=0.7)
plt.title('TNCSI Distribution')
plt.xlabel('TNCSI')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



# from transformers import RobertaTokenizer, RobertaModel
# import torch
# tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
# model = RobertaModel.from_pretrained('roberta-large')
#
# # Test with different inputs
# text1 = "Hello world"
# text2 = "Goodbye world"
#
# inputs1 = tokenizer(text1, return_tensors="pt")
# inputs2 = tokenizer(text2, return_tensors="pt")
#
# outputs1 = model(inputs1['input_ids'])
# outputs2 = model(inputs2['input_ids'])
#
# print(outputs1)  # First layer, first token embeddings
# print(outputs2)  # First layer, first token embeddings
#
# # Check if the outputs are different
# assert not torch.equal(outputs1.hidden_states[0], outputs2.hidden_states[0]), "The outputs are identical!"
