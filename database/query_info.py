import os
import re

from tqdm import tqdm

from database.DBEntity import PaperMapping
from furnace.arxiv_paper import Arxiv_paper

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from furnace.arxiv_paper import Arxiv_paper
from furnace.google_scholar_paper import Google_paper
from tools.gpt_util import *

Base = declarative_base()



# 创建数据库引擎
engine = create_engine('mysql+mysqlconnector://root:1q2w3e4r5t@localhost/literaturedatabase')

# 创建数据库表
Base.metadata.create_all(engine)

# 创建会话
Session = sessionmaker(bind=engine)
session = Session()


def add_info_to_database(dir:str,session):
    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)
            if match:
                arxiv_id = match.group()
                cur_paper = Arxiv_paper(arxiv_id, ref_type='id')
                _ = cur_paper.entity

                doc = PaperMapping(cur_paper)

                # ... 设置其他属性

                # 将对象添加到会话
                session.add(doc)

                # 提交更改以将对象持久化到数据库
                session.commit()
            else:
                print("No arXiv ID found.")


    # 关闭会话
    session.close()
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import time
import random
def update_keyword(dir:str,session):
    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            # print(filename)
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)

            title_parts = filename.split('.')
            if match:
                arxiv_id = '.'.join(title_parts[:2])
                id = 'http://arxiv.org/abs/'+arxiv_id

                # 创建会话工厂


                # id_value = 1  # 指定的ID值
                data = session.query(PaperMapping).filter(PaperMapping.id == id).first()
                # print(data.idLiterature)
                if data and data.gpt_keywords is None:
                    time.sleep(random.uniform(5.5, 8))
                    try:
                        kwd = get_chatgpt_keyword(data.title, data.abstract.replace('\n',''))
                    except Exception as e:
                        print(e)
                        print(data.title)
                        print(data.abstract)
                        return
                    # print(','.join(kwd))  # 根据实际列名修改
                    # kwd=''
                    data.gpt_keywords = ','.join(kwd)
                    # 提交更改
                    # print(','.join(kwd))
                    try:
                        if data.citation_count is None:
                            g_paper = Google_paper(data.title)

                            if g_paper.citation_count>=0:
                                data.citation_count = g_paper.citation_count
                                # print(data.citation_count)
                            if g_paper.publication_source != 'NA':
                                data.publication_source = g_paper.publication_source
                                # print(data.publication_source)

                    except KeyError as E:
                        print(E)
                        return
                    #
                    session.commit()
                    # print("数据更新成功")
                else:
                    pass
                    # print("未找到要更新的数据或该数据已存在无需更新")
                # 输出查询结果
                # for row in data:


            # 关闭会话
            session.close()


    # 关闭会话
    session.close()
# update_keyword(r'D:\download_paper',session)
def statistics_database(session):
    rst = {}
    results = session.query(PaperMapping).all()
    years = []
    ref_counts = []
    citation_counts = []
    pubs = []
    authors_num = []
    for row in tqdm(results):
        if row.is_review == 1 and row.valid == 1:
            if row.publication_date.year is not None:
                years.append(int(row.publication_date.year))
            # print(row.publication_date.year)
            if row.s2_reference_count is not None:
                if row.s2_reference_count >=1:
                    ref_counts.append(int(row.s2_reference_count))
            if row.s2_citation_count is not None:
                citation_counts.append(int(row.s2_citation_count))
            if row.s2_pub_info is not None:
                pubs.append(row.s2_pub_info)
            if row.authors_num is not None:
                authors_num .append(row.authors_num)
    rst['years'] = years
    rst['ref_counts'] = ref_counts
    rst['citation_counts'] = citation_counts
    rst['pubs'] = pubs
    rst['authors_num'] = authors_num
    return rst
# statistics_database(session)