import json
import uuid
from datetime import datetime
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from furnace.Author import Author
from furnace.arxiv_paper import Arxiv_paper
from furnace.semantic_scholar_paper import S2paper


engine = create_engine('mysql+mysqlconnector://root:1q2w3e4r5t@localhost/scitepredict')
Base = declarative_base()



class PaperMapping(Base):
    __tablename__ = 'literature'  # 映射到数据库中的表名

    idLiterature = Column(String(36), primary_key=True)
    arxiv_id = Column(String(45), primary_key=True)
    title = Column(String(255))
    publication_date = Column(DateTime, default=None)

    citation_count = Column(Integer, default=0)
    pub_url = Column(String(255))

    authors = Column(Text)
    TNCSI = Column(Float, default=None)

    downloaded_pth = Column(Text, default=None)
    categories = Column(String(255))
    valid = Column(Integer, default=0)
    last_update_time = Column(DateTime, default=datetime.now())

    key_idea = Column(Text)
    key_fig = Column(Integer)
    # 转2进制 减少网络传输数据量
    paper_type = Column(Integer, default=0)
    authors_title = Column(Text)
    benchmark = Column(Text)
    oa = Column(Text)
    raw_resp = Column(Text)
    search_by_keywords = Column(Text)
    gpt_keyword = Column(Text)
    def __init__(self, arxiv_paper: Arxiv_paper = None,downloaded_pth:str=None,search_by_keywords:str=None):
        if arxiv_paper is not None:
            self.idLiterature = str(uuid.uuid4())  # 生成UUID并转为字符串
            self.title = arxiv_paper.title
            self.publication_date = arxiv_paper.publication_date
            self.citation_count = None  # 初始化引用次数
            self.pub_url = arxiv_paper.pub_url
            self.arxiv_id = arxiv_paper.id
            self.authors = '#'.join([str(author) for author in arxiv_paper.authors])
            self.TNCSI = None  # 默认TNCSI值，如果需要不同的逻辑可以修改
            self.downloaded_pth = downloaded_pth   # 初始化为空
            self.categories = '#'.join([str(cat) for cat in arxiv_paper.categories])
            self.valid = -1  # -1 需要被评估 0 已经被评估 无效 1 评估后 有效
            self.last_update_time = datetime.now()  # 设置当前时间为最后更新时间
            self.key_idea = None  # 初始化关键思想为空
            self.key_fig = None  # 初始化关键图像计数为0
            self.paper_type = None  # 初始化纸张类型，默认值为0
            self.authors_title = None  # 初始化作者标题，默认值为0
            self.benchmark = None  # 初始化基准为空
            self.oa = None
            self.raw_resp = None
            self.search_by_keywords = search_by_keywords
        else:
            raise ValueError("A valid Arxiv_paper object must be provided")


class PaperMapping_detail(Base):
    __tablename__ = 'literature_detail'  # 映射到数据库中的表名

    idLiterature_detail = Column(String(36), primary_key=True)
    fig_caps = Column(Text)
    author_details = Column(Text)
    faiss_db_pth = Column(Text)
    table_content = Column(Text)
    abstract = Column(Text)
    language = Column(String(20), default='eng')
    s2_id = Column(String(45))
    publisher = Column(String(255))
    publication_source = Column(String(255))
    source_type = Column(String(255))
    keywords = Column(String(255))
    references = Column(Text)
    comment = Column(Text)
    journal_ref = Column(String(255))
    gpt_keywords = Column(String(255))
    links = Column(Text)
    search_by_keywords = Column(String(255))
    s2_citation_count = Column(Integer)
    s2_publication_date = Column(DateTime, default=None)
    s2_tldr = Column(Text)
    s2_DOI = Column(String(255))
    s2_pub_info = Column(Text)
    s2_pub_url = Column(String(255))
    s2_reference_count = Column(Integer)
    s2_field = Column(String(255))
    s2_influential_citation_count = Column(Integer)
    reference_details = Column(Text)
    authors_num = Column(Integer)
    gpt_keyword = Column(String(255))

    def __init__(self,idLiterature:str,  arxiv_paper: Arxiv_paper = None, s2_paper: S2paper = None,search_by_keywords=None):
        self.fig_caps = None
        self.author_details = None
        self.faiss_db_pth = None
        self.table_content = None
        self.search_by_keywords = search_by_keywords
        self.idLiterature_detail = idLiterature

        if arxiv_paper is not None:
              # 生成UUID并转为字符串
            self.abstract = arxiv_paper.abstract
            self.comment =arxiv_paper.comment
            self.journal_ref = arxiv_paper.journal_ref
            self.links = arxiv_paper.links[0].href

        if s2_paper is not None:
            self.s2_id = s2_paper.s2id
            self.s2_tldr = s2_paper.tldr
            self.s2_DOI = s2_paper.DOI
            self.s2_pub_info = s2_paper.publication_source
            self.s2_citation_count = s2_paper.citation_count
            self.s2_reference_count = s2_paper.reference_count
            self.s2_field = s2_paper.field
            self.s2_influential_citation_count = s2_paper.influential_citation_count
            self.authors_num = len(s2_paper.authors)
        self.last_update_time = datetime.now()





class Author(Base):
    __tablename__ = 'author'
    idAuthor = Column(String(36), primary_key=True)
    gs_id = Column(String(64))
    name = Column(String(255))
    affiliation = Column(Text)
    title = Column(Text)
    citation_count = Column(Integer)
    access_date = Column(DateTime, default=datetime.now())
    homepage = Column(Text)

    def __init__(self, name: str, affiliation: str):
        self.idAuthor = str(uuid.uuid4())  # 生成UUID并转为字符串
        self.gs_id = None
        self.name = name
        self.affiliation = affiliation
        self.title = None
        self.citation_count = None
        self.homepage = None
        self.access_date = datetime.now()  # 设置当前时间为访问时间

class AoP(Base):
    __tablename__ = 'aop'
    idAoP = Column(String(36), primary_key=True)
    idAuthor= Column(String(36))
    idLiterature = Column(String(36))


    def __init__(self, idAuthor:str, idLiterature: str=None):
        self.idAoP =  str(uuid.uuid4())
        self.idAuthor = idAuthor # 生成UUID并转为字符串
        self.idLiterature = idLiterature




Base.metadata.create_all(engine)
#
#
# class Author(Base):
#     __tablename__ = 'author'  # 映射到数据库中的表名
#     idAuthor = Column(String(36), primary_key=True)
#     gs_id = Column(String(64))
#     name = Column(String(255))
#     affiliation = Column(Text)
#     s2Id = Column(String(100))
#     orcid = Column(String(100))
#     citation_count = Column(Integer)
#     paper_count  = Column(Integer)
#     h_index = Column(Integer)
#     h_index_l5 = Column(Integer)
#     i10 = Column(Integer)
#     i10_l5 = Column(Integer)
#     access_date = Column(DateTime, default=datetime.now())
#
#     def __init__(self, author: Author):
#         # 生成UUID
#         idAuthor = uuid.uuid4()
#
#         # 将UUID转换为字符串
#         self.idAuthor = str(idAuthor)
#         self.name = author.name
#         # self.affiliation = author.affiliation
#         # self.schorlarId = author.scholar_id
#         # self.orcid = author.orcid
#         # self.citedBy = author.cited_by
#         # self.h_index = author.h_index
#         # self.h_index_l5 = author.h_index_l5
#         # self.i10 = author.i10
#         # self.i10_l5 = author.i10_l5
#
#
#
#
#
#
#class Author_detail(Base):
#     __tablename__ = 'author_detail'
#     idAuthor_detail = Column(String(36), primary_key=True)
#     s2Id = Column(String(100))
#     orcid = Column(String(100))
#     paper_count = Column(Integer)
#     h_index = Column(Integer)
#     h_index_l5 = Column(Integer)
#     i10 = Column(Integer)
#     i10_l5 = Column(Integer)
#     bio = Column(Text)
#     access_date = Column(DateTime, default=datetime.now())
#
#     def __init__(self, idAuthor:str, bio: str=None):
#         self.idAuthor_detail = idAuthor # 生成UUID并转为字符串
#         self.s2Id = None
#         self.orcid = None
#         self.paper_count = None
#         self.h_index = None
#         self.h_index_l5 = None
#         self.i10 = None
#         self.i10_l5 = None
#         self.bio = bio
#         self.access_date = datetime.now()  # 设置当前时间为访问时间
#
#
#
# class AoP(Base):
#     __tablename__ = 'aop'
#     idAoP = Column(String(36), primary_key=True)
#     idAuthor= Column(String(36))
#     idLiterature = Column(String(36))
#
#
#     def __init__(self, idAuthor:str, idLiterature: str=None):
#         self.idAoP =  str(uuid.uuid4())
#         self.idAuthor = idAuthor # 生成UUID并转为字符串
#         self.idLiterature = idLiterature
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# class PaperMapping(Base):
#     __tablename__ = 'literature'  # 映射到数据库中的表名
#
#     idLiterature = Column(String(36), primary_key=True)
#     hash_key = Column(Integer)  # 我们将要添加这个列
#     title = Column(String(255))
#     publication_date = Column(DateTime, default=None)
#     language = Column(String(20), default='eng')
#     id = Column(String(45))
#     publisher = Column(String(255))
#     publication_source = Column(String(255))
#     source_type = Column(String(255))
#     keywords = Column(String(255))
#     abstract = Column(Text)
#     citation_count = Column(Integer)
#     references = Column(Text)
#     pub_url = Column(String(255))
#     comment = Column(Text)
#     journal_ref = Column(String(255))
#     authors = Column(Text)
#     gpt_keywords = Column(String(255))
#     categories = Column(String(255))
#     links = Column(Text)
#
#     search_by_keywords = Column(String(255))
#
#     s2_id = Column(String(45))
#     s2_publication_date = Column(DateTime, default=None)
#     s2_tldr = Column(String(255))
#     s2_DOI = Column(String(255))
#     s2_pub_info = Column(Text)
#     s2_pub_url = Column(String(255))
#     s2_citation_count = Column(Integer)
#     s2_reference_count = Column(Integer)
#     s2_field = Column(String(255))
#     s2_influential_citation_count = Column(Integer)
#     valid = Column(Integer)
#     last_update_time = Column(DateTime, default=datetime.now())
#     reference_details = Column(Text)
#     authors_num = Column(Integer)
#     has_been_downloaded = Column(Integer, default=0)
#     gpt_keyword = Column(String(255))
#     TNCSI = Column(Float,default=0)
#
#     TYPE = Column(Integer)
#     def __init__(self, arxiv_paper: Arxiv_paper = None, s2_paper: S2paper = None, search_by_keywords=None):
#         if arxiv_paper is not None:
#             # 生成UUID
#             idLiterature = uuid.uuid4()
#
#             # 将UUID转换为字符串
#             self.idLiterature = str(idLiterature)
#             self.hash_key = int(self.idLiterature.replace('-', '')[:8], 16)
#             self.title = arxiv_paper.title
#             self.publication_date = arxiv_paper.publication_date
#             self.language = arxiv_paper.language
#             self.id = arxiv_paper.id
#             self.publisher = arxiv_paper.publisher
#             self.publication_source = arxiv_paper.publication_source
#             self.source_type = arxiv_paper.source_type
#             self.keywords = arxiv_paper.keywords
#             self.abstract = arxiv_paper.abstract
#             self.citation_count = arxiv_paper.citation_count
#             self.references = arxiv_paper.references
#             self.pub_url = arxiv_paper.pub_url
#             self.comment = arxiv_paper.comment
#             self.journal_ref = arxiv_paper.journal_ref
#             self.primary_category = arxiv_paper.primary_category
#             self.categories = arxiv_paper.categories
#             self.links = arxiv_paper.links
#             self.authors = '#'.join([str(i) for i in arxiv_paper.authors])
#             self.gpt_keywords = None
#             self.gpt_keyword = None
#             self.aop = None
#             self.search_by_keywords = search_by_keywords
#             self.last_update_time = datetime.now()
#             self.TNCSI = None
#
#
#             # for key, value in arxiv_paper.__dict__.items():
#             #     if not key.startswith('_'):
#             #         setattr(self, key, value)
#         if s2_paper is not None:
#             self.s2_id = s2_paper.s2id
#             self.s2_publication_date = s2_paper.publication_date
#             self.s2_tldr = s2_paper.tldr
#             self.s2_DOI = s2_paper.DOI
#             self.s2_pub_info = s2_paper.publisher#str(s2_paper.publisher) + '@' + str(s2_paper.publication_source)
#             self.s2_pub_url = s2_paper.pub_url
#             self.s2_citation_count = s2_paper.citation_count
#             self.s2_reference_count = s2_paper.reference_count
#             self.s2_field = s2_paper.field
#             self.s2_influential_citation_count = s2_paper.influential_citation_count
#             self.valid = 1
#             self.authors_num = len(s2_paper.authors)
#             self.last_update_time = datetime.now()
#
#
# from sqlalchemy import create_engine, Column, Integer, String, DateTime
# from sqlalchemy.orm import declarative_base
# from sqlalchemy.orm import sessionmaker
#
# Base = declarative_base()
#
#
# class Author(Base):
#     __tablename__ = 'author'  # 映射到数据库中的表名
#     idAuthor = Column(String(36), primary_key=True)
#     hash_key = Column(Integer)  # 我们将要添加这个列
#     gs_id = Column(String(64))
#     name = Column(String(255))
#     affiliation = Column(Text)
#     s2Id = Column(String(100))
#     orcid = Column(String(100))
#     citation_count = Column(Integer)
#     paper_count  = Column(Integer)
#     h_index = Column(Integer)
#     h_index_l5 = Column(Integer)
#     i10 = Column(Integer)
#     i10_l5 = Column(Integer)
#     access_date = Column(DateTime, default=datetime.now())
#
#     def __init__(self, author: Author):
#         # 生成UUID
#         idAuthor = uuid.uuid4()
#
#         # 将UUID转换为字符串
#         self.idAuthor = str(idAuthor)
#         self.name = author.name
#         # self.affiliation = author.affiliation
#         # self.schorlarId = author.scholar_id
#         # self.orcid = author.orcid
#         # self.citedBy = author.cited_by
#         # self.h_index = author.h_index
#         # self.h_index_l5 = author.h_index_l5
#         # self.i10 = author.i10
#         # self.i10_l5 = author.i10_l5
#
#         # for key, value in author.__dict__.items():
#         #     if not key.startswith('_'):
#         #         setattr(self, key, value)
#
#
# class AoP(Base):
#     __tablename__ = 'aop'  # 映射到数据库中的表名
#
#     idaop = Column(String(36), primary_key=True)
#     author1 = Column(String(36))
#     author2 = Column(String(36))
#     author3 = Column(String(36))
#     author4 = Column(String(36))
#     author5 = Column(String(36))
#     author6 = Column(String(36))
#     author7 = Column(String(36))
#     author8 = Column(String(36))
#     author9 = Column(String(36))
#     author10 = Column(String(36))
#
#     def __init__(self, author_lst):
#         # 生成UUID
#         self.idaop = str(uuid.uuid4())
#         num_of_author = len(author_lst)
#         if num_of_author <= 10:
#             for i, author in enumerate(author_lst):
#                 setattr(self, 'author' + str(i + 1), author.idAuthor)
#         else:
#             for i, author in enumerate(author_lst[:7]):
#                 setattr(self, 'author' + str(i + 1), author.idAuthor)
#             setattr(self, 'author8', author_lst[8].idAuthor)
#             setattr(self, 'author9', author_lst[9].idAuthor)
#             setattr(self, 'author10', author_lst[10].idAuthor)
#
#
# class RefMapping(Base):
#     __tablename__ = 'reference'  # 映射到数据库中的表名
#
#     idReference = Column(String(36), primary_key=True)
#     title = Column(String)
#     s2_publication_date = Column(String(36))
#     s2_id = Column(String(45))
#     s2_pub_info = Column(String)
#     s2_citation_count = Column(Integer)
#     s2_reference_count = Column(Integer)
#     s2_field= Column(String(255))
#     s2_influential_citation_count = Column(Integer)
#     keywords = Column(String(255))
#     gpt_keywords = Column(String(255))
#     contexts = Column(String)
#     intents = Column(String(45))
#     isInfluential = Column(Integer)
#
#     def __init__(self, s2paper,contexts,intents,isInfluential):
#         # 生成UUID
#         self.idReference = str(uuid.uuid4())
#         self.title = s2paper.title
#         self.s2_publication_date = s2paper.publication_date
#         self.s2_id = s2paper.s2id
#         self.s2_pub_info = s2paper.publication_source
#         self.s2_pub_url = s2paper.pub_url
#         self.s2_citation_count = s2paper.citation_count
#         self.s2_reference_count = s2paper.reference_count
#         self.s2_field = s2paper.field
#         self.s2_influential_citation_count = s2paper.influential_citation_count
#         self.keywords = None
#         self.gpt_keywords = None
#         self.contexts = '$&#'.join(contexts)
#         self.intents = ';'.join(intents)
#         self.isInfluential = 0 if isInfluential else 1
#
# class CoP(Base):
#     __tablename__ = 'cop'  # 映射到数据库中的表名
#     idCoP = Column(String(36), primary_key=True)
#     s2_id = Column(String(45),unique=True)
#     citation = Column(String)
#     full_citation = Column(String)
#
#
#     def __init__(self, s2_id,citation):
#         # 生成UUID
#         self.idCoP = str(uuid.uuid4())
#         self.s2_id = s2_id
#         self.citation = citation
#         self.full_citation = None
#
# class RoP(Base):
#     __tablename__ = 'rop'  # 映射到数据库中的表名
#
#     idrop = Column(String(36), primary_key=True)
#     idLiterature = Column(String(36))
#     idReference = Column(String(36))
#
#
#     def __init__(self, idLiterature, idReference):
#         # 生成UUID
#         self.idrop = str(uuid.uuid4())
#         self.idLiterature = idLiterature
#         self.idReference = idReference
#
#
# class PaperMapping(Base):
#     __tablename__ = 'literature'  # 映射到数据库中的表名
#
#     idLiterature = Column(String(36), primary_key=True)
#     title = Column(String(255))
#     publication_date = Column(DateTime, default=None)
#     language = Column(String(20), default='eng')
#     id = Column(String(45))
#     publisher = Column(String(255))
#     publication_source = Column(String(255))
#     source_type = Column(String(255))
#     keywords = Column(String(255))
#     abstract = Column(Text)
#     categories  = Column(String(255))
#     citation_count = Column(Integer)
#     references = Column(Text)
#     pub_url = Column(String(255))
#     comment = Column(Text)
#     journal_ref = Column(String(255))
#     authors = Column(Text)
#     gpt_keywords = Column(String(255))
#
#     search_by_keywords = Column(String(255))
#
#     s2_id = Column(String(45))
#     s2_publication_date = Column(DateTime, default=None)
#     s2_tldr = Column(String(255))
#     s2_DOI = Column(String(255))
#     s2_pub_info = Column(Text)
#     s2_pub_url = Column(String(255))
#     s2_citation_count = Column(Integer)
#     s2_reference_count = Column(Integer)
#     s2_field = Column(String(255))
#     s2_influential_citation_count = Column(Integer)
#     valid = Column(Integer)
#     last_update_time = Column(DateTime, default=datetime.now())
#     reference_details = Column(Text)
#     authors_num = Column(Integer)
#     has_been_downloaded = Column(Integer)
#     gpt_keyword = Column(String(255))
#     TNCSI = Column(Float)
#
#     def __init__(self, arxiv_paper: Arxiv_paper = None, s2_paper: S2paper = None, search_by_keywords=None):
#         if arxiv_paper is not None:
#             # 生成UUID
#             idLiterature = uuid.uuid4()
#
#             # 将UUID转换为字符串
#             self.idLiterature = str(idLiterature)
#             self.title = arxiv_paper.title
#             self.publication_date = arxiv_paper.publication_date
#             self.language = arxiv_paper.language
#             self.id = arxiv_paper.id
#             self.publisher = arxiv_paper.publisher
#             self.publication_source = arxiv_paper.publication_source
#             self.source_type = arxiv_paper.source_type
#             self.keywords = arxiv_paper.keywords
#             self.abstract = arxiv_paper.abstract
#             self.citation_count = arxiv_paper.citation_count
#             self.references = arxiv_paper.references
#             self.pub_url = arxiv_paper.pub_url
#             self.comment = arxiv_paper.comment
#             self.journal_ref = arxiv_paper.journal_ref
#             self.primary_category = arxiv_paper.primary_category
#             self.categories = arxiv_paper.categories
#             self.authors = '#'.join([str(i) for i in arxiv_paper.authors])
#             self.gpt_keywords = None
#             self.gpt_keyword = None
#             self.aop = None
#             self.search_by_keywords = search_by_keywords
#             self.is_review = 1
#             self.last_update_time = datetime.now()
#             self.TNCSI = None
#
#             for key, value in arxiv_paper.__dict__.items():
#                 if not key.startswith('_'):
#                     setattr(self, key, value)
#         if s2_paper is not None:
#             self.s2_id = s2_paper.s2id
#             self.s2_publication_date = s2_paper.publication_date
#             self.s2_tldr = s2_paper.tldr
#             self.s2_DOI = s2_paper.DOI
#             self.s2_pub_info = s2_paper.publisher  # str(s2_paper.publisher) + '@' + str(s2_paper.publication_source)
#             self.s2_pub_url = s2_paper.pub_url
#             self.s2_citation_count = s2_paper.citation_count
#             self.s2_reference_count = s2_paper.reference_count
#             self.s2_field = s2_paper.field
#             self.s2_influential_citation_count = s2_paper.influential_citation_count
#             self.valid = 1
#             self.authors_num = len(s2_paper.authors)
#             self.last_update_time = datetime.now()

# 创建数据库表
