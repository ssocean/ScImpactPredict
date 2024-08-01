from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from furnace.arxiv_paper import Arxiv_paper, ArxivPaperMapping

Base = declarative_base()



# 创建数据库引擎
engine = create_engine('mysql+mysqlconnector://root:1q2w3e4r5t@localhost/literaturedatabase')

# 创建数据库表
Base.metadata.create_all(engine)

# 创建会话
Session = sessionmaker(bind=engine)
session = Session()
paper = Arxiv_paper(ref_obj='segment anything', ref_type='title')
paper.entity
print(paper.authors)
# 创建Document对象
doc = ArxivPaperMapping(paper)



# ... 设置其他属性

# 将对象添加到会话
session.add(doc)

# 提交更改以将对象持久化到数据库
session.commit()

# 关闭会话
session.close()