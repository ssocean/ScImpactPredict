import pandas as pd

from tools.test import get_filename_without_extension
import copy
import json
import os
from urllib.error import URLError

import requests
import tiktoken
from bs4 import BeautifulSoup
from langchain_core.exceptions import OutputParserException

def get_filename_without_extension(file_path):
    # Extract the filename without extension
    filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    return filename_without_extension


import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.orm import declarative_base

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field

from typing import Dict, List
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import langchain

from langchain.chains import LLMChain

import os

from langchain.prompts import PromptTemplate

from langchain.chains.question_answering import load_qa_chain


import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from retry import retry

langchain.debug = False
import arxiv

import time

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import glob

from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.chat_models import ChatOpenAI

from database.DBEntity import *
from furnace.arxiv_paper import Arxiv_paper, get_arxiv_id_from_url
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, scoped_session
import logging
import datetime

engine = create_engine('xxx/scitepredict')

Base = declarative_base()
 
Base.metadata.create_all(engine)

 
Session = sessionmaker(bind=engine)
session = Session()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session_factory = scoped_session(SessionLocal)   
import PyPDF2
items = [
    "Engaging", "Controversial", "Rigorous", "Innovative", "Accessible", "Methodical", "Concise", "Persuasive",
    "Comprehensive", "Insightful", "Relevant", "Objective", "Replicable", "Structured", "Coherent", "Original",
    "Balanced", "Authoritative", "Impactful", "Interdisciplinary", "Well-sourced", "Technical", "Provocative",
    "Hypothesis-driven", "Ethical", "Difficult to understand", "Exciting", "Not well written", "Theoretical", "To the point",
    "Disengaging", "Uncontroversial", "Lax", "Conventional", "Inaccessible", "Haphazard", "Verbose", "Unconvincing",
    "Superficial", "Uninsightful", "Irrelevant", "Subjective", "Non-replicable", "Unstructured", "Incoherent", "Derivative",
    "Unbalanced", "Unreliable", "Inconsequential", "Narrow", "Poorly-sourced", "Nontechnical", "Unprovocative",
    "Speculation-driven", "Unethical", "Easy to understand", "Dull", "Well written", "Empirical", "Circumlocutory"
]

def parse_scores(content):
     
    try:
        scores = [int(line.split()[1]) for line in content.split('\n')]
         
        mean_score = sum(scores) / len(scores)
        return mean_score
    except Exception as e:
        print(e)
        return 0

def paper_rating(abstract):
    # download_paper(row, out_dir=r'J:\arxiv')

    prompt = f"Please rate the following abstract on each of the 60 items from 0 = Not at all to 100 = Very much. Only provide the numbers. For example:\n\n"
    prompt += "1. 65\n2. 50\n3. 5\n4. 95\n5. …\n\n"
    prompt += f"This is the abstract:\n{abstract}\n\n"
    prompt += "These are the items:\n" + "\n".join([f"{i + 1}. {item}" for i, item in enumerate(items)])
    prompt_template = prompt

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0) # Lite Speed
    rst = llm.invoke(prompt_template)
    content = rst.content

    return parse_scores(content)


def main():
     
    data = pd.read_csv(r'xxx\NAID\NAID_test_extrainfo.csv')

     
    scores = []

     
    with ThreadPoolExecutor(max_workers=10) as executor:   
         
        future_to_abstract = {executor.submit(paper_rating, abstract): abstract for abstract in data['abstract']}

         
        for future in tqdm(as_completed(future_to_abstract)):
            score = future.result()
            scores.append(score)
     
    data['average_score'] = scores

     
    columns_to_save = ['id', 'cites', 'TNCSI', 'TNCSI_SP', 'abstract', 'average_score']
    data[columns_to_save].to_csv(r'gpt_predict.csv', index=False)


import pandas as pd
from sklearn.metrics import ndcg_score
import numpy as np


def calculate_ndcg(file_path):
    
    data = pd.read_csv(file_path)

   
    if 'average_score' not in data.columns or 'cites' not in data.columns:
        return "The required columns are not in the dataframe."

    
    y_true = data['cites'].to_numpy()
    y_score = data['average_score'].to_numpy()

    # Reshape data for ndcg calculation (1, -1) as ndcg expects at least 2D arrays
    y_true = y_true.reshape(1, -1)
    y_score = y_score.reshape(1, -1)

   
    ndcg = ndcg_score(y_true, y_score,k=20)

    return ndcg




if __name__ == "__main__":
    # main()
    ndcg_value = calculate_ndcg('gpt_predict.csv')
    print(f"The NDCG value is: {ndcg_value}")