import random
import time
import warnings
import arxiv
from scholarly import scholarly
# OPENAI SETUP
import openai
import requests
import pandas as pd
import matplotlib.pyplot as plt
import string

openai.api_base = "x"
openai.api_key = "x"



def filter_punctuation(title):
     
    punctuation = string.punctuation

     
    translator = str.maketrans("", "", punctuation)

     
    filtered_title = title.translate(translator)

    return filtered_title


def get_paper_info_from_REST(paper_title, PE=PE):
     
    '''
    data['message']['items'][0]['short-container-title'] 期刊名
    data['message']['items'][0]['DOI']
    data['message']['items'][0]['is-referenced-by-count']
    data['message']['items'][0]['title']
    data['message']['items'][0]['created']['date-parts'][0]
    :param paper_title:
    :param PE:
    :return:
    '''

     
    url = f'https://api.crossref.org/works?query.title={paper_title}&mailto={PE}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        data = dict(data)
        return data
    return None


#
from retry import retry


class Ref:
    '''
    google > arxiv > crossref
    '''

    # @retry()
    def __init__(self, ref_obj, ref_type='ref'):
        self.ref_type = ref_type
        if isinstance(ref_obj, str) and (ref_type == 'ref' or ref_type == 'title'):
            self.ref_text = ref_obj
            self._arxiv_rst = None
            self._google_rst = None
            self._crossref_rst = None
        else:   
            if ref_type.lower() == 'google':
                self.ref_text = ref_obj['bib']['title'].lower()
                self.ref_type = 'title'
                self._google_rst = ref_obj
                self._crossref_rst = None
                self._arxiv_rst = None
            if ref_type.lower() == 'arxiv':
                self.ref_text = ref_obj.title.lower()
                self.ref_type = 'title'
                self._google_rst = None
                self._crossref_rst = None
                self._arxiv_rst = ref_obj

        # property
        self._cited_by = None
        self._pub_date = None
        self._title = None

        self._authors = None
        self._venue = None
        self._citations = None
        self._abstract = None
        self._pub_url = None

    def disable_arxiv(self):
        self._arxiv_rst = False

    def enable_arxiv(self):
        return self.arxiv_rst

    def disable_google(self):
        self._google_rst = False

    def enable_google(self):
        return self.google_rst

    def disable_crossref(self):
        self._crossref_rst = False

    def enable_crossref(self):
        return self._crossref_rst

    @property
    def authors(self):
        if self._authors is None:
            if self.google_rst:
                self._authors = self._google_rst['bib']['author']
            elif self.arxiv_rst:
                self._authors = [i.name for i in self._arxiv_rst.authors]
            elif self.crossref_rst:
                self._authors = self.crossref_rst['message']['items'][0]['author']
            else:
                warnings.warn(
                    "Fetching papaer info failed. No results found on google, arxiv or crossref. Please check the paper title again.",
                    UserWarning)
                self._authors = False
        return self._authors

    @property
    def venue(self):
        if self._venue is None:
            if self.google_rst:
                self._venue = self._google_rst['bib']['venue']
            elif self.arxiv_rst:
                self._venue = 'arxiv'
            elif self.crossref_rst:
                self._venue = self.crossref_rst['message']['items'][0]['container-title']
            else:
                warnings.warn(
                    "Fetching papaer info failed. No results found on google, arxiv or crossref. Please check the paper title again.",
                    UserWarning)
                self._venue = False
        return self._venue

    @property
    def citations(self):
        if self._citations is None:
            if self.google_rst:
                self._citations = self._google_rst['num_citations']
            elif self.crossref_rst:
                self._citations = self.crossref_rst['message']['items'][0]['is-referenced-by-count']
            else:
                warnings.warn(
                    "Fetching papaer info failed. No results found on google, arxiv or crossref. Please check the paper title again.",
                    UserWarning)
                self._citations = False
        return self._citations

    @property
    def abstract(self):
        if self._abstract is None:
            if self.google_rst:
                self._abstract = self._google_rst['bib']['abstract'].replace("\n", "")
            elif self.arxiv_rst:
                self._abstract = self._arxiv_rst.summary.replace("\n", "")
            elif self.crossref_rst:
                self._abstract = self.crossref_rst['message']['items'][0]["abstract"].replace("\n", "")
            else:
                warnings.warn(
                    "Fetching papaer info failed. No results found on google, arxiv or crossref. Please check the paper title again.",
                    UserWarning)
                self._abstract = False
        return self._abstract

    @property
    def pub_url(self):
        if self._pub_url is None:
            if self.google_rst:
                self._pub_url = self._google_rst['pub_url']
            elif self.arxiv_rst:
                self._pub_url = self.arxiv_rst.entry_id
            elif self.crossref_rst:
                self._pub_url = self.crossref_rst['message']['items'][0]['URL']
            else:
                warnings.warn(
                    "Fetching papaer info failed. No results found on google, arxiv or crossref. Please check the paper title again.",
                    UserWarning)
                self._pub_url = False
        return self._pub_url

    @property
    def google_rst(self):

        if self._google_rst is None:
            rst = scholarly.search_single_pub(self.title)
            if rst['bib']['title'].lower().replace(" ", "") != self.title.lower().replace(" ", ""):
                warnings.warn("Haven't fetch anything from google. Please use crossref or arxiv instead", UserWarning)
                self._google_rst = False
            else:
                self._google_rst = rst
        return self._google_rst

    @property
    def arxiv_rst(self):
        if self._arxiv_rst is None:
            search = arxiv.Search(query=f'ti:{filter_punctuation(self.title)}')

            matched_paper = next(search.results())

            if matched_paper.title.lower().replace(" ", "") != self.title.lower().replace(" ", ""):
                if self._google_rst:   
                     
                    if not self._google_rst['pub_url'] == matched_paper.entry_id:
                        warnings.warn(
                            "Haven't fetch anything from arxiv. Please use google or corssref instead",
                            UserWarning)
                        self._arxiv_rst = False
                    else:
                        self._arxiv_rst = matched_paper
                else:
                    self._arxiv_rst = False
            else:
                self._arxiv_rst = matched_paper

        return self._arxiv_rst

    @property
    def crossref_rst(self):
        if self._crossref_rst is None:
            rst = get_paper_info_from_REST(self.title)
            if rst['message']['items'][0]['title'][0].lower().replace(" ", "") != self.title.lower().replace(" ", ""):
                warnings.warn("Haven't fetch anything from crossref. Please use google or arxiv instead", UserWarning)
                self._crossref_rst = False
            else:
                self._crossref_rst = rst
        return self._crossref_rst

    @property
    def cited_by(self):
        if self._cited_by is None:
            if self.google_rst:
                self._cited_by = scholarly.citedby(self.google_rst)
            else:
                warnings.warn("Google fetching failed.", UserWarning)
                self._cited_by = False
        return self._cited_by

    @property
    def title(self):
        if self._title is None:   
            if self.ref_type == 'title':
                self._title = self.ref_text.lower()
            elif self.ref_type == 'ref':
                self._title = self.extract_title().lower()
        return self._title

    @property
    def pub_date(self):
        if self._pub_date is None:
            if self.arxiv_rst:
                self._pub_date = str(self._arxiv_rst.published.year) + '.' + str(self._arxiv_rst.published.month)
            elif self.crossref_rst:
                years = self.crossref_rst['message']['items'][0]['created']['date-parts'][0]
                self._pub_date = f'{years[0]}.{years[1]}'
            elif self.google_rst:
                self._pub_date = f"{self.google_rst['bib']['pub_year']}.1"
            else:
                warnings.warn("Can not fetch pub_date!", UserWarning)
                self._pub_date = False
        return self._pub_date

    def __str__(self):
        status = ''
        if self._google_rst:
            status += 'Ready '
        else:
            status += 'None '

        if self._arxiv_rst:
            status += 'Ready '
        else:
            status += 'None '

        if self._crossref_rst:
            status += 'Ready '
        else:
            status += 'None '

        return (f'Title: {self._title}\t Status[Google||Arxiv||CrossRef]:{status}')

    def extract_title(self):
        messages = [
            {"role": "system",
             "content": "You are a researcher, who is good at reading academic paper and familiar with all of the citation style."},
            {"role": "assistant",
             "content": "This is a piece of text that could be any form of citations or even just the title of the research paper : " + self.ref_text},
            {"role": "user",
             "content": f'Extract the title of the provided text, and answer with the title only.'},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
             
            messages=messages,
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        return result
