#!/usr/bin/env python3
"""Extract pdf structure in XML format"""
import logging
import os.path
import re
import sys
import warnings
from collections import Counter
from typing import Any, Container, Dict, Iterable, List, Optional, TextIO, Union, cast
from argparse import ArgumentParser

import pdfminer
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines, PDFXRefFallback
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjectNotFound, PDFValueError
from pdfminer.pdftypes import PDFStream, PDFObjRef, resolve1, stream_value
from pdfminer.psparser import PSKeyword, PSLiteral, LIT
from pdfminer.utils import isnumber




def get_subtitles(fpth: str):
    def escape(s: Union[str, bytes]) -> str:
        ESC_PAT = re.compile(r'[\000-\037&<>()"\042\047\134\177-\377]')
        if isinstance(s, bytes):
            us = str(s, "latin-1")
        else:
            us = s
        return ESC_PAT.sub(lambda m: "&#%d;" % ord(m.group(0)), us)
    #outfp: TextIO,
    fp = open(fpth, "rb")
    parser = PDFParser(fp)
    try:
        doc = PDFDocument(parser)
    except:
        return {}
    pages = {
        page.pageid: pageno
        for (pageno, page) in enumerate(PDFPage.create_pages(doc), 1)
    }

    def resolve_dest(dest: object) -> Any:
        if isinstance(dest, (str, bytes)):
            dest = resolve1(doc.get_dest(dest))
        elif isinstance(dest, PSLiteral):
            dest = resolve1(doc.get_dest(dest.name))
        if isinstance(dest, dict):
            dest = dest["D"]
        if isinstance(dest, PDFObjRef):
            dest = dest.resolve()
        return dest

    titles = {}
    try:

        outlines = doc.get_outlines()
        # outfp.write("<outlines>\n")
        for (level, title, dest, a, se) in outlines:
            pageno = None
            if dest:
                dest = resolve_dest(dest)
                pageno = pages[dest[0].objid]
            elif a:
                action = a
                if isinstance(action, dict):
                    subtype = action.get("S")
                    if subtype and repr(subtype) == "/'GoTo'" and action.get("D"):
                        dest = resolve_dest(action["D"])
                        pageno = pages[dest[0].objid]
            s = escape(title)
            if f'level{str(level)}' in titles:
                titles[f'level{str(level)}'].append({'title':s, 'pageno':pageno})
            else:
                titles[f'level{str(level)}'] = [{'title':s, 'pageno':pageno}]
            # print(s)
            # titles[f'level{str(level)}'] = titles[f'level{str(level)}'] if f'level{str(level)}' in titles else []
            # if level <= max_level:
            #     titles.append({'title':s,'level':level,'pageno':pageno})
    except PDFNoOutlines:
        warnings.warn(f'PDF NO OUTLINES: {fpth}')
        pass
    parser.close()
    fp.close()
    # print(key_word_freq)
    return titles
from pdfminer.high_level import extract_text
def get_section_title(title):
    pass
def extract_keyword_from_pdf(fpth):
    titles = get_subtitles(fpth)
    print(titles)
    text = extract_text(fpth)
    start_index = text.lower().find('abstract')
    end_index = text.lower().replace('\n', '').replace(' ', '').find(titles['level1'][0]['title'].replace(' ', ''))

    abstract_text = text[start_index:end_index]
    print(start_index,end_index)
    # print(abstract_text)

#     messages = [
#         {"role": "system",
#          "content": "You are a researcher, who is good at reading academic paper, and familiar with all of the "
#                     "citation style. Please note that the provided citation text may not have the correct line breaks "
#                     "or numbering identifiers."},
#
#         {"role": "user",
#          "content": f'''Extract the paper title only from the given reference text, and answer with the following format.
#                 [1] xxx
#                 [2] xxx
#                 [3] xxx
#             Reference text: {text}
# '''},
#     ]
#
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         # prompt需要用英语替换，少占用token。
#         messages=messages,
#     )
#     result = ''
#     for choice in response.choices:
#         result += choice.message.content
#     result = result.split('\n')
#     return result


if __name__ == "__main__":
    titles = get_subtitles(r'C:\Users\Ocean\Downloads\2208.00173.pdf')
    print(titles)