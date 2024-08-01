# https://api.chatanywhere.cn
import os

import openai
from langchain.chains.llm import LLMChain
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from retry import retry

from config.config import openai_key

# openai.api_base = "https://api.chatanywhere.com.cn"
# openai.api_key = openai_key
API_SECRET_KEY = "sk-zk2ea3df5fd097682a92420d3062d3a9b6b65aa2841d938e"
BASE_URL = "https://flag.smarttrot.com/v1/"
os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

def _get_ref_list(text):
    messages = [
        SystemMessage(content="You are a researcher, who is good at reading academic paper, and familiar with all of the "
                    "citation style. Please note that the provided citation text may not have the correct line breaks "
                    "or numbering identifiers."),
        HumanMessage(content=f'''
        Extract the paper title only from the given reference text, and answer with the following format.
                [1] xxx
                [2] xxx
                [3] xxx 
            Reference text: {text}
        '''),
    ]

    chat = ChatOpenAI(model="gpt-3.5-turbo")
    chat.invoke(messages)

    return chat.batch([messages])[0].content


@retry(delay=6,)
def get_chatgpt_keyword(title, abstract):
    messages = [
        SystemMessage(content="You are a profound researcher in the field of artificial intelligence who is good at selecting "
                    "keywords for the paper with given title and abstract. Here are some guidelines for selecting keywords: 1. Represent the content of the title and abstract. 2. Be specific to the field or sub-field. "
                    "3. Keywords should be descriptive. 4. Keywords should reflect a collective understanding of the topic. 5. If the research paper involves a key method or technique, put the term in keywords"),
        HumanMessage(content=f'''Summarize 3-5 keywords only from the given title and abstract, and answer with the following format: xxx, xxx, ..., xxx,
            Given Title: {title}
            Given Abstract: {abstract}
'''),
    ]

    chat = ChatOpenAI(model="gpt-3.5-turbo")
    chat.invoke(messages)

    return chat.batch([messages])[0].content


@retry(delay=6,)
def check_PAMIreview(title, abstract):
    messages = [
            SystemMessage(
                content="You are a profound researcher in the field of artificial intelligence who is good at "
                        "identifying whether a paper is a survey or review paper in the field of pattern analysis and "
                        "machine intelligence."
                    "Note that not all paper that its title contain survey or review is a review paper. "
                    "Here are some examples: 'transformers in medical image analysis: a review' is a survey paper. "
                        "'Creating a Scholarly Knowledge Graph from Survey Article Tables' is Not a survey. "
                        "'Providing Insights for Open-Response Surveys via End-to-End Context-Aware Clustering' is "
                        "Not a survey. 'sifn: a sentiment-aware interactive fusion network for review-based item "
                        "recommendation' is Not a review."
),
            HumanMessage(content=f'''Given title and abstract, identify whether the given paper is a review or survey paper (answer with Y or N)
            Given Title: {title}
            Given Abstract: {abstract}
            Answer with the exact following format:Y||N'''),
        ]
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    chat.invoke(messages)

    return chat.batch([messages])[0].content


def get_unnum_sectitle(sectitle):

    messages = [
        SystemMessage(
            content="You are a profound researcher in the field of artificial intelligence who have read a lot of "
                    "paper. You can figure out what is the title of section, irrespective of whether they are "
                    "numbered or unnumbered, and the specific numbering format utilized."),
        HumanMessage(content=f'This is the title of section, extract the title without chapter numbering(If chapter numbering exists). Answer with the following format: xxx. \n Section Title: {sectitle}'),
    ]
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    chat.invoke(messages)

    return chat.batch([messages])[0].content
@retry()
def get_chatgpt_field(title, abstract=None, sys_content=None, usr_prompt=None, extra_prompt=True):
    if not sys_content:
        sys_content = (
            "You are a profound researcher who is good at identifying the topic keyword from paper's title and "
            "abstract. The keyword will be used to retrieve related paper from online scholar search engines.")
    if not usr_prompt:
        usr_prompt = (
            "Identifying the topic of the paper based on the given title and abstract. So that I can use it as "
            "keyword to search highly related papers from Google Scholar.  Avoid using broad or overly general "
            "terms like 'deep learning', 'computer vision', or 'artificial intelligence'. Instead, focus on keyword that are unique "
            "and directly pertinent to the paper's subject.Answer with the word only in the"
            "following format: xxx")

    messages = [SystemMessage(content=sys_content)]

    extra_abs_content = '''
    Given Title: A Survey of Self-Supervised and Few-Shot Object Detection
    Given Abstract: Labeling data is often expensive and time-consuming, especially for tasks such as object detection and instance segmentation, which require dense labeling of the image. While few-shot object detection is about training a model on novel(unseen) object classeswith little data, it still requires prior training onmany labeled examples of base(seen) classes. On the other hand, self-supervisedmethods aimat learning representations fromunlabeled data which transfer well to downstream tasks such as object detection. Combining few-shot and self-supervised object detection is a promising research direction. In this survey, we reviewand characterize themost recent approaches on few-shot and self-supervised object detection. Then, we give our main takeaways and discuss future research directions. Project page: https://gabrielhuang.github.io/fsod-survey/''' if abstract else ''
    if extra_prompt:
        messages += [HumanMessage(content=f'''{usr_prompt}\n\n{extra_abs_content}'''), AIMessage(content='few-shot objection detection')]

    content = f'''{usr_prompt}
                Given Title: {title}
            '''
    if abstract:
        content += f'Given Abstract: {abstract}'
    messages.append(HumanMessage(content=content))

    chat = ChatOpenAI(model="gpt-3.5-turbo")
    chat.invoke(messages)

    return chat.batch([messages])[0].content
@retry()
def get_chatgpt_fields(title, abstract, extra_prompt=True,sys_content=None,usr_prompt=None):
    if not sys_content:
        sys_content = ("You are a profound researcher who is good at conduct a literature review based on given title and abstract.")
    if not usr_prompt:
        usr_prompt = ("Given title and abstract, please provide 5 seaching keywords for me so that I can use them as "
                      "keywords to search highly related papers from Google Scholar or Semantic Scholar. Please avoid "
                      "responding with overly general keywords such as deep learning, taxonomy, or surveys, etc., "
                      "and provide the output in descending order of relevance to the keywords. Answer with the words "
                      "only in the following format: xxx,xxx,xxx")

    if extra_prompt:
        messages = [SystemMessage(content=sys_content),HumanMessage(content=f'''{usr_prompt}\n Given Title: Diffusion Models in Vision: A Survey \nGiven Abstract: Denoising 
             diffusion models represent a recent emerging topic in computer vision, demonstrating remarkable results 
             in the area of generative modeling. A diffusion model is a deep generative model that is based on two 
             stages, a forward diffusion stage and a reverse diffusion stage. In the forward diffusion stage, 
             the input data is gradually perturbed over several steps by adding Gaussian noise. In the reverse stage, 
             a model is tasked at recovering the original input data by learning to gradually reverse the diffusion 
             process, step by step. Diffusion models are widely appreciated for the quality and diversity of the 
             generated samples, despite their known computational burdens, i.e., low speeds due to the high number of 
             steps involved during sampling. In this survey, we provide a comprehensive review of articles on 
             denoising diffusion models applied in vision, comprising both theoretical and practical contributions in 
             the field. First, we identify and present three generic diffusion modeling frameworks, which are based 
             on denoising diffusion probabilistic models, noise conditioned score networks, and stochastic 
             differential equations. We further discuss the relations between diffusion models and other deep 
             generative models, including variational auto-encoders, generative adversarial networks, energy-based 
             models, autoregressive models and normalizing flows. Then, we introduce a multi-perspective 
             categorization of diffusion models applied in computer vision. Finally, we illustrate the current 
             limitations of diffusion models and envision some interesting directions for future research.'''),
                    AIMessage(content='Denoising diffusion models,deep generative modeling,diffusion models,image generation,noise conditioned score networks'),
                    HumanMessage(content=f'''{usr_prompt}\n
                            Given Title: {title}\n
                            Given Abstract: {abstract}
                        ''')]
    else:
        messages = [SystemMessage(content=sys_content),HumanMessage(content=f'''{usr_prompt}\n
                Given Title: {title}\n
                Given Abstract: {abstract}
            ''')]
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    chat.invoke(messages)

    return chat.batch([messages])[0].content



@retry(delay=6, )
def extract_keywords_from_article_with_gpt(text):
    messages = [
        SystemMessage(
            content="You are a profound researcher in the field of pattern recognition and machine intelligence. You "
                    "are aware of all types of keywords, such as keyword, index terms, etc.Please note: The text is "
                    "extracted from the PDF, so line breaks may appear anywhere, or even footnotes may appear between "
                    "consecutive lines of text."),
        HumanMessage(content= f'''I will give you the text in the first page of an academic paper, you should read it carefully. If there is no provided keywords, ask with None. If there does exist author provided keywords, answer with the extracted keywords (only keywords) in the following format: xxx,xxx,...,xxx. You should answer only with the keyword, do not answer with words like 'index terms'
         The text of the first page: Cryoelectron Microscopy as a Functional Instrument for Systems Biology, Structural Analysis &
        Experimental Manipulations with Living Cells
        (A comprehensive review of the current works).
        Oleg V. Gradov
        INEPCP RAS, Moscow, Russia
        Email: o.v.gradov@gmail.com
        Margaret A. Gradova
        ICP RAS, Moscow, Russia
        Email: m.a.gradova@gmail.com
        Abstract — The aim of this paper is to give an introductory
        review of the cryoelectron microscopy as a complex data source
        for the most of the system biology branches, including the most
        perspective non-local approaches known as "localomics" and
        "dynamomics". A brief summary of various cryoelectron microscopy methods and corresponding system biological approaches is given in the text. The above classification can be
        considered as a useful framework for the primary comprehensions about cryoelectron microscopy aims and instrumental
        tools
        Index Terms — cryo-electron microscopy, cryo-electron tomography, system biology, localomics, dynamomics, micromachining, structural analysis, in silico, molecular machines
        I. TECHNICAL APPLICATIONS OF
        CRYOELECTRON MICROSCOPY
        Since its development in early 1980s [31]
        cryo-electron microscopy has become one of
        the most functional research methods providing
        the study of physiological and biochemical
        changes in living matter at various hierarchical
        levels from single mammalian cell morphology
        [108] to nanostructures
    '''),
        AIMessage(content=f'''cryo-electron microscopy, cryo-electron tomography, system biology, localomics, dynamomics, micromachining, structural analysis, in silico, molecular machines'''),
        HumanMessage(content=f'''I will give you the text in the first page of another academic paper, you should read it carefully. If there is no provided keywords, ask with None. If there does exist author provided keywords, answer with the extracted keywords (only keywords) in the following format: xxx,xxx,...,xxx. You should answer only with the keyword, do not answer with words like 'index terms'
         The text of the first page:{text}''')
    ]
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    chat.invoke(messages)

    return chat.batch([messages])[0].content

