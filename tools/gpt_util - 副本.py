# https://api.chatanywhere.cn
import openai
from retry import retry

from config.config import openai_key

openai.api_base = "https://api.chatanywhere.com.cn"
openai.api_key = openai_key


def _get_ref_list(text):
    messages = [
        {"role": "system",
         "content": "You are a researcher, who is good at reading academic paper, and familiar with all of the "
                    "citation style. Please note that the provided citation text may not have the correct line breaks "
                    "or numbering identifiers."},

        {"role": "user",
         "content": f'''Extract the paper title only from the given reference text, and answer with the following format.
                [1] xxx
                [2] xxx
                [3] xxx 
            Reference text: {text}
'''},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = result.split('\n')
    return result

@retry(delay=6, )
def get_chatgpt_field_oldversion(title, abstract=None, sys_content=None, usr_prompt=None, extra_prompt=True):
    if not sys_content:
        sys_content = (
            "You are a profound researcher who is good at identifying the topic keyword from paper's title and "
            "abstract. The keyword will be used to retrieve related paper from online scholar search engines.")
    if not usr_prompt:
        usr_prompt = (
            "Identifying the topic of the paper based on the given title and abstract. So that I can use it as "
            "keyword to search highly related papers from Google Scholar.  Avoid using broad or overly general "
            "term like 'deep learning', 'taxonomy', or 'surveys'. Instead, focus on keyword that are unique "
            "and directly pertinent to the paper's subject.Answer with the word only in the"
            "following format: xxx")

    messages = [{"role": "system",
                 "content": sys_content}]
    extra_abs_content = '''
    Given Title: A Survey of Self-Supervised and Few-Shot Object Detection
    Given Abstract: Labeling data is often expensive and time-consuming, especially for tasks such as object detection and instance segmentation, which require dense labeling of the image. While few-shot object detection is about training a model on novel(unseen) object classeswith little data, it still requires prior training onmany labeled examples of base(seen) classes. On the other hand, self-supervisedmethods aimat learning representations fromunlabeled data which transfer well to downstream tasks such as object detection. Combining few-shot and self-supervised object detection is a promising research direction. In this survey, we reviewand characterize themost recent approaches on few-shot and self-supervised object detection. Then, we give our main takeaways and discuss future research directions. Project page: https://gabrielhuang.github.io/fsod-survey/''' if abstract else ''
    if extra_prompt:
        messages += [{"role": "user",
                      "content": f'''{usr_prompt}

                        {extra_abs_content}'''},
                     {"role": "assistant",
                      "content": 'few-shot objection detection'}]

    content = f'''{usr_prompt}
                Given Title: {title}
            '''
    if abstract:
        content += f'Given Abstract: {abstract}'

    messages.append(
        {"role": "user",
         "content": content})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = result.split(',')
    result = [i.strip().replace('_', ' ') for i in result]
    return result
    # print('Response: '+ result)
    #
    #
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=  [{"role": "user",
    #          "content": f' What is the key phrase from the given text: {result}. \n '
    #
    #         'This text may only contain the key phrase or the reason why it is the key phrase. Just extract the key phrase, keep it identical to the provided text. Leave the reason behind, Answer with the word only in the following format: xxx'}],
    # )
    # result = response.choices[0].message.content
    # print('ANS: '+result +'DONE')
    #
    # result = result.split(',')
    # result = [i.strip() for i in result]
    # return result
def eval_writting_skill(text):
    messages = [
        {"role": "system",
         "content": "You are a reviewer of the academic journal, who is good at rating the paper from the following perspective: formal tone and academic style (3 points), clear expression (4 points), free of grammma and spelling errors (3 points)."},
        {"role": "assistant",
         "content": "This is a subparagraph of academic paper. I need to read and rate the following paragraph: " + text},
        {"role": "user",
         "content": f'''Rate the paragraph from 1 to 10, 10 is the best, answer with the rate only:
          Follow the format of the output that follows: x
    '''},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = float(result)
    return result


# pub_number_histogram('segment anything', r'C:\Users\Ocean\Downloads')
@retry(delay=6,)
def get_chatgpt_keyword(title, abstract):
    messages = [
        {"role": "system",
         "content": "You are a profound researcher in the field of artificial intelligence who is good at selecting "
                    "keywords for the paper with given title and abstract. Here are some guidelines for selecting keywords: 1. Represent the content of the title and abstract. 2. Be specific to the field or sub-field. "
                    "3. Keywords should be descriptive. 4. Keywords should reflect a collective understanding of the topic. 5. If the research paper involves a key method or technique, put the term in keywords"},

        {"role": "user",
         "content": f'''Summarize 3-5 keywords only from the given title and abstract, and answer with the following format: xxx, xxx, ..., xxx,
            Given Title: {title}
            Given Abstract: {abstract}
'''},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = result.split(',')
    result = [i.strip() for i in result]
    return result

@retry(delay=6,)
def check_PAMIreview(title, abstract):
    messages = [
        {"role": "system",
         "content": "You are a profound researcher in the field of artificial intelligence who is good at identifying whether a paper is a survey or review paper in the field of pattern analysis and machine intelligence. "
                    "Note that not all paper that its title contain survey or review is a review paper. "
                    "Here are some examples: 'transformers in medical image analysis: a review' is a survey paper. 'Creating a Scholarly Knowledge Graph from Survey Article Tables' is Not a survey. 'Providing Insights for Open-Response Surveys via End-to-End Context-Aware Clustering' is Not a survey. 'sifn: a sentiment-aware interactive fusion network for review-based item recommendation' is Not a review."


         },

        {"role": "user",
         "content": f'''Given title and abstract, identify whether the given paper is a review or survey paper (answer with Y or N)
            Given Title: {title}
            Given Abstract: {abstract}
            Answer with the exact following format:Y||N'''},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    return result

def get_unnum_sectitle(sectitle):
    messages = [
        {"role": "system",
         "content": "You are a profound researcher in the field of artificial intelligence who have read a lot of paper. You can figure out what is the title of section, irrespective of whether they are numbered or unnumbered, and the specific numbering format utilized."},

        {"role": "user",
         "content": f'This is the title of section, extract the title without chapter numbering(If chapter numbering exists). Answer with the following format: xxx. \n Section Title: {sectitle}'},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = result.split(',')
    result = [i.strip() for i in result]
    return result


@retry(delay=6, )
def get_chatgpt_field_oldversion(title, abstract=None, sys_content=None, usr_prompt=None, extra_prompt=True):
    if not sys_content:
        sys_content = (
            "You are a profound researcher who is good at identifying the topic keyword from paper's title and "
            "abstract. The keyword will be used to retrieve related paper from online scholar search engines.")
    if not usr_prompt:
        usr_prompt = (
            "Identifying the topic of the paper based on the given title and abstract. So that I can use it as "
            "keyword to search highly related papers from Google Scholar.  Avoid using broad or overly general "
            "term like 'deep learning', 'taxonomy', or 'surveys'. Instead, focus on keyword that are unique "
            "and directly pertinent to the paper's subject.Answer with the word only in the"
            "following format: xxx")

    messages = [{"role": "system",
                 "content": sys_content}]
    extra_abs_content = '''
    Given Title: A Survey of Self-Supervised and Few-Shot Object Detection
    Given Abstract: Labeling data is often expensive and time-consuming, especially for tasks such as object detection and instance segmentation, which require dense labeling of the image. While few-shot object detection is about training a model on novel(unseen) object classeswith little data, it still requires prior training onmany labeled examples of base(seen) classes. On the other hand, self-supervisedmethods aimat learning representations fromunlabeled data which transfer well to downstream tasks such as object detection. Combining few-shot and self-supervised object detection is a promising research direction. In this survey, we reviewand characterize themost recent approaches on few-shot and self-supervised object detection. Then, we give our main takeaways and discuss future research directions. Project page: https://gabrielhuang.github.io/fsod-survey/''' if abstract else ''
    if extra_prompt:
        messages += [{"role": "user",
                      "content": f'''{usr_prompt}

                        {extra_abs_content}'''},
                     {"role": "assistant",
                      "content": 'few-shot objection detection'}]

    content = f'''{usr_prompt}
                Given Title: {title}
            '''
    if abstract:
        content += f'Given Abstract: {abstract}'

    messages.append(
        {"role": "user",
         "content": content})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = result.split(',')
    result = [i.strip().replace('_', ' ') for i in result]
    return result
    # print('Response: '+ result)
    #
    #
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=  [{"role": "user",
    #          "content": f' What is the key phrase from the given text: {result}. \n '
    #
    #         'This text may only contain the key phrase or the reason why it is the key phrase. Just extract the key phrase, keep it identical to the provided text. Leave the reason behind, Answer with the word only in the following format: xxx'}],
    # )
    # result = response.choices[0].message.content
    # print('ANS: '+result +'DONE')
    #
    # result = result.split(',')
    # result = [i.strip() for i in result]
    # return result
import openai

def get_chatgpt_field(title, abstract=None, sys_content=None, usr_prompt=None, extra_prompt=True):
    if not sys_content:
        sys_content = (
            "You are a profound researcher who is good at identifying the topic keyword from paper's title and "
            "abstract. The keyword will be used to retrieve related paper from online scholar search engines.")
    if not usr_prompt:
        usr_prompt = (
            "Identifying the topic of the paper based on the given title and abstract. So that I can use it as "
            "keyword to search highly related papers from Google Scholar.  Avoid using broad or overly general "
            "term like 'deep learning', 'taxonomy', or 'surveys'. Instead, focus on keyword that are unique "
            "and directly pertinent to the paper's subject.Answer with the word only in the"
            "following format: xxx")

    messages = [{"role": "system",
                 "content": sys_content}]
    extra_abs_content = '''
    Given Title: A Survey of Self-Supervised and Few-Shot Object Detection
    Given Abstract: Labeling data is often expensive and time-consuming, especially for tasks such as object detection and instance segmentation, which require dense labeling of the image. While few-shot object detection is about training a model on novel(unseen) object classeswith little data, it still requires prior training onmany labeled examples of base(seen) classes. On the other hand, self-supervisedmethods aimat learning representations fromunlabeled data which transfer well to downstream tasks such as object detection. Combining few-shot and self-supervised object detection is a promising research direction. In this survey, we reviewand characterize themost recent approaches on few-shot and self-supervised object detection. Then, we give our main takeaways and discuss future research directions. Project page: https://gabrielhuang.github.io/fsod-survey/''' if abstract else ''
    if extra_prompt:
        messages += [{"role": "user",
                      "content": f'''{usr_prompt}

                        {extra_abs_content}'''},
                     {"role": "assistant",
                      "content": 'few-shot objection detection'}]

    content = f'''{usr_prompt}
                Given Title: {title}
            '''
    if abstract:
        content += f'Given Abstract: {abstract}'

    messages.append(
        {"role": "user",
         "content": content})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = result.split(',')
    result = [i.strip().replace('_', ' ') for i in result]
    return result

@retry(delay=6,)
def __get_chatgpt_field(title, abstract, extra_prompt=True):
    sys_content = ("You are a profound researcher who is good at identifying the topic keyword from paper's title and "
                   "abstract. The keyword will be used to retrieve related paper from online scholar search engines.")
    usr_prompt = ("Identifying the topic of the paper based on the given title and abstract. I'm going to write a "
                  "review of the same topic and I will directly use it as keyword to retrieve enough related "
                  "reference papers in the same topic from scholar search engine.  Avoid using broad or overly "
                  "general term like 'deep learning', 'taxonomy', or 'surveys'. Instead, focus on keyword that are "
                  "unique and directly pertinent to the paper's subject. Answer with the word only in the following "
                  "format: xxx")
    if extra_prompt:
        messages = [
            {"role": "system",
             "content": sys_content},

            {"role": "user",
             "content": f'''{usr_prompt} Given Title: A Survey of Self-Supervised and Few-Shot Object Detection Given 
             Abstract: Labeling data is often expensive and time-consuming, especially for tasks such as object 
             detection and instance segmentation, which require dense labeling of the image. While few-shot object 
             detection is about training a model on novel(unseen) object classeswith little data, it still requires 
             prior training onmany labeled examples of base(seen) classes. On the other hand, self-supervisedmethods 
             aimat learning representations fromunlabeled data which transfer well to downstream tasks such as object 
             detection. Combining few-shot and self-supervised object detection is a promising research direction. In 
             this survey, we reviewand characterize themost recent approaches on few-shot and self-supervised object 
             detection. Then, we give our main takeaways and discuss future research directions. Project page: 
             https://gabrielhuang.github.io/fsod-survey/ '''},
            {"role": "assistant",
             "content": 'few-shot objection detection'},
            {"role": "user",
             "content": f'''{usr_prompt}
                                    Given Title: {title}
                                    Given Abstract: {abstract}
                                '''},
        ]
    else:

        messages = [
            {"role": "system",
             "content": sys_content},

            {"role": "user",
             "content": f'''{usr_prompt}
                Given Title: {title}
                Given Abstract: {abstract}
            '''},
        ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = result.split(',')
    result = [i.strip() for i in result]
    return result
#
@retry(delay=6,)
def get_chatgpt_field_from_title(title, extra_prompt=True):
    sys_content = ("You are a profound researcher who is good at identifying the topic keyword from paper's title.  "
                   "The keyword will be used to retrieve related paper from online scholar search engines.")
    usr_prompt = ("Identifying the topic of the paper based on the given title. So that I can use it as "
                  "keyword to search highly related papers from Google Scholar.  Avoid using broad or overly general "
                  "term like 'deep learning', 'taxonomy', or 'surveys'. Instead, focus on keyword that are most "
                  "relevant to the paper's subject. Answer with the word only in the"
                  "following format: xxx")
    if extra_prompt:
        messages = [
            {"role": "system",
             "content": sys_content},

            {"role": "user",
             "content": f'''{usr_prompt}
                        Given Title: A Survey of Self-Supervised and Few-Shot Object Detection
                        '''},
            {"role": "assistant",
             "content": 'objection detection'},
            {"role": "user",
             "content": f'''{usr_prompt}
                                    Given Title: {title}
                                '''},
        ]
    else:

        messages = [
            {"role": "system",
             "content": sys_content},

            {"role": "user",
             "content": f'''{usr_prompt}
                Given Title: {title}
            '''},
        ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = result.split(',')
    result = [i.strip() for i in result]
    return result
@retry()
def get_chatgpt_fields(title, abstract, extra_prompt=True,sys_content=None,usr_prompt=None):
    if not sys_content:
        sys_content = "You are a profound researcher who is good at conduct a literature review based on given title and abstract."
    if not usr_prompt:
        usr_prompt = "Given title and abstract, please provide 5 seaching keywords for me so that I can use them as keywords to search highly related papers from Google Scholar or Semantic Scholar. Please avoid responding with overly general keywords such as deep learning, taxonomy, or surveys, etc., and provide the output in descending order of relevance to the keywords. Answer with the words only in the following format: xxx,xxx,xxx"
    if extra_prompt:
        messages = [
            {"role": "system",
             "content": sys_content},

            {"role": "user",
             "content": f'''{usr_prompt}
                Given Title: Diffusion Models in Vision: A Survey
                Given Abstract: Denoising diffusion models represent a recent emerging topic in computer vision, demonstrating remarkable results in the area of generative modeling. A diffusion model is a deep generative model that is based on two stages, a forward diffusion stage and a reverse diffusion stage. In the forward diffusion stage, the input data is gradually perturbed over several steps by adding Gaussian noise. In the reverse stage, a model is tasked at recovering the original input data by learning to gradually reverse the diffusion process, step by step. Diffusion models are widely appreciated for the quality and diversity of the generated samples, despite their known computational burdens, i.e., low speeds due to the high number of steps involved during sampling. In this survey, we provide a comprehensive review of articles on denoising diffusion models applied in vision, comprising both theoretical and practical contributions in the field. First, we identify and present three generic diffusion modeling frameworks, which are based on denoising diffusion probabilistic models, noise conditioned score networks, and stochastic differential equations. We further discuss the relations between diffusion models and other deep generative models, including variational auto-encoders, generative adversarial networks, energy-based models, autoregressive models and normalizing flows. Then, we introduce a multi-perspective categorization of diffusion models applied in computer vision. Finally, we illustrate the current limitations of diffusion models and envision some interesting directions for future research.'''},
            {"role": "assistant",
             "content": 'Denoising diffusion models,deep generative modeling,diffusion models,image generation,noise conditioned score networks'},
            {"role": "user",
             "content": f'''{usr_prompt}
                            Given Title: {title}
                            Given Abstract: {abstract}
                        '''},
        ]
    else:

        messages = [
            {"role": "system",
             "content": sys_content},

            {"role": "user",
             "content": f'''{usr_prompt}
                Given Title: {title}
                Given Abstract: {abstract}
            '''},
        ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    result = result.replace('.','').split(',')
    result = [i.strip() for i in result]
    return result
# t = 'WHAT DO SELF-SUPERVISED VISION TRANSFORMERS LEARN?'
# a = """
# We present a comparative study on how and why contrastive learning (CL) and
# masked image modeling (MIM) differ in their representations and in their performance of downstream tasks. In particular, we demonstrate that self-supervised
# Vision Transformers (ViTs) have the following properties: (1) CL trains selfattentions to capture longer-range global patterns than MIM, such as the shape of
# an object, especially in the later layers of the ViT architecture. This CL property
# helps ViTs linearly separate images in their representation spaces. However, it
# also makes the self-attentions collapse into homogeneity for all query tokens and
# heads. Such homogeneity of self-attention reduces the diversity of representations,
# worsening scalability and dense prediction performance. (2) CL utilizes the lowfrequency signals of the representations, but MIM utilizes high-frequencies. Since
# low- and high-frequency information respectively represent shapes and textures,
# CL is more shape-oriented and MIM more texture-oriented. (3) CL plays a crucial
# role in the later layers, while MIM mainly focuses on the early layers. Upon these
# analyses, we find that CL and MIM can complement each other and observe that
# even the simplest harmonization can help leverage the advantages of both methods
# """
# title = 'ReCLIP: A Strong Zero-Shot Baseline for Referring Expression Comprehension'
# abstract = "Training a referring expression comprehension (ReC) model for a new visual domain requires collecting referring expressions, and potentially corresponding bounding boxes, for images in the domain. While large-scale pre-trained models are useful for image classification across domains, it remains unclear if they can be applied in a zero-shot manner to more complex tasks like ReC. We present ReCLIP, a simple but strong zero-shot baseline that repurposes CLIP, a state-of-the-art large-scale model, for ReC. Motivated by the close connection between ReC and CLIP's contrastive pre-training objective, the first component of ReCLIP is a region-scoring method that isolates object proposals via cropping and blurring, and passes them to CLIP. However, through controlled experiments on a synthetic dataset, we find that CLIP is largely incapable of performing spatial reasoning off-the-shelf. Thus, the second component of ReCLIP is a spatial relation resolver that handles several types of spatial relations. We reduce the gap between zero-shot baselines from prior work and supervised models by as much as 29% on RefCOCOg, and on RefGTA (video game imagery), ReCLIP's relative improvement over supervised ReC models trained on real images is 8%."
# print(get_chatgpt_field(title, abs,True))







@retry(delay=6, )
def extract_keywords_from_article_with_gpt(text):
    messages = [
        {"role": "system",
         "content": "You are a profound researcher in the field of pattern recognition and machine intelligence. You are aware of all types of keywords, such as keyword, index terms, etc.Please note: The text is extracted from the PDF, so line breaks may appear anywhere, or even footnotes may appear between consecutive lines of text."},
        {"role": "user",
         "content": f'''I will give you the text in the first page of an academic paper, you should read it carefully. If there is no provided keywords, ask with None. If there does exist author provided keywords, answer with the extracted keywords (only keywords) in the following format: xxx,xxx,...,xxx. You should answer only with the keyword, do not answer with words like 'index terms'
         The text of the first page:Cryoelectron Microscopy as a Functional Instrument for Systems Biology, Structural Analysis &
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
    '''},
       {"role": "assistant",
         "content": f'''cryo-electron microscopy, cryo-electron tomography, system biology, localomics, dynamomics, micromachining, structural analysis, in silico, molecular machines'''},
        {"role": "user",
         "content": f'''I will give you the text in the first page of another academic paper, you should read it carefully. If there is no provided keywords, ask with None. If there does exist author provided keywords, answer with the extracted keywords (only keywords) in the following format: xxx,xxx,...,xxx. You should answer only with the keyword, do not answer with words like 'index terms'
         The text of the first page:{text}
    '''},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    result = [i.strip() for i in result.split(',')]
    return result


@retry(delay=6, )
def check_relevant(title, abstract, topics):
    messages = [
        {"role": "system",
         "content": "You are a profound researcher in the field of artificial intelligence who is good at identifying whether a paper is talking about specific topic given title and abstract. "
                    "In ohter words, to judge whether a paper is relevant to the topics. "
                    "Note that not all paper that contain the topic keyword is a relevant paper, and missing the topic words could still possiablly be a relevant paper"
         },

        {"role": "user",
         "content": f'''Given title and abstract, identify whether the given paper is a review or survey paper (answer with Y or N)
            Given Title: {title}
            Given Abstract: {abstract}
            Topic: {','.join(topics)}
            Answer with the exact following format:Y||N'''},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # prompt需要用英语替换，少占用token。
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    return result