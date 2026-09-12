## 一、系统架构

## 1.1 核心模块

索引：文档加载→文本分块→向量嵌入→向量存储。

检索：查询嵌入→相似度检索→提取上下文。

生成：提示工程→LLM调度→输出后处理。

服务接口：REST API 端点→流式响应→监控指标。

## 1.2 核心框架

<table><tr><td rowspan=1 colspan=1>组件</td><td rowspan=1 colspan=1>技术选择</td><td rowspan=1 colspan=1>特点</td></tr><tr><td rowspan=1 colspan=1>嵌入模型</td><td rowspan=1 colspan=1>SentenceTransformers</td><td rowspan=1 colspan=1>句子级向量表示，可用于高效计算相似性。</td></tr><tr><td rowspan=1 colspan=1>向量数据库</td><td rowspan=1 colspan=1>Chroma</td><td rowspan=1 colspan=1>轻量级向量数据库，简单易用。支持持久化存储与元数据管理。</td></tr><tr><td rowspan=1 colspan=1>向量数据库</td><td rowspan=1 colspan=1>FAISS</td><td rowspan=1 colspan=1>高效的相似度搜索库，支持多种搜索算法。不具备持久化能力，需要手动持久化。</td></tr><tr><td rowspan=1 colspan=1>向量数据库</td><td rowspan=1 colspan=1> Milvus</td><td rowspan=1 colspan=1>支持大规模数据集的存储和检索，适合大规模生产级应用。提供完整的数据库功能，支持数据管理，持久化存储。</td></tr><tr><td rowspan=1 colspan=1>LLM运行时</td><td rowspan=1 colspan=1>vLLM</td><td rowspan=1 colspan=1>高性能推理，连续批处理。</td></tr><tr><td rowspan=1 colspan=1>服务框架</td><td rowspan=1 colspan=1>FastAPI</td><td rowspan=1 colspan=1>异步支持，自动生成 OpenAPl文档。</td></tr></table>

## 1.3 环境准备

conda create -n rag python=3.12   
conda activate rag

## 安装依赖：

pip install langchain_community unstructured charset-normalizer==3.4.3 markdown pi_heif unstructured_inference pdf2image unstructured_pytesseract python-docx langchain_huggingface sentence-transformers langchain_chroma dashscope jieba faiss-cpu ragas bitsandbytes rank_bm25 pymysql sqlacodegen fastapi 2 pip install langchain langchain-openai

## 二、索引过程

## 2.1 文档加载

数据源可能来自多种格式的文件，如文本文档、Markdown，PDF等，首先需要对各种格式的文件进行处理，将其转化为可用的格式。

langchain_community.document_loaders 中提供了多种格式的文档加载器，包括：

TextLoader（文本文档加载）

UnstructuredMarkdownLoader（Markdown加载）

PyPDFLoader（PDF加载）

UnstructuredPDFLoader（PDF加载）

UnstructuredWordDocumentLoader（Word文档加载）

WebBaseLoader（网站HTML加载）

## 等。

## 1）加载文本文档

```python
2 TextLoader会将一个文档文件加载为一个Document对象，该对象有两个属性：
3 metadata: 存储该文档的来源路径等元数据
4 page_content: 存储文档的内容
5
6
7 from langchain_community.document_loaders import TextLoader
8
9 text_documents = TextLoader("knowledge_base/sample.txt", encoding="utf-8").load()
10 print(text_documents)
```

## 2）加载Markdown

```python
1
2 UnstructuredMarkdownLoader可用于加载Markdown文件
3 mode: 加载模式
4 "single" 返回单个Document对象
5 "elements" 按标题等元素切分文档
6 strategy: 加载策略
7 "fast" 快速粗粒度加载
8 "hi_res" 细粒度加载，按标题层级、列表项、表格等结构细分
9
10
11 from langchain_community.document_loaders import UnstructuredMarkdownLoader
12
13 md_documents = UnstructuredMarkdownLoader(
14 "knowledge_base/sample.md",
15 mode="elements",
16 strategy="fast",
17 ).load()
18 print(md_documents)
```

## 3）加载PDF

（1）使用 PyPDFLoader 解析文档

1  
2 PyPDFLoader  
3 支持页级拆分，每一页作为一个Document返回

```python
4 支持提取图像、提取布局
5 extraction_mode: 提取模式
6 "plain" 提取纯文本
7 "layout" 提取布局
8
9
10 from langchain_community.document_loaders import PyPDFLoader
11
12 pdf_documents = PyPDFLoader(
13 "knowledge_base/sample.pdf",
14 extraction_mode="layout",
15 ).load()
16 print(pdf_documents)
```

## （2）使用 UnstructuredPDFLoader 解析文档

如果使用UnstructuredPDFLoader，需要先下载 Poppler 和 Tesseract OCR 。

Poppler 是一个开源的 PDF 文档处理库，用于渲染、解析和操作 PDF 文件。下载后将 .../poppler-24.08.0/Library/bin 添加到环境变量 Path 中即可。

Tesseract OCR 提取图像中的文字，在安装时需要选择 Additional language data(download) 来添加中文包。

![](images/25b66c1e74bfcc3e90559ba759b516cfd11153d3a075639b22d4ce275b70869a.jpg)

安装后，将安装时设置的安装目录添加到环境变量 Path 中。

使用 UnstructuredPDFLoader 解析文档：

1  
2 UnstructuredPDFLoader  
3 支持结构化提取，支持OCR  
4 仅当 PDF 文档中不存在文本时，才会应用 OCR  
5 mode: 加载模式

```python
6 "single" 返回单个Document对象
7 "elements" 按标题等元素切分文档
8 strategy: 加载策略
9 "fast" 提取并处理文本
10 "ocr_only" 先进行 OCR 处理，再处理原始文本
11 "hi_res" 识别文档布局并处理，自动下载YOLOX模型（识别页面布局）
12 infer_table_structure: 是否推断表格结构
13 仅 hi_res 下起效
14 如果为 True，提取出的表格元素会包含一个 text_as_html 元数据，将文本内容转换为 html
格式
15 languages: OCR使用的语言
16 需传入语言列表
17 语言列表参考 https://github.com/tesseract-ocr/langdata
18 更多参数详见 https://github.com/Unstructured-
IO/unstructured/blob/main/unstructured/partition/pdf.py
19
20
21 from langchain_community.document_loaders import UnstructuredPDFLoader
22
23 pdf_documents = UnstructuredPDFLoader(
24 "knowledge_base/sample.pdf",
25 mode="elements",
26 strategy="hi_res",
27 infer_table_structure=True,
28 languages=["eng", "chi_sim"],
29 ).load()
30 print(pdf_documents)
```

策略配置为hi_res时，会自动下载 yolox 模型用于页面布局的识别，会从 huggingface 下载到本地路径： C:\Users\<你的用户名>\.cache\huggingface\hub\

不想下载可以直接解压资料中的模型到该目录下。

Document 包含 metadata 和 page_content ， metadata 字段的详细说明：

source:表示文档的原始来源路径

detection_class_prob:这是元素检测的置信度概率，表示模型对该元素分类的置信程度，值越接近1表示越确定

coordinates:包含元素在页面上的坐标信息

points: 元素的四个顶点坐标，按顺时针或逆时针排列，格式为((x1,y1), (x2,y2), (x3,y3), (x4,y4))

system: 坐标系统，这里是'PixelSpace'，表示像素空间坐标

layout_width 和 layout_height: 页面的布局尺寸，用于定位元素

last_modified: 文件的最后修改时间

filetype:文件的MIME类型

languages: 检测到的语言列表，使用ISO 639-2代码，如'zho'代表中文，'kor'代表韩文

page_number: 元素所在的页码

file_directory: 文件所在的目录

filename: 文件名

parent_id: 父元素的唯一标识符，用于追踪元素的层级关系

category: 元素的类别，如'FigureCaption'(图表标题)、'Table'(表格)、'Text'(文本)等

element_id: 该元素的唯一标识符

text_as_html:仅在表格等元素中出现，将元素内容转换为HTML格式的文本，特别适用于表格结构，便于后续处理和展示

这些元数据字段为每个提取的元素提供了丰富的上下文信息，包括位置信息、分类信息、来源信息等，对于后续的文档分析、信息检索和RAG应用非常有价值。特别是坐标信息，可以用于精确地定位元素在页面中的位置，而分类信息可以帮助区分不同类型的文档元素。

## 4）加载Word文档

```python
1
2 UnstructuredWordDocumentLoader
3 适用于 .docx 和 .doc 文件
4 mode: 加载模式
5 "single" 返回单个Document对象
6 "elements" 按标题等元素切分文档
7 strategy: 加载策略
8 "fast" 快速粗粒度加载
9 "hi_res" 细粒度加载，按结构细分
10
11
12 from langchain_community.document_loaders import UnstructuredWordDocumentLoader
13
14 word_documents = UnstructuredWordDocumentLoader(
15 "knowledge_base/sample.docx",
16 mode="elements",
17 strategy="fast",
18 ).load()
19 print(word_documents)
```

## 5）通过网址链接加载HTML

1   
2 WebBaseLoader   
3 适用于网页   
4 web_paths: 网址序列   
5 bs_kwargs: 传给 BeautifulSoup 的解析参数   
6 parse_only 只提取指定标签的元素   
7   
8 import bs4   
9 from langchain_community.document_loaders import WebBaseLoader   
10   
11 web_documents = WebBaseLoader(   
12 web_paths=(   
13 " https://news.sina.com.cn/c/xl/2025-09-07/doc-infprmwn0510979.shtml",   
14 ),   
15 bs_kwargs={"parse_only": bs4.SoupStrainer(id="article")}, # 只提取正文区域   
16 ).load()   
17 print(web_documents)

## 6）封装函数，加载文件夹中多种文件类型

```python
from langchain_community.document_loaders import (
2 TextLoader,
3 UnstructuredMarkdownLoader,
4 UnstructuredPDFLoader,
5 UnstructuredWordDocumentLoader
6 )
8 def load_documents():
9
10 加载多种类型的文档，包括text、markdown、PDF和Word文档
11
12 Returns:
13 list: 包含所有加载文档的列表
14
```

```python
15 # 加载文本文件
16 text_documents = TextLoader(
17 "knowledge_base/sample.txt",
18 encoding="utf8"
19 ).load()
20
21 # 加载Markdown文件
22 md_documents = UnstructuredMarkdownLoader(
23 "knowledge_base/sample.md"
24 ).load()
25
26 # 加载PDF文件
27 pdf_documents = UnstructuredPDFLoader(
28 "knowledge_base/sample.pdf",
29 mode="elements", # 元素模式
30 strategy="hi_res", # 高分辨率策略
31 # strategy="fast",
32 languages=["eng", "chi_sim"], # 支持的语言：英文和简体中文
33 ).load()
34
35 # 加载Word文档
36 word_documents = UnstructuredWordDocumentLoader(
37 "knowledge_base/sample.docx"
38 ).load()
39
40 # 返回所有文档的列表
41 return text_documents + md_documents + pdf_documents + word_documents
42
43 documents = load_documents()
```

## 7）文本清洗

对加载的文本进行清洗。去除 HTML 标签，去除多余的空格。

后续使用的Chroma向量数据库，元数据只支持str、int、float、bool类型，因此需要将 metadata中所有非 Chroma 支持类型的值转为 JSON 格式的字符串。

```python
1 import re
2 import json
3
4 def clean_content(documents: list):
5 """文本清洗"""
6 cleaned_docs = []
7
8 for doc in documents:
9 # 1、page_content处理：去除多余换行和空格
10 text = doc.page_content
11
12 # 将连续的换行符替换为两个换行符，正则表达式模式：r"\n{2,}"
13 # r"" 表示原始字符串（raw string），避免转义字符的特殊处理
14 # \n 表示换行符
15 # {2,} 是量词，表示前面的字符（换行符）出现 2 次或更多次
16 text = re.sub(r"\n{2,}", "\n\n", text)
17 text = text.strip()
18
19 # 2、metadata处理：将所有非 Chroma 支持类型的值转为 JSON 格式字符串
20 allowed_types = (str, int, float, bool)
21 for key, value in doc.metadata.items():
```

```python
22 if not isinstance(value, allowed_types):
23 try:
24 doc.metadata[key] = json.dumps(value, ensure_ascii=False)
25 except (TypeError, ValueError):
26 # 如果 json.dumps 失败（如包含不可序列化对象），转为 str
27 doc.metadata[key] = str(value)
28
29 # 3、更新文档内容
30 doc.page_content = text
31 cleaned_docs.append(doc)
32
33 return cleaned_docs
```

## 2.2 文本分块

加载后的文档往往过长，不适合直接作为 LLM 的上下文。因此在加载文档之后需要对文档内容进行分块，将一个长文档分割成多个块。

如何对文档进行分块，往往决定了检索系统的下限。但是分块方式的选择与具体业务具有很强的关联性，针对不同业务不同文档往往需要定制不同的分块方式。

langchain-text-splitters 中提供了多种文档分块方式，包括：

CharacterTextSplitter（根据分隔符按字符拆分文档）

RecursiveCharacterTextSplitter（递归使用多个分隔符按字符拆分文档）

TokenTextSplitter（使用模型分词器将文本拆分为 token）

SentenceTransformersTokenTextSplitter（使用句子模型分词器将文本拆分为 token）

## 等。

文本分块时常用的两个参数 chunk_size 和 chunk_overlap，分别定义了块的大小和两个块之间重叠部分的大小。

![](images/c24d8627d911eeb218c0e051eb14081ce5f5eaeecb26c2735a8969875ce2099e.jpg)

```python
1 from langchain_text_splitters import RecursiveCharacterTextSplitter
3 # 文本分块
4 text_splitter = RecursiveCharacterTextSplitter(
5 separators=["\n\n", "。"], # 分隔符列表
6 chunk_size=400, # 每个块的最大长度
chunk_overlap=40, # 每个块重叠的长度
8 )
9 texts = text_splitter.split_documents(documents)
```

## 2.3向量嵌入与存储

使用 bge-base-zh-v1.5 模型将文本转换为向量。

BGE 是一个向量模型，能够将任何文本映射为低维稠密向量，可用于检索、分类、聚类或语义搜索等任务。

```python
1 import torch
2 from langchain_huggingface import HuggingFaceEmbeddings
3
4 # 加载嵌入模型
5 embedding_model = HuggingFaceEmbeddings(
6 model_name="./bge-base-zh-v1.5",
7 model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
8 encode_kwargs={
9 "normalize_embeddings": True
10 }, # 输出归一化向量，更适合余弦相似度计算
11 )
12
13 # 从 Document 中取出文本
14 page_content_list = [text.page_content for text in texts]
15 # 进行嵌入
16 embeddings = embedding_model.embed_documents(page_content_list)
17 # 打印嵌入结果
18 for i, (page_content, vector) in enumerate(zip(page_content_list, embeddings)):
19 print("Text:\n", page_content)
20 print("Embedding:\n", vector[:5])
21 print()
22 if i == 5:
23 break
```

将文档嵌入并存入 Chroma 中。

Chroma 中存储和查询的基本单位是 Collection，类似传统数据库中的表。每个 Collection 包含一组元素，每个元素包含以下内容：

唯一标示 ID

嵌入向量

嵌入向量对应文档

元数据键值对

```python
from langchain_chroma import Chroma
3 # 嵌入并存储到向量数据库
4 vectorstore = Chroma.from_documents(
5 texts, # 文档列表
6 embedding_model, # 嵌入模型
persist_directory="vectorstore", # 存储路径
8
```

通过 get 方法可以查看 Chroma 中的数据。

print(vectorstore.get().keys()) # 查看所有属性   
2 print(vectorstore.get(include=["embeddings"])["embeddings"][:5, :5]) # 查看嵌入向量

## 三、检索过程

对用户的问题进行嵌入处理，转换成向量之后与向量数据库中的向量进行余弦相似度计算，返回最相似的若干个文档。

首先加载嵌入模型和向量数据库。

```python
import torch
2 from langchain_huggingface import HuggingFaceEmbeddings
3 from langchain_chroma import Chroma
4
5 # 加载嵌入模型
6 embedding_model = HuggingFaceEmbeddings(
7 model_name="./bge-base-zh-v1.5",
8 model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
9 encode_kwargs={"normalize_embeddings": True},
10 )
11
12 # 初始化 Chroma 客户端
13 vectorstore = Chroma(
14 persist_directory="vectorstore",
15 embedding_function=embedding_model,
16 )
```

检索向量数据库中与用户查询相似的文档。

```python
# 相似度检索
2 query = "不动产或者动产被占有人占有怎么办"
3 sim_docs = vectorstore.similarity_search(query, k=3) # 返回 3 条结果
4 for doc in sim_docs:
5 print(doc)
```

也可以使用最大边际相关性检索（maximal marginal relevance，mmr），该检索方法的目的是在保证搜索结果相关性的同时，尽量减少结果之间的冗余和重复，提高多样性。它会首先返回一个最相关的文档，再迭代选择下一个文档，这个文档既要与查询相关，又不能和已经选出的文档太相似。

```python
1 # 最大边际相关性检索
2 query = "不动产或者动产被占有人占有怎么办"
3 sim_docs = vectorstore.max_marginal_relevance_search(query, k=3)
4 for doc in sim_docs:
5 print(doc)
```

也可以先获取向量存储检索器，再通过检索器进行检索。

```python
# 通过检索器检索
2 query = "不动产或者动产被占有人占有怎么办"
3 # 获取向量存储检索器
4 retriever = vectorstore.as_retriever(
5 search_type="similarity", # 检索方式，similarity 或 mmr
6 search_kwargs={"k": 3},
7 )
8 sim_docs = retriever.invoke(query)
9 for doc in sim_docs:
10 print(doc)
```

## 四、生成过程

## 4.1搭建检索生成链路

使用用户查询检索出相关文档之后，将文档和用户查询处理为 prompt 输入给 LLM。

![](images/8af8e25e7bf4a64041ab42f6d970716991d5a740531a9de0f28f56d76bdb5a29.jpg)

可以使用 LCEL 将这些步骤组合成一个链条。LangChain Expression Language（LCEL，LangChain 表达式语言）是用于构建和组合链的一种简洁的声明式语法，允许像管道操作符一样将多个组件组合起来构成复杂的应用流程。并且支持诸如流式处理、并行处理和日志记录等开箱即用的功能。

每个 LCEL 对象都实现了 Runnable 接口，该接口定义了一组公共的调用方法（invoke、batch、stream、ainvoke等）。这使得 LCEL 对象链也自动支持这些调用。也就是说，每个 LCEL 对象链本身也是一个LCEL 对象。同时 Runnable 通过定义 or 方法重载了 | 运算符，因此可以通过 | 将多个 Runnable 对象组合成一个 RunnableSequence。

首先创建 .env 文件，在该文件中写入 LLM 的 API-Key。

在开发环境中可以使用.env，但在实际生产环境中建议使用实际的环境变量。

1 TONGYI_API_KEY=我的 API-Key

首先加载嵌入模型与向量数据库，并创建检索器。

```python
import os
2 import torch
3 from dotenv import load_dotenv
4 from langchain_chroma import Chroma
5 from langchain_core.prompts import PromptTemplate
6 from langchain_community.llms.tongyi import Tongyi
7 from langchain_huggingface import HuggingFaceEmbeddings
```

```python
8 from langchain_core.output_parsers import StrOutputParser
9 from langchain_core.runnables.passthrough import RunnablePassthrough
10
11 # 加载嵌入模型
12 embedding_model = HuggingFaceEmbeddings(
13 model_name="./bge-base-zh-v1.5",
14 model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
15 encode_kwargs={"normalize_embeddings": True},
16 )
17
18 # 初始化 Chroma 客户端
19 vectorstore = Chroma(
20 persist_directory="vectorstore",
21 embedding_function=embedding_model,
22 )
23
24 # 创建检索器
25 retriever = vectorstore.as_retriever(
26 search_type="similarity", # 检索方式，similarity 或 mmr
27 search_kwargs={"k": 3},
28 )
```

构建检索生成流程，使用检索器进行检索，LLM 再基于检索内容生成答案。

```python
# 检索与生成链条
2 load_dotenv()
3 TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
4
5 # 将检索到的文档中的 page_content 取出组合到一起
6 def format_docs(docs):
7 return "\n\n".join(doc.page_content for doc in docs)
8
9 # Prompt 模板
10 prompt = PromptTemplate(
11 input_variables=["context", "query"],
12 template="""
13 你是一个专业的中文问答助手，擅长基于提供的资料回答用户问题。
14 请仅根据以下背景资料回答问题，如无法找到答案，请直接回答“我不知道”。
15
16 背景资料：{context}
17
18 问题：{query}
19
20 回答：
21 )
22
23 # 大模型
24 llm = Tongyi(model="qwen-turbo", api_key=TONGYI_API_KEY)
25
26 # RAG 链条
27 rag_chain = (
28 {"context": retriever | format_docs, "query": RunnablePassthrough()}
29 | prompt
30 | (lambda x: print(x.text, end="") or x)
31 | llm
32 | StrOutputParser() # 输出解析器，将输出解析为字符串
33 )
34
```

35 query = "不动产或者动产被占有人占有怎么办"   
36 response = rag_chain.invoke(query)   
37 print(response)

```python
from langchain_core.runnables import RunnableLambda
2
3 func1 = RunnableLambda(lambda x: x + 1)
4 func2 = RunnableLambda(lambda x: x + 100)
5 rag = (func1 | func2)
6 print(rag.invoke(100))
```

## 补充：使用Agent完成RAG过程

## 使用工具 ..

```python
1 from langchain.tools import tool
2 from langchain.chat_models import init_chat_model
3
4 # 1. 把 retriever 包装成 @tool
5 @tool
6 def retrieve(query: str) -> str:
7
8 从知识库中检索与问题相关的法律条文。
9 当需要回答关于不动产、动产、占有、债权等法律问题时调用。
10 """
11 docs: list[Document] = retriever.invoke(query)
12 if not docs:
13 return "知识库中未找到相关内容。"
14 return "\n\n".join(
15 f"[片段{i+1}]\n{doc.page_content}"
16 for i, doc in enumerate(docs)
17 )
18
19 # ── 2. 定义 LLM
20 llm = init_chat_model(model="gpt-4o")
21
22 # ── 3. 系统提示（替代原来的 PromptTemplate）
23 SYSTEM_PROMPT = """你是一个专业的中文法律问答助手。
24
25 回答问题前，请先调用 retrieve 工具检索相关法律条文。
26 仅根据检索到的资料作答。若工具返回"未找到相关内容"，则直接回答"我不知道"。
27 回答要简洁、准确，引用具体条文编号。"""
28
29 # ── 4. 创建 Agent
30 agent = create_agent(
31 model=llm,
32 tools=[retrieve],
33 system_prompt=SYSTEM_PROMPT,
34 )
35
```

```python
36 # 5. 调用
37 query = "不动产或者动产被占有人占有怎么办"
38
39 result = agent.invoke(
40 {"messages":[{"role": "user", "content": query}]}
41
42
43 # 取最后一条 AI 消息作为答案
44 final_answer = result["messages"][-1].content
45 print(final_answer)
```

## 自定义系统提示词中间件实现： dynamic_prompt

```python
from langchain.agents import create_agent
2 from langchain.tools import tool
3 from langchain.agents.middleware import dynamic_prompt, ModelRequest
4 from dotenv import load_dotenv
5 from langchain_chroma import Chroma
6 from langchain_core.prompts import PromptTemplate
7 from langchain_huggingface import HuggingFaceEmbeddings
8 from langchain.chat_models import init_chat_model
9 # 检索与生成链条
10 load_dotenv()
11
12 # 1、加载嵌入模型
13 embedding_model = HuggingFaceEmbeddings(
14 model_name="./bge-base-zh-v1.5",
15 model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
16 encode_kwargs={"normalize_embeddings": True},
17 )
18
19 # 2、初始化 Chroma 客户端
20 vectorstore = Chroma(
21 persist_directory="vectorstore",
22 embedding_function=embedding_model,
23 )
24
25 # 3、创建检索器
26 retriever = vectorstore.as_retriever(
27 search_type="similarity", # 检索方式，similarity 或 mmr
28 search_kwargs={"k": 3},
29 )
30
31 # 4、构建动态提示词
32 @dynamic_prompt
33 def system_prompt(request: ModelRequest) -> str:
34 """根据用户的问题生成动态提示词，包含相关的背景资料。
35 prompt = PromptTemplate(
36 input_variables=["context"],
37 template="""
38 你是一个专业的中文问答助手，擅长基于提供的资料回答用户问题。
39 请仅根据以下背景资料回答问题，如无法找到答案，请直接回答“我不知道”。
40
41 背景资料：{context}
42 """)
43 docs = retriever.invoke(query)
```

```javascript
44 context = "\n\n".join(doc.page_content for doc in docs)
45 base_prompt = prompt.invoke({"context":context})
46 return base_prompt.text
47
48 # 5、创建智能体
49 agent = create_agent(
50 model=llm,
51 middleware=[system_prompt]
52 )
53
54 query = "不动产或者动产被占有人占有怎么办"
55 result = agent.invoke({"messages":[{"role":"user","content":query}]})
```

## 4.2 添加历史对话记录

在上述流程中，LLM 在交流过程中并不能记住历史的交流信息，这就导致模型不能在每次回复时关联历史交流信息，无法生成上下文连贯的回复。

可以在每轮对话结束之后将本轮对话消息记录到历史记录中，并在下次对话时将历史记录作为 prompt的一部分输入给 LLM。

同时需要注意因为模型存在上下文长度限制，因此不能一股脑将所有历史记录都塞进 prompt 中，这样可能超出 LLM的上下文长度限制，导致模型无法正常生成。

```python
# 带历史对话记录
2 load_dotenv()
3 TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
4
5 # 将检索到的文档中的 page_content 取出组合到一起
6 def format_docs(docs):
7 return "\n\n".join(doc.page_content for doc in docs)
8
9 # Prompt 模板
10 prompt = PromptTemplate(
11 input_variables=["context", "history", "query"],
12 template="""
13 你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
14 请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。
15
16 背景资料：{context}
17
18 历史消息：[{history}]
19
20 问题：{query}
21
22 回答：
23 )
24
25 # 大模型
26 llm = Tongyi(model="qwen-turbo", api_key=TONGYI_API_KEY)
27
28 # 历史消息
29 history = []
30
31 # 格式化历史消息
32 def format_history(history):
33 # 只保留最近 3 轮对话记录
34 max_epoch = 3
```

```python
35 if len(history) > 2 * max_epoch:
36 history = history[-2 * max_epoch :]
37 return "\n".join([f"{i['role']}：{i['content']}" for i in history])
38
39 # RAG 链条
40 rag_chain = (
41 {
42 "context": lambda x: format_docs(retriever.invoke(x["query"], k=3)),
43 "history": lambda x: format_history(x["history"]),
44 "query": lambda x: x["query"],
45
46 | prompt
47 | (lambda x: print(x.text, end="") or x)
48 | llm
49 | StrOutputParser() # 输出解析器，将输出解析为字符串
50 )
51
52 query_list = ["不动产或者动产被人占有怎么办", "那要是被损毁了呢"]
53 for query in query_list:
54 print(f"===== 查询: {query} =====")
55 response = rag_chain.invoke({"query": query, "history": history})
56 print(response, end="\n\n")
57 history.extend(
58 [
59 {"role": "用户", "content": query},
60 {"role": "助手", "content": response},
61 ]
62 )
```

## 4.3 重述用户消息

虽然模型已经能够利用上下文消息进行连贯对话，但是用户在根据上文继续提问时可能不会完整的表述所有的内容，而是使用代词指代或直接省略某些实体。如果使用这样的用户表述直接进行检索，可能无法准确检索到真正相关的文档，影响最终生成质量。

因此，在用户发送一个消息之后，我们可以根据历史对话记录重述用户消息，对用户消息进行指代消解或者补全缺失的实体，以保证后续检索的准确性。

在使用查询进行检索之前，先用LLM根据历史记录完善查询信息，再使用完善后的查询信息进行检索。  
如果不存在历史记录则直接使用查询信息进行检索。

```python
1 # 重述用户消息
2 load_dotenv()
3 TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
4
5 # 将检索到的文档中的 page_content 取出组合到一起
6 def format_docs(docs):
7 return "\n\n".join(doc.page_content for doc in docs)
8
9 # Prompt 模板
10 prompt = PromptTemplate(
11 input_variables=["context", "history", "query"],
12 template="""
13 你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
14 请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。
15
16 背景资料：{context}
17
```

```python
18 历史消息：[{history}]
19
20 问题：{query}
21
22 回答：
23 )
24
25 # 大模型
26 llm = Tongyi(model="qwen-turbo", api_key=TONGYI_API_KEY)
27
28 # 历史消息
29 history = []
30
31 # 格式化历史消息
32 def format_history(history):
33 # 只保留最近 3 轮对话记录
34 max_epoch = 3
35 if len(history) > 2 * max_epoch:
36 history = history[-2 * max_epoch :]
37 return "\n".join([f"{i['role']}：{i['content']}" for i in history])
38
39 # rephrase Prompt 模板
40 rephrase_prompt = PromptTemplate(
41 input_variables=["history", "query"],
42 template="""
43 根据历史消息简要完善用户的问题，使其更加具体。只输出完善后的问题。
44
45 历史消息：[{history}]
46
47 问题：{query}
48 """,
49 )
50
51 # 重述链条：根据历史和当前 query 生成更具体问题
52 rephrase_chain = (
53 {
54 "history": lambda x: format_history(x["history"]),
55 "query": lambda x: x["query"],
56 }
57 rephrase_prompt
58 | llm
59 | StrOutputParser()
60 | (lambda x: print(f"===== 重述后的查询: {x}=====") or x)
61 )
62
63 # Prompt 模板
64 prompt = PromptTemplate(
65 input_variables=["context", "history", "query"],
66 template=""
67 你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
68 请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。
69
70 背景资料：{context}
71
72 历史消息：[{history}]
73
74 问题：{query}
75
```

76 回答：   
77 )   
78   
79 # RAG 链条   
80 rag_chain = (   
81 {   
82 "context": lambda x: format_docs(   
83 retriever.invoke(   
84 rephrase_chain.invoke({"history": x.get("history"), "query":   
x.get("query")}),   
85 k=3,   
86 )   
87 ),   
88 "history": lambda x: format_history(x.get("history")),   
89 "query": lambda x: x.get("query"),   
90 }   
91 | prompt   
92 (lambda x: print(x.text, end="") or x)   
93 | llm   
94 | StrOutputParser() # 输出解析器，将输出解析为字符串   
95 )   
96   
97 query_list = ["不动产或者动产被人占有怎么办", "那要是被损毁了呢"]   
98 for query in query_list:   
99 print(f"===== 查询: {query} =====")   
100 response = rag_chain.invoke({"query": query, "history": history})   
101 print(response, end="\n\n")   
102 history.extend(   
103 [   
104 {"role": "用户", "content": query},   
105 {"role": "助手", "content": response},   
106 ]   
107 )

## 五、RAG 完整应用开发

基于以上内容可以将其整合成一个完整的应用，具体步骤如下：

## 5.1 索引模块

indexing.py ：

```python
1 from langchain_community.document_loaders import (
2 TextLoader,
3 UnstructuredMarkdownLoader,
4 UnstructuredPDFLoader,
5 UnstructuredWordDocumentLoader
6
7 from langchain_text_splitters import RecursiveCharacterTextSplitter
8 from langchain_huggingface import HuggingFaceEmbeddings
9 from langchain_chroma import Chroma
10 import torch
11 import re
12 import json
13
14 def load_documents():
15
```

```python
16 加载多种类型的文档，包括text、markdown、PDF和Word文档
17
18 Returns:
19 list: 包含所有加载文档的列表
20
21 # 加载文本文件
22 text_documents = TextLoader(
23 "knowledge_base/sample.txt",
24 encoding="utf8"
25 ).load()
26
27 # 加载Markdown文件
28 md_documents = UnstructuredMarkdownLoader(
29 "knowledge_base/sample.md"
30 ).load()
31
32 # 加载PDF文件
33 pdf_documents = UnstructuredPDFLoader(
34 "knowledge_base/sample.pdf",
35 mode="elements", # 元素模式
36 strategy="hi_res", # 高分辨率策略
37 # strategy="fast",
38 languages=["eng", "chi_sim"], # 支持的语言：英文和简体中文
39 ).load()
40
41 # 加载Word文档
42 word_documents = UnstructuredWordDocumentLoader(
43 "knowledge_base/sample.docx"
44 ).load()
45
46 # 返回所有文档的列表
47 return text_documents + md_documents + pdf_documents + word_documents
48
49 def clean_content(documents: list):
50
51 """文本清洗"""
52 cleaned_docs = []
53
54 for doc in documents:
55
56 # 1、page_content处理：去除多余换行和空格
57 text = doc.page_content
58
59 # 将连续的换行符替换为两个换行符，正则表达式模式：r"\n{2,}"
60 # r"" 表示原始字符串（raw string），避免转义字符的特殊处理
61 # \n 表示换行符
62 # {2,} 是量词，表示前面的字符（换行符）出现 2 次或更多次
63 text = re.sub(r"\n{2,}", "\n\n", text)
64 text = text.strip()
65
66 # 2、metadata处理：将所有非 Chroma 支持类型的值转为 JSON 格式字符串
67 allowed_types = (str, int, float, bool)
68 for key, value in doc.metadata.items():
69 if not isinstance(value, allowed_types):
70 try:
71 doc.metadata[key] = json.dumps(value, ensure_ascii=False)
72 except (TypeError, ValueError):
73 # 如果 json.dumps 失败（如包含不可序列化对象），转为 str
```

```python
74 doc.metadata[key] = str(value)
75
76 # 3、更新文档内容
77 doc.page_content = text
78 cleaned_docs.append(doc)
79
80 return cleaned_docs
81
82 def text_split(documents):
83 # 文本分块
84 text_splitter = RecursiveCharacterTextSplitter(
85 separators=["\n\n", "], # 分隔符列表
86 chunk_size=400, # 每个块的最大长度
87 chunk_overlap=40, # 每个块重叠的长度
88
89 texts = text_splitter.split_documents(documents)
90 return texts
91
92 def save_to_db(texts):
93 # 加载嵌入模型
94 embedding_model = HuggingFaceEmbeddings(
95 model_name="./bge-base-zh-v1.5",
96 model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
97 encode_kwargs={
98 "normalize_embeddings": True
99 }, # 输出归一化向量，更适合余弦相似度计算
100
101 # 嵌入并存储到向量数据库
102 vectorstore = Chroma.from_documents(
103 texts, # 文档列表
104 embedding_model, # 嵌入模型
105 persist_directory="vectorstore", # 存储路径
106 )
107 return vectorstore
108
109 if __name__ == _main
110 # 1. 加载文档
111 documents = load_documents()
112 # 2、清洗文档
113 cleaned_docs = clean_content(documents)
114 # 3、切分文档
115 texts = text_split(cleaned_docs)
116 # 4、保存到数据库中
117 vectorstore = save_to_db(texts)
118 # 5、查看数据库内容
119 print(vectorstore.get().keys()) # 查看所有属性
120 print(vectorstore.get(include=["embeddings"])["embeddings"][:5, :5]) # 查看
嵌入向量
```

## 5.2 检索模块

```python
1 from typing import Dict
2
3 from dotenv import load_dotenv
4 from langchain_chroma import Chroma
5 from langchain.chat_models import init_chat_model
```

```python
6 from langchain_core.documents import Document
7 from langchain_core.output_parsers import StrOutputParser
8 from langchain_core.prompts import PromptTemplate
9
10 def format_history(history,max_epoch=3):
11 # 每轮对话有 用户问题 和 助手回复
12 if len(history) > 2 * max_epoch:
13 history = history[-2 * max_epoch :]
14 return "\n".join([f"{i['role']}：{i['content']}" for i in history])
15
16 def format_docs(docs: list[Document]) -> str:
17 """拼接多个 Document 的 page_content """
18 return "\n\n".join(doc.page_content for doc in docs)
19
20 def get_retriever(k=20,embedding_model=None):
21 """获取向量数据库的检索器"""
22 # 1、初始化 Chroma 客户端
23 vectorstore = Chroma(
24 persist_directory="vectorstore",
25 embedding_function=embedding_model,
26 )
27
28 # 2、创建向量数据库检索器
29 retriever = vectorstore.as_retriever(
30 search_type="similarity", # 检索方式，similarity 或 mmr
31 search_kwargs={"k": k},
32 )
33 return retriever
34
35 def get_llm():
36 # 大模型
37 load_dotenv()
38 llm = init_chat_model(model="gpt-4o")
39 return llm
40
41 def rephrase_retrieve(input:Dict[str,str],llm,retriever):
42 """重述用户query，检索向量数据库"""
43
44 # 1、重述query的prompt
45 rephrase_prompt = PromptTemplate.from_template(
46
47 根据对话历史简要完善最新的用户消息，使其更加具体。只输出完善后的问题。如果问题不需要完善，
请直接输出原始问题。
48
49 {history}
50 用户：{query}
51
52 )
53
54 # 2、重述链条：根据历史和当前 query 生成更具体问题
55 rephrase_chain = (
56 {
57 "history": lambda x :format_history(x.get("history")),
58 "query": lambda x: x.get("query"),
59 }
60 | rephrase_prompt
61 llm
62 | StrOutputParser()
```

```python
63 | (lambda x: print(f"===== 重述后的查询: {x}=====") or x)
64
65
66 # 3、执行重述
67 rephrase_query = rephrase_chain.invoke({"history": input.get("history"),
"query": input.get("query")})
68
69 # 4、使用重述后的query进行检索
70 retrieve_result = retriever.invoke(rephrase_query,k=3)
71
72 return retrieve_result
73
74 def get_rag_chain(retrieve_result,llm):
75 """构建RAG链条：使用检索结果、历史记录、用户查询，提交大模型生成回复"""
76
77 # 1、Prompt 模板
78 prompt = PromptTemplate(
79 input_variables=["context", "history", "query"],
80 template=""
81 你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
82 请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。
83
84 背景资料：{context}
85
86 历史消息：[{history}]
87
88 问题：{query}
89
90 回答：
91 )
92
93 # 2、定义 RAG 链条
94 rag_chain = (
95 {
96 "context": lambda x:format_docs(retrieve_result),
97 "history": lambda x: format_history(x.get("history")),
98 "query": lambda x: x.get("query"),
99 }
100 | prompt
101 (lambda x: print(x.text, end="") or x) #打印
102 | llm
103 StrOutputParser() # 输出解析器，将输出解析为字符串
104
105
106 return rag_chain
```

## 5.3 RAG 链条

用于执行检索模块封装好的 rag_chain

1 import asyncio   
2 import torch   
3 from langchain_huggingface import HuggingFaceEmbeddings   
4 from retrieve import rephrase_retrieve, get_rag_chain, get_llm, get_retriever   
5   
6 # 存储对话历史

```python
7 chat_history = []
8
9 # 1、初始化Embedding模型
10 embedding_model = HuggingFaceEmbeddings(
11 model_name="./bge-base-zh-v1.5",
12 model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
13 encode_kwargs={
14 "normalize_embeddings": True
15 }, # 输出归一化向量，更适合余弦相似度计算
16 )
17
18 # 2、初始化 LLM
19 llm = get_llm()
20
21 async def invoke_rag(query,conversation_id,chat_history):
22
23 answer = ""
24
25 input={"query":query,"history":chat_history}
26
27 # 1、获取检索器
28 retriever=get_retriever(k=20,embedding_model=embedding_model)
29 # 2、执行重述、检索
30 retrieve_result= rephrase_retrieve(input,llm,retriever)
31 # 3、获取RAG链
32 rag_chain = get_rag_chain(retrieve_result,llm)
33 # 4、异步执行RAG链，流式输出
34 async for chunk in rag_chain.astream(input):
35 answer += chunk
36 yield chunk # 将大模型生成的内容逐块(chunk)地返回给调用者，而不是等待整个回答完成后
一次性返回
37
38 # 5、更新对话历史，添加用户查询和AI回答
39 chat_history.append(
40 {"role": "user", "content": query, "conversation_id": conversation_id}
41 )
42 chat_history.append(
43 {"role": "ai", "content": answer, "conversation_id": conversation_id}
44 )
45
46
47 if __name__ == '__main__
48 async def main():
49 query_list = ["不动产或者动产被人占有怎么办", "那要是被损毁了呢"]
50 for query in query_list:
51 print(f"===== 查询: {query} =====")
52 async for chunk in invoke_rag(query,1,chat_history):
53 print(chunk, end="", flush=True)
54
55 asyncio.run(main())
```

## 5.4 流式响应 API 封装

```python
1 import uvicorn
2 from fastapi import FastAPI
3 from fastapi.responses import FileResponse
```

```python
4 from fastapi.staticfiles import StaticFiles
5 from fastapi.responses import StreamingResponse
6
7 from rag import invoke_rag, chat_history
8
9 # 创建 FastAPI 实例
10 app = FastAPI()
11
12 # 挂载静态文件
13 app.mount("/static", StaticFiles(directory="templates"))
14
15 @app.get("/")
16 async def homepage():
17 return FileResponse("templates/naive_index.html")
18
19 @app.get("/stream_response")
20 async def stream_response(query: str):
21 return StreamingResponse(
22 invoke_rag(query,1,chat_history), media_type="text/event-stream"
23 )
24
25 if __name__ == _main
26 uvicorn.run(app, host="127.0.0.1", port=8089)
```

拷贝资料中的 templates 文件夹到工程根目录中，运行 app.py ,打开页面：

http://127.0.0.1:8089

不动产或者动产被人占有怎么办  
如果不动产或者动产被占有人占有，权利人可以请求返还原物及其擎息。但是，应当支付善意占有人因维护该不动产或  
者动产支出的必要费用。  
发送

如果运行 app.py ，提示端口占用，可以 cmd 执行如下命令释放：