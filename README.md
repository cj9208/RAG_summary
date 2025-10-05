

This repo is to summarize relevant pages and repos that are related to RAG and Agent.

- [Overview](#overview)
- [Target or Object](#target-or-object)
  - [RAG](#rag)
  - [Agent](#agent)
- [Data Preparation](#data-preparation)
  - [Generation](#generation)
  - [Extraction and Parsing](#extraction-and-parsing)
  - [Evaluation](#evaluation)
- [Prompt](#prompt)
- [Framework for LLM and Agent](#framework-for-llm-and-agent)
- [Algorithms and Implementations](#algorithms-and-implementations)
- [ML/AIOps (Deployment, Monitoring, Evaluation)](#mlaiops-deployment-monitoring-evaluation)
- [Courses and Learning Resouces](#courses-and-learning-resouces)
- [Reference](#reference)


## Overview

Components of RAG and Agent
* Retrieval
  * Data preparation (parsing, extraction from all kinds of data)
  * 
* Generation
* Evaluation
* Deployment
* Monitoring
* Workflow

## Target or Object
### RAG
Enhance LLM with external data
1. Data
    * text, image, video, audio, etc.
    * pdf, docx, txt, json, csv, etc.
2. Algorithm, i.e. how to do the retrival and generation
    * e.g. graphRAG, generate a knowledge graph to help answer global questions
3. Deployment 
    * LLM (API or local deployment)
    * database
    * monitoring and evaluation

### Agent
Autonomous system that uses an underlying large language model (LLM) to perform complex, multi-step tasks to achieve a specific goal.
1. Key characteritics
   * goal oriented
   * reasoning and planning
   * tool use
   * execution and iteration
2. e.g. Deep research, use a series of iterative and systematic steps to autonomously conduct in-depth research on a given topic.
    * planning and question generation
    * information retrieval
    * synthesis and analyze (including evaluation, if the retrieval is not satisfied, the agent may go to step 1 again) 
    * report generation
3. Deployment is similar to RAG


## Data Preparation 
### Generation
* [GraphGEN ](https://github.com/idea-iitd/graphgen)
* [Synthetic Data Vault (SDV)](https://github.com/sdv-dev/SDV)
* [Generative Data Refinement by DeepMind](https://arxiv.org/abs/2509.08653v1)
### Extraction and Parsing

| format /method | Tool/Project                | Links   | Description                                                                                                                                                                                |
| -------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| pdf            | PyMuPDF                     | [github](https://github.com/pymupdf/PyMuPDF)                                                                                                              | can read fonts                                                                                                                                                                             |
|                | MinerU                      | [github](https://github.com/opendatalab/MinerU)                                                                                                           | using vlm to help extract tables                                                                                                                                                           |
|                | Agentic Document Extraction | [github](https://github.com/landing-ai/agentic-doc)                                                                                                       |
|                | PDF-Extract-Kit             | [github](https://github.com/opendatalab/PDF-Extract-Kit)                                                                                                  |
|                | pypdf                       | [github](https://github.com/py-pdf/pypdf)                                                                                                                 |
| OCR            | Mistral OCR                 | [offical page](https://mistral.ai/news/mistral-ocr)                                                                                                       |
|                | olm OCR                     | [github](https://github.com/allenai/olmocr)                                                                                                               |
|                | PaddleOCR                   | [github](https://github.com/PaddlePaddle/PaddleOCR)    | from Baidu, updated to V5                                                                                                   |
|                | dots.ocr             | [github](https://github.com/rednote-hilab/dots.ocr) | from RedNote                                                                                                       |
|                | GOT-OCR                     | [github](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)                                                                                                    |
|                | SmolDocling OCR             | [github](https://github.com/AIAnytime/SmolDocling-OCR-App)                                                                                                |
|                | EasyOCR                     | [github](https://github.com/JaidedAI/EasyOCR)                                                                                                             |
|                | LatexOCR                    | [github](https://github.com/lukas-blecher/LaTeX-OCR)                                                                                                      |
|                | surya                       | [github](https://github.com/datalab-to/surya)                                                                                                             | Surya is an OCR toolkit supporting 90+ languages, with layout analysis, reading order, table, and LaTeX recognition                                                                        |
| format related | maker                       | [github](https://github.com/datalab-to/marker)                                                                                                            | converts documents to markdown, JSON, chunks, and HTML quickly and accurately.                                                                                                             |
|                | markitdown by Microsoft     | [github](https://github.com/microsoft/markitdown)                                                                                                         | convert pdf, ppt and etc to markdown                                                                                                                                                       |
| LLM-based      | DocETL                      | [github](https://github.com/ucbepic/docetl)                                                                                                               | An interactive UI for iterative prompt engineering and pipeline development, and a Python package                                                                                          |
|                | LangExtract by Google       | [github](https://github.com/google/langextract)                                                                                                           |
|                | data juicer by alibaba      | [github](https://github.com/modelscope/data-juicer)                                                                                                       | Data-Juicer is a one-stop system to process text and multimodal data for and with foundation models (typically LLMs).                                                                      |
|                | deepdoc                     | [github](https://github.com/infiniflow/ragflow/tree/main/deepdoc)                                                                                         | DeepDoc addresses the challenge of accurately analyzing diverse documents with varying formats and retrieval needs through its two components: vision and parser.                          |
|                | easy-dataset                | [github](https://github.com/ConardLi/easy-dataset)                                                                                                        | Create fine-tuning datasets for Large Language Models (LLMs)                                                                                                                               |
| others         | PP-StructreV3               | [official page](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html), [arxiv](https://arxiv.othersorg/abs/2210.05391) |
|                | unstructrued                | [github](https://github.com/Unstructured-IO/unstructured)                                                                                                 |
|                | docling                     | [github](https://github.com/docling-project/docling)                                                                                                      | Docling parses diverse documents with advanced PDF, OCR, and multimodal support, offers flexible exports and local processing, and integrates seamlessly with major AI frameworks via CLI. |
|                | nougat                      | [github](https://github.com/facebookresearch/nougat)                                                                                                      | the academic document PDF parser that understands LaTeX math and tables.                                                                                                                   |
|                | LayoutLMv3                  | [github](https://github.com/microsoft/unilm/tree/master/layoutlmv3)                                                                                       |
|                | DocLayout-YOLO              | [github](https://github.com/opendatalab/DocLayout-YOLO)                                                                                                   | layout detection model for diverse documents, based on YOLO-v10                                                                                                                            |
|                | RapidTable                  | [github](https://github.com/RapidAI/RapidTable)                                                                                                           |
|                | DataFlow                    | [github](https://github.com/OpenDCAI/DataFlow), [document](https://opendcai.github.io/DataFlow-Doc/zh/)                                                   | Data-centric AI system                                                                                                                                                                     |

### Evaluation
* Dingo, [github](https://github.com/MigoXLab/dingo), data quality evaluation tool 

## Prompt
* [MineContext](https://github.com/volcengine/MineContext)


## Framework for LLM and Agent
| Tool/Project                       | Links                                                               | Description                                                                                                                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| langgraph                          | [api](https://langchain-ai.github.io/langgraph/concepts/low_level/) | focus on workflow of agents                                                                                                                                                                        |
| autogen by Microsoft               | [github](https://github.com/microsoft/autogen)                      | focus on interaction of agents                                                                                                                                                                     |
| pydantic AI                        | [doc](https://docs.pydantic.dev/latest/)                            | lightweight framework for agent   |
|AutoAgent by HKUDS| [github](https://github.com/HKUDS/AutoAgent) | fully-Automated and highly Self-Developing framework that enables users to create and deploy LLM agents through Natural Language Alone
| n8n                                | [oficial page](https://n8n.io/)                                     | graphical workflow engine                                                                                                                                                                          |
| anyLLM                             | [github](https://github.com/mozilla-ai/any-llm)                     |
| oneapi                             | [github](https://github.com/songquanpeng/one-api)                   | unified API for multiple LLM providers                                                                                                                                                             |
| openai agent python SDK            | [github](https://github.com/openai/openai-agents-python)            |
| GenAI Processors Library by Google | [github](https://github.com/google-gemini/genai-processors)         | The concept of Processor provides a common abstraction for Gemini model calls and increasingly complex behaviors built around them, accommodating both turn-based interactions and live streaming. |
| UltraRAG by Tsinghua University| [github](https://github.com/OpenBMB/UltraRAG) | Less Code, Lower Barrier, Faster Deployment. Can build high-performance RAG with just a few dozen lines of code
| Qwen Agent                         | [github](https://github.com/QwenLM/Qwen-Agent)                      |
| WebAgent by Tongyi Lab             | [github](https://github.com/Alibaba-NLP/WebAgent)                   |
| Tongyi DeepResearch| [blog](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/), [github](https://github.com/Alibaba-NLP/DeepResearch)
| open deep research by langchain    | [github](https://github.com/langchain-ai/open_deep_research)        |
| ragflow                            | [github](https://github.com/infiniflow/ragflow)                     |
| tooluse                            | [github](https://github.com/BeautyyuYanli/tooluser)                 | Enable tool-use ability for any LLM model (DeepSeek V3/R1, etc.)                                                                                                                                   |
| pocketflow                         | [github](https://github.com/The-Pocket/PocketFlow)                  | Pocket Flow is a 100-line minimalist LLM framework                                                                                                                                                 |




## Algorithms and Implementations
| Date | Tool/Project | Links                                                   | Description                                                                                                                                             |
| ---- | ------------ | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2410 | LightRAG     | [github](https://github.com/HKUDS/LightRAG)             |
| 2410 | RAG-Anything | [github](https://github.com/HKUDS/RAG-Anything)         |
| 2404 | graphRAG     | [github](https://github.com/microsoft/graphrag)         |
|| parlant | [github](https://github.com/emcie-co/parlant) | Finally, LLM agents that actually follow instructions
| 2507 | DeepSieve    | [Arixv](https://arxiv.org/abs/2507.22050)               |
| 2507 | SemRAG       | [arxiv](https://arxiv.org/pdf/2507.21110)               |
| 2507 | MemAgent   by ByteDance  | [github](https://github.com/BytedTsinghua-SIA/MemAgent) | superb long-context capabilities, able to extrapolate from an 8K context to a 3.5M QA task with performance loss < 5% and achieves 95%+ accuracy in 512K RULER test.
| 2410 | memary| [github](https://github.com/kingjulio8238/Memary)
| 2506 | memvid       | [github](https://github.com/Olow304/memvid)             | Memvid compresses an entire knowledge base into MP4 files while keeping millisecond-level semantic search                                               |
|      | SemHash      | [github](https://github.com/MinishLab/semhash)          | SemHash is a lightweight and flexible tool for deduplicating datasets, filtering outliers, and finding representative samples using semantic similarity |
| 2509 | Lexical Diversity-aware Relevance Assessment for RAG | [acl](https://aclanthology.org/2025.acl-long.1346.pdf)
| 2508 |SynRewrite | [arxiv]([SynRewrite](https://arxiv.org/abs/2509.22325))


## ML/AIOps (Deployment, Monitoring, Evaluation)
Key components:

a. CI/CD (Continuous Integration/Continuous Deployment)
  * CI: Automatically tests new code changes (e.g., data preprocessing scripts, model training code) to ensure they don't break the system.
  * CD: Automates the deployment of a new model or a new version of the serving code to a staging or production environment.

b. Versioning and Registry
  * Model Registry: A centralized repository to store and manage different versions of trained models. This allows you to track which model version is in production and easily roll back if necessary.
  * Data and Code Versioning: Just as code needs to be versioned, so does data. Tools like DVC (Data Version Control) help track changes in datasets.

c. Monitoring
  * Model Performance Monitoring: Once a model is in production, it's crucial to monitor its performance. This includes:
  * Drift Detection: Monitoring for data drift (when the distribution of new data changes) or concept drift (when the relationship between input variables and the target variable changes).
  * Prediction Quality: Tracking the model's accuracy, precision, and other metrics in the production environment.
  * Infrastructure Monitoring: Monitoring the health of the underlying infrastructure, such as CPU and memory usage of the serving endpoints.

d. Automated Retraining
  * Trigger: When monitoring detects model performance degradation (e.g., a drop in accuracy or significant data drift), the MLOps pipeline can be automatically triggered to retrain the model.
  * Process: The pipeline uses a fresh dataset to retrain the model, evaluates the new version, and then automatically deploys it if it meets the performance criteria.



| Tool/Project | Links                            | Description                                       |
| ------------ | -------------------------------- | ------------------------------------------------- |
| langfuse     | [doc](https://langfuse.com/docs) | Observability, Pompot Optimization, and Evalution |


## Courses and Learning Resouces
* [RAG Challenge Winner Solution](https://github.com/IlyaRice/RAG-Challenge-2) 
* [Huggingface Daily papers](https://huggingface.co/papers/week/2025-W30)
* [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
* [gemini fullstack langgraph quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)
* [graph-rag-agent](https://github.com/1517005260/graph-rag-agent) 
    * combine graphrag and deepresearch
* [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production)
* [deep research with Gemini](https://github.com/u14app/deep-research)
* [Microsoft AI and ML Engineering Professional Certificate](https://www.coursera.org/professional-certificates/microsoft-ai-and-ml-engineering#courses)
* [RAG Time: Ultimate Guide to Mastering RAG](https://github.com/microsoft/rag-time)
* [RAGHub](https://github.com/Andrew-Jang/RAGHub)
* [Engineering at Anthropic](https://www.anthropic.com/engineering)
* [How is NASA Building a People Knowledge Graph with LLMs and Memgraph](https://www.crowdcast.io/c/how-is-nasa-building-a-people-knowledge-graph-with-llms-and-memgraph)
* [From Search to Reasoning: A Five-Level RAG Capability Framework for Enterprise Data](https://www.arxiv.org/pdf/2509.21324)


## Reference

| Description                                                                | Link                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RAG techniques                                                             | [github](https://github.com/NirDiamant/RAG_Techniques)                                                                                                                                                                                                                                                                                                                                                                     |
| 开源工具全家桶，实战总结，落地技术选型                                     | [link](https://www.xiaohongshu.com/explore/6807c26c000000001d0242f1?app_platform=android&ignoreEngage=true&app_version=8.94.2&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CBrnuw9M3TKKHmh51KCDVCqespfZMlr7U-XAuVlTmt0KA=&author_share=1&xhsshare=WeixinSession&shareRedId=N0o4RTRHNU02NzUyOTgwNjY0OThGOjY7&apptime=1754375009&share_id=ea20ea452df84d88a122ff85f3dff592&share_channel=wechat) |
| 总结！企业级RAG系统PDF解析工具！                                           | [link](https://www.xiaohongshu.com/explore/680fa4e4000000000e006567?app_platform=android&ignoreEngage=true&app_version=8.94.2&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CBy4faorZqbjvw01M4DcK27S4j_p_QPbMtMWjlcTaZnWI=&author_share=1&xhsshare=WeixinSession&shareRedId=N0o4RTRHNU02NzUyOTgwNjY0OThGOjY7&apptime=1754375059&share_id=97c7e2ede8a7400e909e55a0eff6c687&share_channel=wechat) |
|                                                                            |
| A Survey of self-evolving agents: on path to Artificial Super Intelligence | [arxiv](https://arxiv.org/pdf/2507.21046)        
| A Survey on AgentOps: Categorization, Challenges, and Future Directions | [arxiv](https://arxiv.org/pdf/2508.02121v1)        
| 5-Day Gen AI Intensive Course with Google | [link](https://www.kaggle.com/learn-guide/5-day-genai?utm_medium=email&utm_source=gamma&utm_campaign=learn-5daygenai)      
| What makes Claude Code so damn good  | [link](https://minusx.ai/blog/decoding-claude-code/#31-llm-search---rag-based-search)        
|awesome-RAG| [github](https://github.com/liunian-Jay/Awesome-RAG)                                                                                                                                                                                                                                                                                                                                                     |
