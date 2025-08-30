

This repo is to summarize relevant pages and repos that are related to RAG and Agent.

- [Target or Object](#target-or-object)
  - [RAG](#rag)
  - [Agent](#agent)
- [Data Preparation](#data-preparation)
  - [Extraction and Parsing](#extraction-and-parsing)
  - [Evaluation](#evaluation)
- [Framework for LLM and Agent](#framework-for-llm-and-agent)
- [Algorithms and Implementations](#algorithms-and-implementations)
- [ML/AIOps (Deployment, Monitoring, Evaluation)](#mlaiops-deployment-monitoring-evaluation)
- [Courses and Learning Resouces](#courses-and-learning-resouces)
- [Reference](#reference)




## Target or Object
### RAG
1. Deal with various types of data
    * text, image, video, audio, etc.
    * pdf, docx, txt, json, csv, etc.
2. Answer questions by searching relevant data. Mention the source of the answer and its confidence level.

### Agent
1. Leverage LLM to automate tasks


## Data Preparation 
### Extraction and Parsing

| format /method | Tool/Project                | Links                                                                                                                                                     | Description                                                                                                                                                                                |
| -------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| pdf            | PyMuPDF                     | [github](https://github.com/pymupdf/PyMuPDF)                                                                                                              | can read fonts                                                                                                                                                                             |
|                | MinerU                      | [github](https://github.com/opendatalab/MinerU)                                                                                                           | using vlm to help extract tables                                                                                                                                                           |
|                | Agentic Document Extraction | [github](https://github.com/landing-ai/agentic-doc)                                                                                                       |
|                | PDF-Extract-Kit             | [github](https://github.com/opendatalab/PDF-Extract-Kit)                                                                                                  |
|                | pypdf                       | [github](https://github.com/py-pdf/pypdf)                                                                                                                 |
| OCR            | Mistral OCR                 | [offical page](https://mistral.ai/news/mistral-ocr)                                                                                                       |
|                | olm OCR                     | [github](https://github.com/allenai/olmocr)                                                                                                               |
|                | PaddleOCR                   | [github](https://github.com/PaddlePaddle/PaddleOCR)                                                                                                       |
|                | dots.ocr                    | [github](https://github.com/rednote-hilab/dots.ocr)                                                                                                       |
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


## Framework for LLM and Agent
| Tool/Project                       | Links                                                               | Description                                                                                                                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| langgraph                          | [api](https://langchain-ai.github.io/langgraph/concepts/low_level/) | focus on workflow of agents                                                                                                                                                                        |
| autogen by Microsoft               | [github](https://github.com/microsoft/autogen)                      | focus on interaction of agents                                                                                                                                                                     |
| pydantic AI                        | [doc](https://docs.pydantic.dev/latest/)                            | lightweight framework for agent                                                                                                                                                                    |
| n8n                                | [oficial page](https://n8n.io/)                                     | graphical workflow engine                                                                                                                                                                          |
| anyLLM                             | [github](https://github.com/mozilla-ai/any-llm)                     |
| oneapi                             | [github](https://github.com/songquanpeng/one-api)                   | unified API for multiple LLM providers                                                                                                                                                             |
| openai agent python SDK            | [github](https://github.com/openai/openai-agents-python)            |
| GenAI Processors Library by Google | [github](https://github.com/google-gemini/genai-processors)         | The concept of Processor provides a common abstraction for Gemini model calls and increasingly complex behaviors built around them, accommodating both turn-based interactions and live streaming. |
| Qwen Agent                         | [github](https://github.com/QwenLM/Qwen-Agent)                      |
| WebAgent by Tongyi Lab             | [github](https://github.com/Alibaba-NLP/WebAgent)                   |
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
| 2507 | DeepSieve    | [Arixv](https://arxiv.org/abs/2507.22050)               |
| 2507 | SemRAG       | [arxiv](https://arxiv.org/pdf/2507.21110)               |
|      | MemAgent     | [github](https://github.com/BytedTsinghua-SIA/MemAgent) |
| 2506 | memvid       | [github](https://github.com/Olow304/memvid)             | Memvid compresses an entire knowledge base into MP4 files while keeping millisecond-level semantic search                                               |
|      | SemHash      | [github](https://github.com/MinishLab/semhash)          | SemHash is a lightweight and flexible tool for deduplicating datasets, filtering outliers, and finding representative samples using semantic similarity |


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


## Reference

| Description                                                                | Link                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RAG techniques                                                             | [github](https://github.com/NirDiamant/RAG_Techniques)                                                                                                                                                                                                                                                                                                                                                                     |
| 开源工具全家桶，实战总结，落地技术选型                                     | [link](https://www.xiaohongshu.com/explore/6807c26c000000001d0242f1?app_platform=android&ignoreEngage=true&app_version=8.94.2&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CBrnuw9M3TKKHmh51KCDVCqespfZMlr7U-XAuVlTmt0KA=&author_share=1&xhsshare=WeixinSession&shareRedId=N0o4RTRHNU02NzUyOTgwNjY0OThGOjY7&apptime=1754375009&share_id=ea20ea452df84d88a122ff85f3dff592&share_channel=wechat) |
| 总结！企业级RAG系统PDF解析工具！                                           | [link](https://www.xiaohongshu.com/explore/680fa4e4000000000e006567?app_platform=android&ignoreEngage=true&app_version=8.94.2&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CBy4faorZqbjvw01M4DcK27S4j_p_QPbMtMWjlcTaZnWI=&author_share=1&xhsshare=WeixinSession&shareRedId=N0o4RTRHNU02NzUyOTgwNjY0OThGOjY7&apptime=1754375059&share_id=97c7e2ede8a7400e909e55a0eff6c687&share_channel=wechat) |
|                                                                            |
| A Survey of self-evolving agents: on path to Artificial Super Intelligence | [arxiv](https://arxiv.org/pdf/2507.21046)                                                                                                                                                                                                                                                                                                                                                                                  |
