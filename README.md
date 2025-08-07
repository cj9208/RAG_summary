# RAG and Agent Summary

This repo is to summarize relevant pages and repos that are related to RAG

## Target or Object of RAG
1. Deal with various types of data
    * text, image, video, audio, etc.
    * pdf, docx, txt, json, csv, etc.
2. Answer questions by searching relevant data. Mention the source of the answer and its confidence level.


## Data Preparation 
### Extraction and Parsing

| Tool/Project | Links | Description |
|--------------|-------| -------|
|RAG Challenge Winner Solution| [github](https://github.com/IlyaRice/RAG-Challenge-2) |
| Agentic Document Extraction | [github](https://github.com/landing-ai/agentic-doc) |
| DocETL | [github](https://github.com/ucbepic/docetl) |
| PDF-Extract-Kit | [github](https://github.com/opendatalab/PDF-Extract-Kit)
| LangExtract by Google | [github](https://github.com/google/langextract)
| deepdoc | [github](https://github.com/infiniflow/ragflow/tree/main/deepdoc)
| MinerU | [github](https://opendatalab.github.io/MinerU/)
| pypdf | [github](https://github.com/py-pdf/pypdf)
| PyMuPDF | [github](https://github.com/pymupdf/PyMuPDF)
| Mistral OCR| [offical page](https://mistral.ai/news/mistral-ocr)
| olm OCR | [github](https://github.com/allenai/olmocr)
| PaddleOCR | [github](https://github.com/PaddlePaddle/PaddleOCR)
| PP-StructreV3 | [official page](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html), [arxiv](https://arxiv.org/abs/2210.05391)
| unstructrued | [github](https://github.com/Unstructured-IO/unstructured)
| dots.ocr | [github](https://github.com/rednote-hilab/dots.ocr)
| docling | [github](https://github.com/docling-project/docling)
| maker | [github](https://github.com/datalab-to/marker)
| nougat | [github](https://github.com/facebookresearch/nougat)
| GOT-OCR | [github](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
| SmolDocling OCR | [github](https://github.com/AIAnytime/SmolDocling-OCR-App)
| EasyOCR | [github](https://github.com/JaidedAI/EasyOCR)
| surya | [github](https://github.com/datalab-to/surya)
| LayoutLMv3 | [github](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
| DocLayout-YOLO| [github](https://github.com/opendatalab/DocLayout-YOLO) 
| LatexOCR| [github](https://github.com/lukas-blecher/LaTeX-OCR)
| RapidTable| [github](https://github.com/RapidAI/RapidTable) |
| DataFlow| [github](https://github.com/OpenDCAI/DataFlow), [document](https://opendcai.github.io/DataFlow-Doc/zh/) | Data-centric AI system



### Evaluation
* Dingo, [github](https://github.com/MigoXLab/dingo)


## Framework and LLM related
| Tool/Project | Links | Description |
|--------------|-------| -------|
| langgraph | [api](https://langchain-ai.github.io/langgraph/concepts/low_level/)
| autogen by Microsoft | [github](https://github.com/microsoft/autogen)
| anyLLM | [github](https://github.com/mozilla-ai/any-llm)
| oneapi | [github](https://github.com/songquanpeng/one-api)
| openai agent python| [github](https://github.com/openai/openai-agents-python)
|GenAI Processors Library by Google | [github](https://github.com/google-gemini/genai-processors) | The concept of Processor provides a common abstraction for Gemini model calls and increasingly complex behaviors built around them, accommodating both turn-based interactions and live streaming.
| Qwen Agent | [github](https://github.com/QwenLM/Qwen-Agent)
| WebAgent by Tongyi Lab | [github](https://github.com/Alibaba-NLP/WebAgent)
| ragflow | [github](https://github.com/infiniflow/ragflow)
|BMad-Method |[github](https://github.com/bmadcode/BMAD-METHOD) | A Universal AI Agent Framework
|tooluse| [github](https://github.com/BeautyyuYanli/tooluser) | Enable tool-use ability for any LLM model (DeepSeek V3/R1, etc.) 
| pocketflow | [github](https://github.com/The-Pocket/PocketFlow) | Pocket Flow is a 100-line minimalist LLM framework




## Algorithms and Implementations
|Date  | Tool/Project | Links | Description |
|------|------------|-------| -------|
|2410 | LightRAG | [github](https://github.com/HKUDS/LightRAG)
| 2507 | DeepSieve | [Arixv](https://arxiv.org/abs/2507.22050)
| 2506 | Agentic large language models improve retrieval-based radiology question answering | [Arxiv](https://arxiv.org/pdf/2508.00743)
| 2507 | SemRAG | [arxiv](https://arxiv.org/pdf/2507.21110)
| 2507 | RAG-Anything | [github](https://github.com/HKUDS/RAG-Anything)
|  | MemAgent | [github](https://github.com/BytedTsinghua-SIA/MemAgent)
| | graphRAG | [github](https://github.com/microsoft/graphrag)
| 2506 | memvid | [github](https://github.com/Olow304/memvid) | Memvid compresses an entire knowledge base into MP4 files while keeping millisecond-level semantic search
|| SemHash | [github](https://github.com/MinishLab/semhash) | SemHash is a lightweight and flexible tool for deduplicating datasets, filtering outliers, and finding representative samples using semantic similarity


## Courses and Learning Resouces

* [Huggingface Daily papers](https://huggingface.co/papers/week/2025-W30)
* [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
* [gemini fullstack langgraph quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)
* [graph-rag-agent](https://github.com/1517005260/graph-rag-agent) 
    * combine graphrag and deepresearch
* [graphrag-more](https://github.com/guoyao/graphrag-more)
    * add support for other LLM (better for using one-api or any-LLM as in graph-rag-agent)
* [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production)
* [deep research with Gemini](https://github.com/u14app/deep-research)
* [Microsoft AI and ML Engineering Professional Certificate](https://www.coursera.org/professional-certificates/microsoft-ai-and-ml-engineering#courses)


## reference

| Description | Link |
|-------------|------|
|RAG techniques|[github](https://github.com/NirDiamant/RAG_Techniques)|
| 开源工具全家桶，实战总结，落地技术选型 | [link](https://www.xiaohongshu.com/explore/6807c26c000000001d0242f1?app_platform=android&ignoreEngage=true&app_version=8.94.2&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CBrnuw9M3TKKHmh51KCDVCqespfZMlr7U-XAuVlTmt0KA=&author_share=1&xhsshare=WeixinSession&shareRedId=N0o4RTRHNU02NzUyOTgwNjY0OThGOjY7&apptime=1754375009&share_id=ea20ea452df84d88a122ff85f3dff592&share_channel=wechat) |
| 总结！企业级RAG系统PDF解析工具！ | [link](https://www.xiaohongshu.com/explore/680fa4e4000000000e006567?app_platform=android&ignoreEngage=true&app_version=8.94.2&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CBy4faorZqbjvw01M4DcK27S4j_p_QPbMtMWjlcTaZnWI=&author_share=1&xhsshare=WeixinSession&shareRedId=N0o4RTRHNU02NzUyOTgwNjY0OThGOjY7&apptime=1754375059&share_id=97c7e2ede8a7400e909e55a0eff6c687&share_channel=wechat)|

