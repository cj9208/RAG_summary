

This repo is to summarize relevant pages and repos that are related to RAG and Agent.

- [Overview](#overview)
- [Target or Object](#target-or-object)
  - [RAG](#rag)
  - [Agent](#agent)
- [Framework for LLM and Agent](#framework-for-llm-and-agent)
- [Algorithms and Implementations](#algorithms-and-implementations)


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
| keep udating | Mem0 | [github](https://github.com/mem0ai/mem0) | Enhance AI assistants and agents with an intelligent memory layer, enabling personalized interactions. Github Trend #1
| 2510 | Scalable In-context Ranking with Generative Models | [arxiv](https://arxiv.org/pdf/2510.05396v2) | information retrieval through in-context ranking, not vector similarity
| 2411 | ColiVara | [github](https://github.com/tjmlabs/ColiVara) | Use visual embeddings, so no text extraction step


