

This repo is to summarize relevant pages and repos that are related to RAG and Agent.


1. data preparation
2. framework
3. prompt engineering
4. deployment and operations
5. learning materials




- [Overview](#overview)
- [Target or Object](#target-or-object)
  - [RAG](#rag)
  - [Agent](#agent)


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



