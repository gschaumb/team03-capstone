---
title: Team03 Capstone
emoji: 📚
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.8.0
app_file: app.py
pinned: false
license: bsd-3-clause
---

This file is integral to the configuration and deployment of our Capstone project on HuggingFace Spaces. The configuration information above provides build information for the HuggingFace Space. 

### Project Description

 Our Capstone project for the University of Michigan MADS Program expands upon our team's work in Milestone 2, with the goal of testing whether agentic RAG improves results with an Enron source knowledgebase when compared with basic RAG and unaugmented LLM responses. Additionally, it includes an agent to search via API through the extensive Enron email corpus (which we have implemented in an AWS service) and retrieve emails related to the user's query.

 When deployed to a HuggingFace Gradio Space the code provided in this repository creates the Capstone Demo application linked below.

### Capstone Project Links

- **Capstone Project Overview and Report:**  
  For our Capstone project overview and report, see the GitHub Page at:  
  [https://gschaumb.github.io/team03-capstone/](https://gschaumb.github.io/team03-capstone/)

- **Capstone Demo:**  
  Our Capstone demo can be found at:  
  [https://huggingface.co/spaces/gschaumb/team03-capstone](https://huggingface.co/spaces/gschaumb/team03-capstone)

- **Milestone 2 Demo:**  
  Our Milestone 2 demo (used in Capstone evaluation) can be found at:  
  [https://huggingface.co/spaces/gschaumb/M2-Team01-SS24-RAG-demo](https://huggingface.co/spaces/gschaumb/M2-Team01-SS24-RAG-demo)

- **Evaluation Report:**
  Our detailed evaluation of queries can be found at:
  <a href="https://docs.google.com/spreadsheets/d/e/2PACX-1vSazYHLTkUGFPhN0KqDidJPrdtEYcs3apmemOkPjzgAiDhl2EkmxVk55jNxtcmBArZyzzSyyAPiA6nU/pubhtml?widget=true&amp;headers=false" target="_blank">Link to Evaluation Sheet</a>
  

### HuggingFace Space Setup

As mentioned above, this file is used in the setup of the HuggingFace Space and provides necessary configuration information. See the following for more information.

- **Configuration Reference:**  
  Check out the configuration reference at:  
  [https://huggingface.co/docs/hub/spaces-config-reference](https://huggingface.co/docs/hub/spaces-config-reference)


### Features

- **Perception Agents**: Specialized agents that retrieve and summarize information from distinct datasets:
  - **Case Documents**: Enron case study paper.
  - **SEC Filings**: Legal complaints involving Enron key defendants.
  - **Financial Reports**: Annual Enron 10-K financial reports.

- **Integration Agent**: Combines outputs from perception agents into a concise summary relevant to the user query.

- **Email Retrieval Agent**: Utilizes keyphrase extraction and vector search to retrieve relevant emails from the Enron corpus.

- **Interactive Gradio Interface**: Allows users to input queries, view synthesized summaries, and explore related emails interactively.


### Required Environment Variables in the HuggingFace Gradio Space

- **`OPENAI_API_KEY`**:  
  - This key is required to authenticate with the OpenAI API for generating summaries and extracting keyphrases using GPT-3.5 Turbo.  
  - **How to get it**: Sign up for an OpenAI account at [OpenAI](https://platform.openai.com/signup/) and obtain an API key from the API section of your dashboard.

- **`EMAIL_VECTOR_HOST`**:  
  - Specifies the host address for the email Chroma vector store, which is used to perform vector similarity searches for email retrieval.
  - **How to set it up**: Deploy a Chroma database instance (see build_email_vectorstore.py) and set this environment variable to access that host.
