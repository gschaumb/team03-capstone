# Agentic RAG Chatbot with Enron Knowledgebase
<a href="https://github.com/gschaumb/team03-capstone" target="_blank">Link to GitHub Repository</a>

<br><br>

# Table of Contents 
- [Introduction](#introduction)
  - [Project Overview](#project-overview)
  - [Inspiration: Unstructured.io's RAG Solution](#inspiration-unstructuredios-rag-solution)
  - [Capstone Project Hypothesis](#capstone-project-hypothesis)
- [Methods](#methods)
  - [Project Methodology](#project-methodology)
    - [Phase 0: Experimentation and Feasibility Prototyping](#phase-0-experimentation-and-feasibility-prototyping)
    - [Phase 1: Minimum Viable Product (MVP)](#phase-1-minimum-viable-product-mvp)
    - [Phase 2: Enhancements](#phase-2-enhancements)
  - [Agent Design Rationale](#agent-design-rationale)
    - [Case Study Agent](#case-study-agent)
    - [SEC Legal Complaint Documents Agent](#sec-legal-complaint-documents-agent)
    - [Annual Financial Reports Agent](#annual-financial-reports-agent)
    - [Integration Agent](#integration-agent)
    - [Email Agent](#email-agent)
    - [Email Agent Architecture](#email-agent-architecture)
  - [Evaluation Strategy and Results](#evaluation-strategy)
    - [Agentic Response Evaluation](#agentic-response-evaluation)
    - [Email Agent Evaluation](#email-agent-evaluation)
    - [Evaluation and Demo Links](#evaluation-links)
- [Discussion](#discussion)
  - [Reflections](#reflections)
  - [Ethical Considerations](#ethical-considerations)
  - [Conclusion](#conclusion)
- [Appendix](#appendix)
  - [References](#references)
  - [Statement of Work](#statement-of-work)
  - [Data Access Statement](#data-access-statement)
  - [Research Notes](#research-notes)

<br><br>

# Introduction {#introduction}

Agentic retrieval-augmented generation (RAG) with large language models (LLMs) is a cutting-edge research area, combining the power of large language models with the ability to interact with external data and perform actions in real-world contexts. This enables the creation of agents capable of reasoning, planning, and executing tasks to solve complex problems. For instance, agentic RAG systems can perform:

- **Math Problem Solving**: Using agents to autonomously or collaboratively solve math problems, with agents specializing in areas like equation solving, symbolic manipulation, or accessing external mathematical resources.
  
- **Retrieval-Augmented Code Generation**: Employing agents to generate code, with one agent focused on retrieving relevant code snippets from a database, and another responsible for assembling and refining the code based on the retrieved information.
  
- **Decision Making in Text-Based Environments**: Building agents that can navigate and interact within simulated environments using natural language commands, with specialized agents handling tasks like planning, object manipulation, or reasoning about the environment.

The combination of reasoning and acting makes agentic RAG a promising approach for developing more capable and trustworthy AI systems with enhanced understanding and interaction capabilities<sup>[1](#ref-1)</sup>.

## Project Overview {#project-overview}

Our Capstone project builds on our Milestones 1 and 2<sup>[2](#ref-2)</sup>, where we extracted data sources and developed a basic RAG solution for a **customer support use case**. In this Capstone project:

- **Public legal, financial, and historical documents related to the Enron case** serve as a proxy for the knowledgebase. Securities and Exchange Commission (SEC) Legal Complaints and an Enron case study provide a rich, unstructured data history simulating fact-based manuals or implementation guides. And the annual financial reports provide an even less structured set of possibly relevant details, simulating incorporation of noisy historical information or data logs.
- **The Enron email corpus** represents a range of employee and/or customer emails.

Ideally, we would have liked to have used a product or service company's corporate technical manuals, implementation guides, support FAQs, and related customer support emails as our data sources. Given the difficulty in obtaining such materials, the Enron data has proven a valuable proxy due to its broad content coverage across various document types and email exchanges.

Our goal was to investigate whether agentic RAG improved results over basic RAG using specialized agents and workflows, such as:

- **Chaining Agents**: Chaining agents can be used to break up the task into a series of smaller steps. Each agent in the chain would be responsible for a specific part of the overall task, like retrieval, summarization, or question answering. An agent that specializes in legal documents could, for example, retrieve the appropriate sections of a document based on what the user asks. The information retrieved by the first agent could then be passed to a summarization agent to be condensed into smaller, more concise insights.

- **Specializing Agent Interaction with Data**: Each agent can be specialized for a specific data source so that the agent becomes an expert in understanding and retrieving information from that source. Having specialized agents could really improve the relevance and accuracy of the information that they retrieve. For instance, an agent that was specifically trained on financial reports would be better at answering questions about how Enron was performing financially.

- **LLM as an Intermediate Summary Processor**: LLMs can be used as a central processing unit to take the summaries from different specialized agents and synthesize them into a single response. This approach would help the chatbot to present a clear and consistent picture of the Enron case, even though the case is very complex.

## Inspiration: Unstructured.io's RAG Solution {#inspiration-unstructuredios-rag-solution}

One example of an effective support use case RAG system is Unstructured.io's chatbot, which operates on their documentation page and Slack channel. We experimented with document extraction using Unstructured's products in Milestone 2, and our interactions with this chatbot demonstrated its high capacity for providing relevant information.

Visit their documentation page here:  
<a href="https://docs.unstructured.io/welcome" target="_blank">Unstructured.io Documentation</a>

Their site linked above includes the chatbot in the lower right corner, which we speculate may also support their internal employees and automated email responses. This tool provides detailed and accurate answers, representing the type of capability we aimed to emulate in our Enron project.

## Capstone Project Hypothesis {#capstone-project-hypothesis}

In Milestone 2, we performed initial testing with a basic RAG setup (see Figure 1). Our Capstone project expands upon this with the goal of testing whether agentic RAG improves results with the Enron knowledgebase when compared with basic RAG or unaugmented LLM responses, all using the GPT-3.5 Turbo model. Additionally, we created an agent to search via API through the extensive Enron email corpus, located on a cloud database service, and retrieve emails related to the user's query.

Our primary hypothesis is that utilizing one agent per data source type will enhance the chatbot's overall response quality, particularly for compound questions. Each agent is equipped with customized retrieval parameters, in an attempt to optimize for its respective data source. See Figure 2 for a visual of the anticipated benefits.

<br>

**Figure 1**  
*Basic RAG Architecture:*  

<br>

![Figure 1](basic_RAG.png)

<br><br>

**Figure 2**  
*Advantages of Specialized Agentic RAG Configuration:*  

<br>

![Figure 2](high_level_benefits.png)
<br>

# Methods {#methods}

**Project Methodology** explains how we attempted to solve the problem and justifies our methodological approach, while **Evaluation Strategy** defines what a successful outcome would be.

## Project Methodology {#project-methodology}

### Phase 0: Experimentation and Feasibility Prototyping {#phase-0-experimentation-and-feasibility-prototyping}
- **High-Level Design - Requirements:** Researched agentic patterns<sup>[3](#ref-3)</sup>.
- **High-Level Design - Implementation:** Experimented with agentic libraries and did prototype feasibility coding.

### Phase 1: Minimum Viable Product (MVP) {#phase-1-minimum-viable-product-mvp}
- **Detailed Design - Implementation:** Decided to code agents with python classes rather than using an agentic library. Defined our agentic pattern flow and email retrieval (see Figure 3).
- **Detailed Design - Evaluation:** Defined evaluation criteria after agent design was defined.

### Phase 2: Enhancements {#phase-2-enhancements}
- **Implemented Enhancements:** Integrated the Email Agent.
- **Updated Evaluation Criteria:** Email Agent evaluation.

<br>

## Agent Design Rationale {#agent-design-rationale}
Our code uses the term "Perception Agents" for our data source agents, reflecting their ability to both retrieve chunks and extract specific useful information relevant to the query. Each agent's retrieval pipeline and summarization strategy leverages specific parameters and prompts tailored to the nature of the documents it handles, aiding our goal of summaries that are both precise and contextually relevant.
<br>

**Figure 3**
*Agent Design and Workflow:*  
<br>

![Figure 3](agent_flow.svg)

<br>

### Case Study Agent (PerceptionAgent1) {#case-study-agent}

**Retrieval Pipeline**:
- **Threshold of 0**: Includes all documents with cosine similarity greater than zero to ensure comprehensive coverage.
- **Top 2 Documents**: Focuses on the most relevant documents to maintain precision.
- **Context Window of 1**: Adds plus one and minus one adjacent documents for richer context and understanding.

**Summarization Strategy**:
- **Concise Summarization**: Summarizes in two sentences focusing on the most important details.

**Objective**:
- Designed to efficiently distill a dense, chronologically consistent case summary of Enron, providing key insights while attempting to ensure no detail is overlooked. The initial broad threshold ensures no potentially relevant document is missed, while the top-k selection focuses on the best matches. The context window adds adjacent documents, which can be helpful for understanding necessary context or additional insights.

<br>

### SEC Legal Complaint Documents Agent (PerceptionAgent2) {#sec-legal-complaint-documents-agent}

**Retrieval Pipeline**:
- **Threshold of 0.3**: Filters for moderate to high relevance, reducing noise.
- **Top 2 Documents**: Ensures focus on the most pertinent content.
- **Context Window of 1**: Includes immediate neighbors to capture essential legal contexts.

**Summarization Strategy**:
- **Legal Focus**: Summarizes critical legal elements like relevant people and dates in two sentences.

**Objective**:
- Suitable for well-organized legal data where accurate document retrieval is seen as helpful for precision, compliance and regulatory affairs. The higher threshold helps to ensure that only documents significantly related to the query are retrieved, reducing noise from less relevant documents. The context window, although small, is useful with legal data where neighboring paragraphs or sections often contain pertinent arguments or references.

<br>

### Annual Financial Reports Agent (PerceptionAgent3) {#annual-financial-reports-agent}

**Retrieval Pipeline**:
- **Threshold of 0**: Captures all potentially relevant financial reports.
- **Top 1 Document**: Concentrates on the single most pertinent document for detailed analysis.
- **Context Window of 2**: Expands context significantly with the two before and after to capture related financial data broad context.

**Summarization Strategy**:
- **Financial Analysis**: Integrates financial insights into a concise two-sentence summary focusing on key financial facts.

**Objective**:
- Succinct summaries that highlight key information and financials from the relevant document chunks. Suitable for financial reports where key results and statements can be scattered across documents, but where the primary focus should be on the most relevant document. The broader context window helps in gathering insights where the desired financial information is informed by adjacent chunks.

<br>

### Integration Agent {#integration-agent}

**Summarization Strategy**:
- **Comprehensive Integration**: Combines summaries from all agents, using a prompt that directs the LLM to answer the query with a response synthesized from the relevant provided data. Agent explicitly asks the LLM to ignore details that aren't clearly linked to the query subject, in an attempt to counteract instances of over-retrieval and noise.

**Objective**:
- To provide a comprehensive and nuanced overview of the diverse data, attempting to provide users with an answer to their query that aids in complex decision-making processes across various document types.

<br>

### Email Agent {#email-agent}

**Objective**:
- The Email Agent supplements the response returned to the User Interface by the Integration Agent by returning emails relevant to the query (see Figure 4). It leverages the summaries from each Perception Agent to do this.

**Implementation**:
- **Vector Database**: We use Chroma DB to store and query the email embeddings, which are generated from a corpus of 500k emails using Huggingface's all-MiniLM-l6-v2 model. The vector database is hosted on an AWS instance.
- **Keyphrase Extraction**: When a query is received, each agent generates its own summary (the "intermediate response"). Using GPT-3.5-Turbo, we extract keyphrases from these summaries as a string of comma-separated phrases.
- **Retrieve Relevant Emails**:
  - The extracted keyphrases are converted into embeddings using the same all-MiniLM-l6-v2 model.
  - Using Cosine Similarity, we fetch the top 20 emails relevant to the query for each agent.
  - Finally, we merge and return a unique set of emails as the final output.

<br>

### Email Agent Architecture {#email-agent-architecture}
**Figure 4**
*Email Agent:*  
![Figure 4](VectorStore.png)

<br><br>

# Evaluation Strategy and Results {#evaluation-strategy}


### Agentic Response Evaluation {#agentic-response-evaluation}
- We used a test set of 100 queries (50 "Single Topic" and 50 "Compound" queries). The queries were created using generative AI tools to extract questions related to our source material and then curated by human review. We divided them into "Single Topic" vs. "Compound" queries with a secondary goal of determining if the Agentic RAG performed better on Compound questions.
- To assess our Agentic RAG we used GPT-4o's responses as the Ground Truth. This choice was enabled by the Enron dataset and case's prominence as a key training source for most large language models. Through a high level review of answers, we felt GPT-4o responses provided a suitable Ground Truth that exceeded our GPT-3.5 Turbo based system responses in terms of accuracy and completeness.
- The Ground Truth was evaluated against our three GPT-3.5 Turbo based systems:
  - **Agentic RAG**: Our custom-built RAG developed for the Capstone project.
  - **Base RAG**: A foundational RAG created during Milestone 2 as a proof of concept.
  - **Unaugmented RAG**: Responses generated directly by GPT-3.5 Turbo.
  <br>
  
- The response was evaluated in two ways:
  - **BERTScore**
    
    - Unlike traditional metrics like ROUGE and BLEU that rely on n-gram overlap, BERTScore evaluates semantic similarity by leveraging contextual embeddings. We utilized the microsoft/deberta-xlarge-mnli model to compare the Ground Truth response with the three systems. BERTScore provides three metrics: Precision, Recall, and F1 Score, capturing how well the generated response aligns with the Ground Truth<sup>[4](#ref-4)</sup>.
  - **Entity Coverage Score**
    
    - We wanted to look at the evaluation from another angle. To address the limitations of BERTScore due to its sensitivity to response length, we introduced the Entity Coverage Score. This metric is based on the principle that a good summary should cover key entities, regardless of its length. The score is computed as the ratio of entities in the generated response to those in the Ground Truth. Entities—such as names, locations, dates, times, quantities, and currencies were identified using spaCy's Named Entity Recognition (NER) package. This method emphasizes coverage over verbosity.
<br>

- **Results**

  - **BERTScore**
    - The table below presents the mean **F1**, **Precision**, and **Recall** scores from the BERTScore algorithm for each system compared to the Ground Truth.
    - To evaluate the performance of Agentic RAG relative to other systems, we conducted paired t-tests on the F1, Precision, and Recall scores to assess whether the differences between the systems were statistically significant. A p-value less than 0.05 was used as the threshold for significance, indicating strong evidence against the null hypothesis of no difference between methods.

    - <table>
        <tr>
          <th>Metric</th>
          <th>Agentic RAG</th>
          <th>Base RAG</th>
          <th>Unaugmented Response</th>
        </tr>
        <tr>
          <th>F1 Score</th>
          <td>
            <table>
              <tr><th>Score</th><th>P-value</th></tr>
              <tr><td>0.624</td><td>NA</td></tr>
            </table>
          </td>
          <td>
            <table>
              <tr><th>Score</th><th>P-value</th></tr>
              <tr><td>0.650</td><td>0.0001</td></tr>
            </table>
          </td>
          <td>
            <table>
              <tr><th>Score</th><th>P-value</th></tr>
              <tr><td>0.700</td><td>1.03e-24</td></tr>
            </table>
          </td>
        </tr>
        <tr>
          <th>Precision</th>
          <td>
            <table>
              <tr><th>Score</th><th>P-value</th></tr>
              <tr><td>0.66</td><td>NA</td></tr>
            </table>
          </td>
          <td>
            <table>
              <tr><th>Score</th><th>P-value</th></tr>
              <tr><td>0.68</td><td>0.003</td></tr>
            </table>
          </td>
          <td>
            <table>
              <tr><th>Score</th><th>P-value</th></tr>
              <tr><td>0.73</td><td>2.02e-18</td></tr>
            </table>
          </td>
        </tr>
        <tr>
          <th>Recall</th>
          <td>
            <table>
              <tr><th>Score</th><th>P-value</th></tr>
              <tr><td>0.59</td><td>NA</td></tr>
            </table>
          </td>
          <td>
            <table>
              <tr><th>Score</th><th>P-value</th></tr>
              <tr><td>0.62</td><td>3.61e-05</td></tr>
            </table>
          </td>
          <td>
            <table>
              <tr><th>Score</th><th>P-value</th></tr>
              <tr><td>0.67</td><td>3.60e-28</td></tr>
            </table>
          </td>
        </tr>
      </table>

    - There is a statistically significant difference in F1, Precision, and Recall between the Agentic RAG and both the Base RAG and Unaugmented responses. Specifically, the Unaugmented system had the highest performance, followed by Base RAG, with Agentic RAG consistently showing the lowest scores across all metrics. The F1 scores show that Unaugmented Response is significantly better than both Agentic and Base RAG, which aligns with the individual precision and recall findings. This outcome challenges the expectation that LLM prompt engineering and combining specialized agent responses would enhance response quality.

    - When we analyzed the Single Topic and Compound questions separately, the performance of Agentic RAG was closer to that of Unaugmented Response on Compound questions, suggesting a slightly smaller gap. However, the overall ranking across the response types remained the same, with Unaugmented Response performing the best, followed by Base RAG, and then Agentic RAG.


  - **Entity Coverage Score**
    - Additionally, to ensure that we're evaluating the RAG on its merits and not the length (as GPT 4o does provide with additional context, and therefore, has much more text in the response), we also calculated the ratio of the number of entities in each system to the number of entities in the Ground Truth response as the Entity Coverage Score. On this metric, the Agentic RAG performed better than the other two systems.
    - <table>
        <tr>
        <th>Agentic RAG</th>
        <th>Base RAG</th>
          <th>Unaugmented Response</th>
          </tr>
        <tr>
          <td>0.73</td>
          <td>0.56</td>
          <td>0.53</td>
        </tr>
      </table>

<br>

### Email Agent Evaluation {#email-agent-evaluation}

We attempted to evaluate the email agent responses manually in two ways.

- **Query based evaluation**: Using the same queries as the Agentic Response Evaluation, we checked whether each returned email was relevant to the query. Relevance was determined by either the existence of the key phrases in the intermediate summaries, or contextual similarity to those keyphrases.
- **Keyphrase based evaluation**: For a single keyphrase, we assessed the top 20 emails retrieved for relevance.


Both methods faced the same challenge: evaluating cosine similarity scores is inherently subjective and qualitative.

- In the first method, broad queries allowed for nearly any result to be justified as relevant or irrelevant, making consistent evaluation difficult.
- In the second, more focused method, the same issue persisted. For example, the keyphrase 'Fraudulent Transactions' retrieved emails containing personal opinions and news articles about the Enron scandal, which were contextually relevant but not directly linked to fraudulent activity.
- Another challenge was duplicate or near-duplicate emails from forwards and long email chains, leading to redundancy in the results.


Despite these limitations, manual testing showed that the responses generally maintained contextual relevance or textual overlap with the queries.

<br>

### Evaluation and Demo Links {#evaluation-links}


<a href="https://docs.google.com/spreadsheets/d/e/2PACX-1vSazYHLTkUGFPhN0KqDidJPrdtEYcs3apmemOkPjzgAiDhl2EkmxVk55jNxtcmBArZyzzSyyAPiA6nU/pubhtml?widget=true&amp;headers=false" target="_blank">Link to Evaluation Sheet</a>

<a href="https://huggingface.co/spaces/gschaumb/team03-capstone" target="_blank">Link to Capstone Agentic RAG Demo Application</a>

<a href="https://huggingface.co/spaces/gschaumb/M2-Team01-SS24-RAG-demo" target="_blank">Link to Milestone 2 Basic RAG Demo Application</a>

<br><br>

# Discussion {#discussion}

## Reflections {#reflections}

- This project aimed to evaluate the performance of Agentic RAGs on raw corpora. However, we encountered a unique constraint: most large language models are already well-versed in the Enron case's facts and issues. As a result, we could not conclusively determine that Agentic RAGs outperform LLMs on publicly available knowledgebases. Our inclusion of the Entity Coverage Score metric did indicate that the Agentic RAG effectively balances content with conciseness in its responses.
- The fact that we (for project efficiency's sake) used GPT-4o responses as our Ground Truth may have contributed to the Unaugmented Response being rated highest on BERTScore, as the common model training between GPT-4o and GPT-3.5-Turbo can be considered a form of data leakage potentially favoring the non-RAG responses.
- In real-world applications this approach would be more impactful when applied to private datasets that are not exposed to OpenAI's GPTs or other LLMs. In such cases Agentic RAGs could outperform unaugmented or base RAGs by leveraging specialized agents for distinct data sources, assuming other factors (such as data availability) remain constant.


**Next Steps**:

- With additional time, we would have focused more extensively on prompt engineering. This could include experimenting with different structures, tones, and techniques to optimize how prompts influence the outcomes of the system.
- Another key area for improvement lies in experimenting further with chunk sizes and text preprocessing. Adjusting chunk sizes directly impacts the granularity of text representation and the balance between precision and recall. 
- We would also like to study how these changes affect other critical parameters such as the similarity threshold, the choice of 'k' in nearest neighbor searches, and overall system accuracy.
- We might have also experimented with an algorithm other than cosine similarity for finding the closest text.
- One specific improvement we aimed for was using techniques like Locality Sensitive Hashing (LSH) to identify forwards or replies that contain repeated email content and perhaps make a decision to retain the longest email whilst eliminating all others to remove redundancy.
- We also found that the Chroma DB occupied a large amount of space and would've liked to experiment with alternatives like FAISS/Elasticsearch DB/traditional DBs like PostgreSQL.
- We might've wanted to introduce explicit guardrails to ensure users only asked questions related to Enron.


## Ethical Considerations {#ethical-considerations}


- As the Enron case is widely publicized, we attempted to refrain from picking up studies, papers and other material that would reflect opinions of any individuals or group of individuals, so as to not perpetuate any bias. The exception to this is that the SEC Legal Complaint data sources, which are rich in details regarding Enron, also represent only the SEC's allegations and do not include the defendants' perspectives.
- On the user interface, we have intentionally displayed the intermediate responses generated by the agents to promote transparency and provide full visibility into the system's decision-making process.
- One ethical consideration that is particularly important for chatbot assistants is recognizing that the probabilistic nature of Large Language Models may mean that responses are not always accurate. Examples of this in recent news include companies being held accountable for erroneous chatbot commitments<sup>[5](#ref-4)</sup>. This is one of the major hurdles in deploying LLM based chatbots today and should be seriously considered in production system development. 


## Conclusion {#conclusion}


Using the Enron corpus as a proxy knowledgebase posed unique challenges since modern LLMs are already trained on it, but it also reinforced the promise of this approach for real-world applications involving private, unseen data sources. This project leaves us optimistic about the future of Agentic RAG systems. With further design variations, prompt engineering, external tool access, and adaptation to proprietary datasets, systems similar to our Capstone have the potential to improve responses by AI Agents, and subsequently decision making. 

<br><br><br>


# Appendix {#appendix}

<br>

## References {#references}

1. [La Cava, L., & Tagarelli, A. (2024). Safeguarding Decentralized Social Media: LLM Agents for Automating Community Rule Compliance. *arXiv*.](#ref-1) <a id="ref-1"></a>https://arxiv.org/abs/2409.08963
2. [Bodas, A., Goldhardt, N., & Schaumburg, G. (n.d.). *Milestone 2*. GitHub.](#ref-2) <a id="ref-2"></a>https://github.com/bodasa-umich/Milestone2
3. [Doe, J. (2023, September 22). Designing Cognitive Architectures: Agentic Workflow Patterns from Scratch. *Medium*.](#ref-3) <a id="ref-3"></a>https://medium.com/google-cloud/designing-cognitive-architectures-agentic-workflow-patterns-from-scratch-63baa74c54bc
4. [Sahu, S. (2024, October 30). LLM Evaluation Metrics. *Medium*.](#ref-4) <a id="ref-4"></a>https://ssahuupgrad-93226.medium.com/llm-evaluation-metrics-the-ultimate-llm-evaluation-guide-e9bc94dba1e1
5. [Cecco, L. (2024, February 16). Air Canada ordered to pay customer who was misled by airline’s chatbot. *The Guardian*.](#ref-5) <a id="ref-5"></a>https://www.theguardian.com/world/2024/feb/16/air-canada-chatbot-lawsuit

<br>

## Statement of Work {#statement-of-work}

**Statement on Use of Generative AI:** Tools such as ChatGPT, Gemini, and Copilot were used in our project for experimentation on tasks such as evaluation query creation, code consultation, as well as searching on approaches and concepts.

**Team Member Scope** 
- **Anandita:** Email Agent Architecture, Email Agent Coding, Evaluation Approach and Scoring, Evaluation Strategy and Results Report Sections Drafts, Project Video Segments and Video Integration
- **Gary:** HuggingFace Setup, Perception and Integration Agent Coding, Introduction and Methods Drafts, Evaluation Test Query and Responses Generation, Demo Video Segments

<br>

## Data Access Statement {#data-access-statement}

The data used in this project are publicly available from the following sources:

### Data Sources

- [Enron Email Dataset](https://enrondata.readthedocs.io/en/latest/data/calo-enron-email-dataset/)
- [SEC Financial Filings](https://www.sec.gov/search-filings)
- [Enron: The Good, The Bad, The Lessons (ResearchGate)](https://www.researchgate.net/publication/267715799_Enron_The_Good_The_Bad_The_Lessons)
- [SEC Legal Complaints](https://www.justice.gov/archive/index-enron.html)


<br>

## Research Notes {#research-notes}

Summaries of relevant Phase 0, LLM agent research we conducted appear below.

#### Agentic Approaches

- **Workflow Management Systems**: Systems like StateFlow can be used to define the steps that agents need to follow to complete their task. This could be helpful for the Enron chatbot, because the information about the case comes from so many different sources. For example, the workflow could define a state for "retrieving legal documents" and a state for "summarizing financial reports."

- **Dynamic Group Chat**: Systems like AutoGen can be used to simulate a conversation between different agents. This type of system could be more flexible and engaging for users. For example, the Enron chatbot could have a "legal expert" agent and a "financial analyst" agent that users could interact with directly.

- **Prompt Engineering**: Different prompting strategies can be used to influence how LLMs reason about information. "Chain-of-thought" prompting and "self-consistency" prompting are two promising strategies. These strategies may be evaluated to see how well they work for the Enron chatbot.

- **Decoding Methods**: By default, LLMs use a "greedy decoding" approach to generate responses. However, LLMs can also generate responses using different approaches. We may experiment with alternative decoding methods, such as "top-k sampling," to see if these methods lead to better results for the Enron chatbot.


#### Agentic LLM Project Design Guidelines Which May Serve As Our Goals

Projects that benefit from agentic LLM structures often involve tasks that can be broken down into distinct sub-tasks, each potentially requiring different capabilities or data sources. This structure becomes particularly useful when these sub-tasks need to be executed in a specific order or require a dynamic interaction based on the outcome of previous steps. Here are some guidelines for designing such projects:


#### Identify the Core Sub-Tasks:

- **Different Data Types**: If your project requires retrieving and processing information from different data types like text, code, or structured data, consider using different agents specialized for each type. For instance, one agent could be responsible for parsing code, while another handles database queries.
  
- **Different Knowledge Domains**: When dealing with tasks that span multiple knowledge domains, using separate agents with expertise in each domain can be beneficial. For example, in a medical diagnosis project, one agent could focus on analyzing symptoms, while another specializes in interpreting medical literature.
  
- **Different Tasks within a Workflow**: Break down the overall project workflow into distinct tasks, each handled by a specific agent. This approach proves valuable when tasks require different capabilities. For instance, one agent could be responsible for generating code, another for checking its safety, and a third for executing it.


#### Design the Interaction Flow:

- **Static vs. Dynamic Conversations**: Determine whether the interaction between agents should follow a predefined, static order, or if it requires flexibility and adaptation based on intermediate results. Static patterns are simpler to implement, but dynamic patterns offer greater flexibility in complex scenarios.
  
- **Centralized vs. Decentralized Control**: Decide whether a central agent will coordinate the workflow, dispatching tasks to specialized agents and processing their results, or if agents will communicate directly in a more decentralized manner. Centralized control can simplify coordination, while decentralized control might be more robust and scalable.
  
- **Human Involvement**: Determine the role of humans in the loop. This could range from providing initial instructions, to validating intermediate outputs, to taking over control in case of errors. Striking the right balance between human oversight and agent autonomy is crucial for trust and reliability.


#### Agent Capabilities and Design:

- **LLM Configuration**: Carefully choose the appropriate LLM for each agent, considering factors like model size, capabilities, and cost-effectiveness. For instance, a smaller LLM might suffice for straightforward tasks, while a larger, more capable model could be necessary for complex reasoning or code generation.
  
- **Tool Integration**: Incorporate external tools to augment agent capabilities. This could involve accessing databases, executing code, or leveraging APIs for specific functionalities.
  
- **Memory Management**: Design mechanisms for agents to access and manage shared or individual memories, allowing them to retain relevant information from previous interactions or access external knowledgebases.

#### Evaluation and Refinement:

- **Define Clear Metrics**: Establish clear metrics to evaluate the performance of your agentic system, aligning them with the specific goals of your project. These metrics could include task completion rate, accuracy, cost, latency, or human effort saved.
  
- **Iterative Refinement**: Continuously monitor, analyze, and refine your agentic system based on its performance and identify areas for improvement, such as agent capabilities, interaction patterns, or tool integration.


#### Key Considerations:

- **Robustness and Error Handling**: Agentic LLM systems can be complex and prone to errors. Implement mechanisms to handle unexpected situations, such as agent failures, ambiguous instructions, or conflicting information.
  
- **Ethical Implications**: Consider the ethical implications of your project, particularly when agents interact with real-world systems or make decisions that could impact users. Ensure fairness, transparency, and accountability in agent behavior.
