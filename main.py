import os
import pandas as pd
import logging
from typing import Literal
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langgraph.graph import StateGraph, START, END

# Configure logging to output to stdout so it can be captured in the hosted environment logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Variables
GLOBAL_SENTENCE_MODEL = None
GLOBAL_HUGGINGFACE_MODEL = None
GLOBAL_HUGGINGFACE_TOKENIZER = None

# Load SentenceTransformer model globally
def load_sentence_transformer_model(model_name="all-MiniLM-L6-v2"):
    global GLOBAL_SENTENCE_MODEL
    if GLOBAL_SENTENCE_MODEL is None:
        logger.debug("Loading SentenceTransformer model: %s", model_name)
        GLOBAL_SENTENCE_MODEL = SentenceTransformer(model_name)

# Load HuggingFace model and tokenizer globally (using Flan-T5-Large for this example)
def load_huggingface_model(model_name="google/flan-t5-large"):
    global GLOBAL_HUGGINGFACE_MODEL, GLOBAL_HUGGINGFACE_TOKENIZER
    if GLOBAL_HUGGINGFACE_MODEL is None or GLOBAL_HUGGINGFACE_TOKENIZER is None:
        logger.debug("Loading HuggingFace model: %s", model_name)
        GLOBAL_HUGGINGFACE_MODEL = T5ForConditionalGeneration.from_pretrained(model_name)
        GLOBAL_HUGGINGFACE_TOKENIZER = T5Tokenizer.from_pretrained(model_name)

# Helper Functions
def generate_embeddings(texts):
    if GLOBAL_SENTENCE_MODEL is None:
        raise ValueError("SentenceTransformer model not loaded.")
    logger.debug("Generating embeddings for texts.")
    embeddings = GLOBAL_SENTENCE_MODEL.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def compute_similarities(query_embedding, document_embeddings):
    logger.debug("Computing cosine similarities.")
    return cosine_similarity(query_embedding, document_embeddings).flatten()

# Generate response using the advanced HuggingFace model
def generate_response_with_huggingface(query, top_k_documents):
    if GLOBAL_HUGGINGFACE_MODEL is None or GLOBAL_HUGGINGFACE_TOKENIZER is None:
        raise ValueError("HuggingFace model or tokenizer not loaded.")
    
    logger.debug("Generating response using HuggingFace model.")
    augmented_query = query + " " + " ".join(top_k_documents['chunked_text'].tolist())
    inputs = GLOBAL_HUGGINGFACE_TOKENIZER(augmented_query, return_tensors="pt", truncation=True, max_length=512)
    
    try:
        outputs = GLOBAL_HUGGINGFACE_MODEL.generate(inputs["input_ids"], max_length=300, num_return_sequences=1)
        response = GLOBAL_HUGGINGFACE_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        logger.debug("Generated response: %s", response)
        return response
    except Exception as e:
        logger.error("Error during model generation: %s", e)
        return "An error occurred during response generation."

# Define Perception Agent
class PerceptionAgent:
    def __init__(self, data_df, name):
        self.data_df = data_df
        self.name = name
        logger.debug("Initialized PerceptionAgent: %s", self.name)

    def extract_data(self, query):
        logger.debug("PerceptionAgent (%s) extracting data for query: %s", self.name, query)
        document_texts = self.data_df['chunked_text'].tolist()
        query_embedding = generate_embeddings([query])
        document_embeddings = generate_embeddings(document_texts)

        similarities = compute_similarities(query_embedding, document_embeddings)
        top_k_indices = similarities.argsort()[-5:][::-1]  # Retrieve top 5 similar documents

        top_k_documents = self.data_df.iloc[top_k_indices]
        logger.debug("PerceptionAgent (%s) found top documents: %s", self.name, top_k_documents)
        return top_k_documents

# Define Integration Agent
class IntegrationAgent:
    def synthesize_data(self, perception_results, query):
        logger.debug("IntegrationAgent synthesizing data for query: %s", query)
        response = generate_response_with_huggingface(query, perception_results)
        logger.debug("IntegrationAgent generated response: %s", response)
        return response

# Shared State
class AgentState:
    def __init__(self):
        self.messages = []  # Sequence of messages exchanged
        self.sender = ""  # Tracks last sender
        self.data = {}  # Stores intermediate results

# Initialize Models
load_sentence_transformer_model()  # Load embedding model
load_huggingface_model()  # Load HuggingFace LLM

# Instantiate Perception Agents with DataFrames
sec_df = pd.read_csv("data/sec_docs.csv")
financial_df = pd.read_csv("data/financial_reports.csv")

perception_agent_1 = PerceptionAgent(sec_df, "SEC_Perception")
perception_agent_2 = PerceptionAgent(financial_df, "Financial_Perception")

# Node Functions for the Perception Agents
def perception_node_1(state):
    logger.debug("Executing PerceptionNode1 with initial state: %s", state.__dict__)
    query = state.messages[-1]['content']
    result = perception_agent_1.extract_data(query)
    
    if result.empty:
        logger.warning("PerceptionNode1 returned empty results.")
        state.data['perception_1'] = {
            "message": "No relevant documents found",
            "data": pd.DataFrame(),
            "status": "no_data"
        }
    else:
        state.data['perception_1'] = {
            "message": "Documents found",
            "data": result,
            "status": "data_found"
        }
    
    logger.debug("Updated state after PerceptionNode1: %s", state.__dict__)
    return state

def perception_node_2(state):
    logger.debug("Executing PerceptionNode2 with initial state: %s", state.__dict__)
    query = state.messages[-1]['content']
    result = perception_agent_2.extract_data(query)

    if result.empty:
        logger.warning("PerceptionNode2 returned empty results.")
        state.data['perception_2'] = {
            "message": "No relevant documents found",
            "data": pd.DataFrame(),
            "status": "no_data"
        }
    else:
        state.data['perception_2'] = {
            "message": "Documents found",
            "data": result,
            "status": "data_found"
        }

    logger.debug("Updated state after PerceptionNode2: %s", state.__dict__)
    return state

# Integration Node
def integration_node(state):
    logger.debug("Executing IntegrationNode with initial state: %s", state.__dict__)
    agent = IntegrationAgent()

    valid_results = [
        state.data[key]['data'] for key in state.data.keys()
        if key.startswith('perception') and isinstance(state.data[key]['data'], pd.DataFrame) and not state.data[key]['data'].empty
    ]

    if len(valid_results) == 0:
        logger.warning("IntegrationNode has no valid perception results to integrate.")
        state.data['integration_result'] = {
            "message": "No relevant information found.",
            "status": "no_data"
        }
    else:
        perception_results = pd.concat(valid_results, ignore_index=True)
        query = state.messages[-1]['content']
        response = agent.synthesize_data(perception_results, query)
        state.data['integration_result'] = {
            "message": response,
            "status": "data_integrated"
        }

    logger.debug("Updated state after IntegrationNode: %s", state.__dict__)
    return state

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("UserInterfaceNode", lambda state: state)  # Placeholder to ensure graph has a UI start point
workflow.add_node("PerceptionNode1", perception_node_1)
workflow.add_node("PerceptionNode2", perception_node_2)
workflow.add_node("IntegrationNode", integration_node)

# Start workflow from the UI node
workflow.add_edge(START, "UserInterfaceNode")
workflow.add_edge("UserInterfaceNode", "PerceptionNode1")
workflow.add_edge("PerceptionNode1", "PerceptionNode2")
workflow.add_edge("PerceptionNode2", "IntegrationNode")
workflow.add_edge("IntegrationNode", END)

# Compile the graph
logger.debug("Compiling workflow graph.")
graph = workflow.compile()
logger.debug("Workflow graph compiled successfully.")
