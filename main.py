import os
import pandas as pd
import logging
from typing import Literal
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langgraph.graph import StateGraph, START, END

# Configure logging to output to stdout for hosted environment logs
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

# Define UI Agent
class UserInterfaceAgent:
    def handle_user_input(self, user_input, state):
        logger.debug("Handling user input: %s", user_input)
        state.messages.append({'sender': 'User', 'content': user_input})
        return state

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
    logger.debug("Executing PerceptionNode1 with state: %s", state.__dict__)
    query = state.messages[-1]['content']
    result = perception_agent_1.extract_data(query)
    if result.empty:
        logger.warning("PerceptionNode1 returned empty results.")
        state.data['perception_1'] = "No relevant documents found."
    else:
        state.data['perception_1'] = result
    logger.debug("Updated state after PerceptionNode1: %s", state.__dict__)
    return state

def perception_node_2(state):
    logger.debug("Executing PerceptionNode2 with state: %s", state.__dict__)
    query = state.messages[-1]['content']
    result = perception_agent_2.extract_data(query)
    if result.empty:
        logger.warning("PerceptionNode2 returned empty results.")
        state.data['perception_2'] = "No relevant documents found."
    else:
        state.data['perception_2'] = result
    logger.debug("Updated state after PerceptionNode2: %s", state.__dict__)
    return state

# Integration Node
def integration_node(state):
    logger.debug("Executing IntegrationNode with state: %s", state.__dict__)
    agent = IntegrationAgent()
    perception_results = pd.concat(
        [state.data[key] for key in state.data.keys() if key.startswith('perception') and not isinstance(state.data[key], str)],
        ignore_index=True
    )
    if perception_results.empty:
        logger.warning("IntegrationNode has no valid perception results to integrate.")
        state.data['integration_result'] = "No relevant information found."
    else:
        query = state.messages[-1]['content']
        response = agent.synthesize_data(perception_results, query)
        state.data['integration_result'] = response
    logger.debug("Updated state after IntegrationNode: %s", state.__dict__)
    return state

# UI Node
def ui_node(state):
    logger.debug("Executing UserInterfaceNode with state: %s", state.__dict__)
    agent = UserInterfaceAgent()
    updated_state = agent.handle_user_input(state.messages[-1]['content'], state)
    logger.debug("Updated state after UserInterfaceNode: %s", updated_state.__dict__)
    return updated_state

# Routing Logic
def router(state) -> Literal["call_tool", "__end__", "to_integration", "to_perception_1"]:
    last_message = state.messages[-1]
    logger.debug("Routing based on last message: %s", last_message)

    if "clarify" in last_message['content']:
        return "to_perception_1"
    elif "FINAL ANSWER" in last_message['content']:
        return "__end__"
    elif "tool_needed" in last_message['content']:
        return "call_tool"
    else:
        return "to_integration"

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("UserInterfaceNode", ui_node)
workflow.add_node("PerceptionNode1", perception_node_1)
workflow.add_node("PerceptionNode2", perception_node_2)
workflow.add_node("IntegrationNode", integration_node)

workflow.add_conditional_edges("UserInterfaceNode", router, {
    "to_perception_1": "PerceptionNode1",
    "to_perception_2": "PerceptionNode2",
    "__end__": END,
})

workflow.add_conditional_edges("PerceptionNode1", router, {
    "to_integration": "IntegrationNode",
})

workflow.add_conditional_edges("PerceptionNode2", router, {
    "to_integration": "IntegrationNode",
})

workflow.add_conditional_edges("IntegrationNode", router, {
    "return_ui": "UserInterfaceNode",
    "__end__": END,
})

# Compile the graph
graph = workflow.compile()
