import os
import pandas as pd
import logging
from typing import TypedDict, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BloomForCausalLM, BloomTokenizerFast
from langgraph.graph import StateGraph, START, END
import pickle

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Variables
GLOBAL_SENTENCE_MODEL = None
GLOBAL_HUGGINGFACE_MODEL = None
GLOBAL_HUGGINGFACE_TOKENIZER = None

# Paths for Embeddings
SEC_EMBEDDINGS_PATH = "/data/sec_embeddings.pkl"
FINANCIAL_EMBEDDINGS_PATH = "/data/financial_embeddings.pkl"

# TypedDicts for Perception and Integration Result
class PerceptionResult(TypedDict, total=False):
    status: str
    data: Optional[pd.DataFrame]

class IntegrationResult(TypedDict, total=False):
    status: str
    message: Optional[str]

# Define the State as a TypedDict with clear structure
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    sender: str
    perception_1: PerceptionResult
    perception_2: PerceptionResult
    integration_result: IntegrationResult

# Load SentenceTransformer model globally
def load_sentence_transformer_model(model_name="all-MiniLM-L6-v2"):
    global GLOBAL_SENTENCE_MODEL
    if GLOBAL_SENTENCE_MODEL is None:
        logger.debug("Loading SentenceTransformer model: %s", model_name)
        GLOBAL_SENTENCE_MODEL = SentenceTransformer(model_name)

# Load HuggingFace model and tokenizer globally
def load_huggingface_model(model_name="bigscience/bloomz-7b1"):
    global GLOBAL_HUGGINGFACE_MODEL, GLOBAL_HUGGINGFACE_TOKENIZER
    if GLOBAL_HUGGINGFACE_MODEL is None or GLOBAL_HUGGINGFACE_TOKENIZER is None:
        logger.debug("Loading HuggingFace model: %s", model_name)
        # Load Bloom model and tokenizer
        GLOBAL_HUGGINGFACE_MODEL = BloomForCausalLM.from_pretrained(model_name)
        GLOBAL_HUGGINGFACE_TOKENIZER = BloomTokenizerFast.from_pretrained(model_name)

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

# Define Perception Agent
class PerceptionAgent:
    def __init__(self, data_df, name, embeddings_path):
        self.data_df = data_df
        self.name = name
        self.embeddings_path = embeddings_path
        self.document_embeddings = self.load_or_generate_embeddings()
        logger.debug("Initialized PerceptionAgent: %s", self.name)

    def load_or_generate_embeddings(self):
        if os.path.exists(self.embeddings_path):
            logger.debug(f"Loading precomputed embeddings for {self.name} from {self.embeddings_path}")
            with open(self.embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
        else:
            logger.debug(f"Generating and storing embeddings for {self.name}")
            embeddings = generate_embeddings(self.data_df['chunked_text'].tolist())
            with open(self.embeddings_path, "wb") as f:
                pickle.dump(embeddings, f)
        return embeddings

    def extract_data(self, query):
        logger.debug("PerceptionAgent (%s) extracting data for query: %s", self.name, query)

        # Generate query embedding and log it
        query_embedding = generate_embeddings([query])
        logger.debug("Generated query embedding: %s", query_embedding)

        # Compute similarities between the query and document embeddings
        similarities = compute_similarities(query_embedding, self.document_embeddings)
        logger.debug("Computed similarities: %s", similarities)

        # Retrieve top 5 similar documents
        top_k_indices = similarities.argsort()[-5:][::-1]
        logger.debug("Top 5 similar document indices: %s", top_k_indices)

        # Extract top documents
        top_k_documents = self.data_df.iloc[top_k_indices]
        logger.debug("PerceptionAgent (%s) found top documents: %s", self.name, top_k_documents)

        return top_k_documents

# Define Integration Agent
class IntegrationAgent:
    def synthesize_data(self, perception_results, query):
        if GLOBAL_HUGGINGFACE_MODEL is None or GLOBAL_HUGGINGFACE_TOKENIZER is None:
            raise ValueError("HuggingFace model or tokenizer not loaded.")
        
        logger.debug("IntegrationAgent synthesizing data for query: %s", query)
        
        # Combine all top documents' text for context
        augmented_query = query + " " + " ".join(perception_results['chunked_text'].tolist())
        
        # Tokenize the input
        inputs = GLOBAL_HUGGINGFACE_TOKENIZER(augmented_query, return_tensors="pt", padding="longest", truncation=True, max_length=512)
        
        try:
            # Generate the model's response (text generation step)
            GLOBAL_HUGGINGFACE_MODEL.eval()
            outputs = GLOBAL_HUGGINGFACE_MODEL.generate(inputs["input_ids"], max_length=300, num_return_sequences=1)
            response = GLOBAL_HUGGINGFACE_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            logger.debug("IntegrationAgent generated response: %s", response)
            return response
        except Exception as e:
            logger.error("Error during response generation: %s", e)
            return "An error occurred during response generation."

# Perception Nodes
def perception_node_1(state: AgentState) -> AgentState:
    logger.debug("Executing PerceptionNode1 with initial state: %s", state)
    
    if not state['messages']:
        logger.error("No messages in state.")
        state['perception_1'] = {"status": "no_data", "data": pd.DataFrame()}
        return state
    
    query = state['messages'][-1]['content']
    result = perception_agent_1.extract_data(query)
    
    if result.empty:
        state['perception_1'] = {"status": "no_data", "data": pd.DataFrame()}
    else:
        state['perception_1'] = {"status": "data_found", "data": result}
    
    return state

def perception_node_2(state: AgentState) -> AgentState:
    logger.debug("Executing PerceptionNode2 with initial state: %s", state)
    
    if not state['messages']:
        logger.error("No messages in state.")
        state['perception_2'] = {"status": "no_data", "data": pd.DataFrame()}
        return state

    query = state['messages'][-1]['content']
    result = perception_agent_2.extract_data(query)

    if result.empty:
        state['perception_2'] = {"status": "no_data", "data": pd.DataFrame()}
    else:
        state['perception_2'] = {"status": "data_found", "data": result}

    return state

# Integration Node
def integration_node(state: AgentState) -> AgentState:
    logger.debug("Executing IntegrationNode with initial state: %s", state)
    agent = IntegrationAgent()

    valid_results = [
        state[key]['data'] for key in ['perception_1', 'perception_2']
        if state[key]['status'] == 'data_found' and isinstance(state[key]['data'], pd.DataFrame) and not state[key]['data'].empty
    ]

    if len(valid_results) == 0:
        state['integration_result'] = {"status": "no_data", "message": "No relevant information found."}
    else:
        perception_results = pd.concat(valid_results, ignore_index=True)
        query = state['messages'][-1]['content']
        response = agent.synthesize_data(perception_results, query)
        state['integration_result'] = {"status": "data_integrated", "message": response}

    return state

# Initialize Models
load_sentence_transformer_model()  # Load embedding model
load_huggingface_model()  # Load HuggingFace LLM

# Instantiate Perception Agents with DataFrames
sec_df = pd.read_csv("data/sec_docs.csv")
financial_df = pd.read_csv("data/financial_reports.csv")

perception_agent_1 = PerceptionAgent(sec_df, "SEC_Perception", SEC_EMBEDDINGS_PATH)
perception_agent_2 = PerceptionAgent(financial_df, "Financial_Perception", FINANCIAL_EMBEDDINGS_PATH)

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("UserInterfaceNode", lambda state: state)  # Placeholder to ensure graph has a UI start point
workflow.add_node("PerceptionNode1", lambda state: perception_node_1(state))
workflow.add_node("PerceptionNode2", lambda state: perception_node_2(state))
workflow.add_node("IntegrationNode", lambda state: integration_node(state))

# Set entry point and add edges
workflow.set_entry_point("UserInterfaceNode")
workflow.add_edge("UserInterfaceNode", "PerceptionNode1")
workflow.add_edge("PerceptionNode1", "PerceptionNode2")
workflow.add_edge("PerceptionNode2", "IntegrationNode")
workflow.add_edge("IntegrationNode", END)

# Compile the graph
logger.debug("Compiling workflow graph.")
graph = workflow.compile()
logger.debug("Workflow graph compiled successfully.")
