import os
import pandas as pd
import logging
from typing import TypedDict, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Variables for models
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
    else:
        logger.debug("SentenceTransformer model already loaded.")

# Load HuggingFace model and tokenizer globally
def load_huggingface_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    global GLOBAL_HUGGINGFACE_MODEL, GLOBAL_HUGGINGFACE_TOKENIZER
    if GLOBAL_HUGGINGFACE_MODEL is None or GLOBAL_HUGGINGFACE_TOKENIZER is None:
        logger.debug("Loading HuggingFace model: %s", model_name)
        hf_token = os.getenv("HF_TOKEN")
        GLOBAL_HUGGINGFACE_MODEL = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
        GLOBAL_HUGGINGFACE_TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

        # Ensure that the tokenizer has a padding token (use eos_token as padding token if needed)
        if GLOBAL_HUGGINGFACE_TOKENIZER.pad_token is None:
            logger.debug("Assigning pad_token as eos_token for the tokenizer.")
            GLOBAL_HUGGINGFACE_TOKENIZER.pad_token = GLOBAL_HUGGINGFACE_TOKENIZER.eos_token
    else:
        logger.debug("HuggingFace model and tokenizer already loaded.")

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
        logger.debug("Generated query embedding.")

        # Compute similarities between the query and document embeddings
        similarities = compute_similarities(query_embedding, self.document_embeddings)
        logger.debug("Computed similarities.")

        # Retrieve top 3 similar documents
        top_k_indices = similarities.argsort()[-3:][::-1]
        logger.debug("Top 3 similar document indices: %s", top_k_indices)

        # Extract top documents
        top_k_documents = self.data_df.iloc[top_k_indices]
        logger.debug("PerceptionAgent (%s) found top documents.", self.name)

        return top_k_documents

# Define Integration Agent
class IntegrationAgent:
    def synthesize_data(self, perception_results, query):
        if GLOBAL_HUGGINGFACE_MODEL is None or GLOBAL_HUGGINGFACE_TOKENIZER is None:
            raise ValueError("HuggingFace model or tokenizer not loaded.")
    
        logger.debug("IntegrationAgent synthesizing data for query: %s", query)
    
        # Combine all top documents' text for context
        combined_text = " ".join(perception_results['chunked_text'].tolist())
        
        # Create a system prompt to guide the LLM
        system_prompt = (
            "You are a financial analyst. Please summarize the following information in clear, "
            "well-structured prose, focusing on the most important points relevant to the user's query. "
            "Avoid technical jargon where possible and ensure the response is coherent and easy to understand.\n\n"
        )

        # Augment the user query with the perception data and the system prompt
        augmented_query = system_prompt + f"Query: {query}\n\nContext:\n{combined_text}"

        # Tokenize the input for Mistral (adjust max_length to fit model's capabilities)
        inputs = GLOBAL_HUGGINGFACE_TOKENIZER(augmented_query, return_tensors="pt", padding="longest", truncation=True, max_length=512)

        try:
            # Generate the model's response
            logger.debug("Calling HuggingFace model to generate a response.")
            GLOBAL_HUGGINGFACE_MODEL.eval()
        
            # Generate output (adjust max_new_tokens as needed)
            outputs = GLOBAL_HUGGINGFACE_MODEL.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                max_new_tokens=300,  # Increased from 150 to handle longer responses
                num_return_sequences=1,
                pad_token_id=GLOBAL_HUGGINGFACE_TOKENIZER.pad_token_id  
            )
        
            # Decode the output
            raw_response = GLOBAL_HUGGINGFACE_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            logger.debug("IntegrationAgent generated raw response: %s", raw_response)

            cleaned_response = raw_response.replace("[INST]", "").replace("[/INST]", "").strip()
            logger.debug("IntegrationAgent cleaned response: %s", cleaned_response)

            return cleaned_response
    
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
    
    logger.debug("PerceptionNode1 result: %s", state['perception_1'])
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

    logger.debug("PerceptionNode2 result: %s", state['perception_2'])
    return state

# Integration Node (with checks before calling the LLM)
def integration_node(state: AgentState) -> AgentState:
    logger.debug("Executing IntegrationNode with initial state: %s", state)
    agent = IntegrationAgent()

    # Collect valid perception results
    valid_results = [
        state[key]['data'] for key in ['perception_1', 'perception_2']
        if state[key]['status'] == 'data_found' and isinstance(state[key]['data'], pd.DataFrame) and not state[key]['data'].empty
    ]

    if len(valid_results) == 0:
        logger.error("No valid perception data available for integration.")
        state['integration_result'] = {"status": "no_data", "message": "No relevant information found."}
    else:
        # Combine the valid results
        perception_results = pd.concat(valid_results, ignore_index=True)
        query = state['messages'][-1]['content']  
        logger.debug("Perception results to be integrated: %s", perception_results)

        try:
            # Generate the response using the IntegrationAgent
            response = agent.synthesize_data(perception_results, query)

            if response and response.strip():
                logger.debug("IntegrationAgent generated a valid response: %s", response.strip())
                state['integration_result'] = {"status": "data_integrated", "message": response.strip()}
            else:
                logger.error("IntegrationAgent returned an empty response.")
                state['integration_result'] = {"status": "no_data", "message": "Failed to generate response."}
        
        except Exception as e:
            logger.error("Error during response generation in IntegrationAgent: %s", e)
            state['integration_result'] = {"status": "error", "message": "An error occurred during response synthesis."}

    logger.debug("Final integration result: %s", state['integration_result'])
    return state

# Initialize Models
load_sentence_transformer_model()  
load_huggingface_model()  

# Instantiate Perception Agents with DataFrames
sec_df = pd.read_csv("data/sec_docs.csv")
financial_df = pd.read_csv("data/financial_reports.csv")

perception_agent_1 = PerceptionAgent(sec_df, "SEC_Perception", SEC_EMBEDDINGS_PATH)
perception_agent_2 = PerceptionAgent(financial_df, "Financial_Perception", FINANCIAL_EMBEDDINGS_PATH)
