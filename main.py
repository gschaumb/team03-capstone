import os
import pandas as pd
import logging
from typing import TypedDict, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global Variables for models
GLOBAL_SENTENCE_MODEL = None
GLOBAL_HUGGINGFACE_MODEL = None
GLOBAL_HUGGINGFACE_TOKENIZER = None

# Paths for Embeddings
SEC_EMBEDDINGS_PATH = "/data/sec_embeddings.pkl"
FINANCIAL_EMBEDDINGS_PATH = "/data/financial_embeddings.pkl"


class PerceptionResult(TypedDict, total=False):
    status: str
    data: Optional[pd.DataFrame]


class IntegrationResult(TypedDict, total=False):
    status: str
    message: Optional[str]


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
def load_huggingface_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    global GLOBAL_HUGGINGFACE_MODEL, GLOBAL_HUGGINGFACE_TOKENIZER
    if GLOBAL_HUGGINGFACE_MODEL is None or GLOBAL_HUGGINGFACE_TOKENIZER is None:
        logger.debug("Loading HuggingFace model: %s", model_name)
        hf_token = os.getenv("HF_TOKEN")
        GLOBAL_HUGGINGFACE_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name, use_auth_token=hf_token
        )
        GLOBAL_HUGGINGFACE_TOKENIZER = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=hf_token
        )

        if GLOBAL_HUGGINGFACE_TOKENIZER.pad_token is None:
            GLOBAL_HUGGINGFACE_TOKENIZER.pad_token = (
                GLOBAL_HUGGINGFACE_TOKENIZER.eos_token
            )


def generate_embeddings(texts):
    if GLOBAL_SENTENCE_MODEL is None:
        raise ValueError("SentenceTransformer model not loaded.")
    logger.debug("Generating embeddings for texts.")
    embeddings = GLOBAL_SENTENCE_MODEL.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()


def compute_similarities(query_embedding, document_embeddings):
    logger.debug("Computing cosine similarities.")
    return cosine_similarity(query_embedding, document_embeddings).flatten()


# Base class for shared functionality between Perception Agents
class PerceptionAgentBase:
    def __init__(self, data_df, name, embeddings_path):
        self.data_df = data_df
        self.name = name
        self.embeddings_path = embeddings_path
        self.document_embeddings = self.load_or_generate_embeddings()
        logger.debug("Initialized PerceptionAgentBase: %s", self.name)

    def load_or_generate_embeddings(self):
        if os.path.exists(self.embeddings_path):
            logger.debug(f"Loading precomputed embeddings for {self.name}")
            with open(self.embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
        else:
            logger.debug(f"Generating and storing embeddings for {self.name}")
            embeddings = generate_embeddings(self.data_df["chunked_text"].tolist())
            with open(self.embeddings_path, "wb") as f:
                pickle.dump(embeddings, f)
        return embeddings

    def compute_similarity_and_retrieve(self, query_embedding):
        similarities = compute_similarities(query_embedding, self.document_embeddings)
        top_k_indices = similarities.argsort()[-2:][::-1]
        return self.data_df.iloc[top_k_indices]


# PerceptionAgent1 - specific logic for Agent 1
class PerceptionAgent1(PerceptionAgentBase):
    def extract_data(self, query):
        logger.debug("PerceptionAgent1 extracting data for query: %s", query)

        # Generate query embedding
        query_embedding = generate_embeddings([query])

        # Retrieve top 2 documents
        top_k_documents = self.compute_similarity_and_retrieve(query_embedding)

        # Summarize retrieved documents using LLM
        combined_text = " ".join(top_k_documents["chunked_text"].tolist())
        system_prompt = (
            "You are a legal analyst. Provide a concise summary of the key people, dates, actions, "
            "and terms from the following information. Make it two sentences long."
        )
        augmented_query = system_prompt + f"\n\nContext:\n{combined_text}"

        summary = IntegrationAgent().synthesize_data_llm(
            augmented_query, max_length=100
        )
        logger.debug("Generated summary for PerceptionAgent1: %s", summary)

        return summary, top_k_documents


# PerceptionAgent2 - specific logic for Agent 2
class PerceptionAgent2(PerceptionAgentBase):
    def extract_data(self, summary_from_agent_1):
        logger.debug(
            "PerceptionAgent2 using summary from Agent 1 as query: %s",
            summary_from_agent_1,
        )

        # Generate query embedding based on the summary
        query_embedding = generate_embeddings([summary_from_agent_1])

        # Retrieve top 2 documents
        top_k_documents = self.compute_similarity_and_retrieve(query_embedding)

        # Summarize retrieved documents using LLM
        combined_text = " ".join(top_k_documents["chunked_text"].tolist())
        system_prompt = (
            "You are a financial analyst. Provide a concise summary of the key people, dates, actions, "
            "and terms from the following financial data. Make it two sentences long."
        )
        augmented_query = system_prompt + f"\n\nContext:\n{combined_text}"

        summary = IntegrationAgent().synthesize_data_llm(
            augmented_query, max_length=100
        )
        logger.debug("Generated summary for PerceptionAgent2: %s", summary)

        return summary, top_k_documents


# IntegrationAgent remains unchanged
class IntegrationAgent:
    def synthesize_data(self, perception_summaries, query):
        logger.debug("IntegrationAgent synthesizing data.")

        combined_summaries = " ".join(perception_summaries)
        system_prompt = (
            "You are a financial expert. Provide a one-sentence summary of the following information."
            " Focus only on the most important facts and avoid repetition. "
            "Your summary should be concise and to the point."
        )
        augmented_query = (
            system_prompt + f"\n\nUser Query: {query}\n\nContext:\n{combined_summaries}"
        )

        summaries = []
        for _ in range(3):
            summary = self.synthesize_data_llm(
                augmented_query, max_length=100
            )  # Limiting to 100 tokens
            summaries.append(summary.strip())

        return summaries

    def synthesize_data_llm(self, input_text, max_length=150):
        if GLOBAL_HUGGINGFACE_MODEL is None or GLOBAL_HUGGINGFACE_TOKENIZER is None:
            raise ValueError("HuggingFace model or tokenizer not loaded.")

        inputs = GLOBAL_HUGGINGFACE_TOKENIZER(
            input_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        )

        outputs = GLOBAL_HUGGINGFACE_MODEL.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            num_return_sequences=1,
            pad_token_id=GLOBAL_HUGGINGFACE_TOKENIZER.pad_token_id,
        )
        raw_response = GLOBAL_HUGGINGFACE_TOKENIZER.decode(
            outputs[0], skip_special_tokens=True
        )
        return raw_response.strip()


# Perception Node 1 for PerceptionAgent1
def perception_node_1(state: AgentState) -> AgentState:
    logger.debug("Executing PerceptionNode1 with initial state: %s", state)

    if not state["messages"]:
        logger.error("No messages in state.")
        state["perception_1"] = {"status": "no_data", "data": pd.DataFrame()}
        return state

    query = state["messages"][-1]["content"]
    summary_1, result_1 = perception_agent_1.extract_data(query)

    if result_1.empty:
        state["perception_1"] = {"status": "no_data", "data": pd.DataFrame()}
    else:
        state["perception_1"] = {"status": "data_found", "data": summary_1}

    logger.debug("PerceptionNode1 result: %s", state["perception_1"])
    return state


# Perception Node 2 for PerceptionAgent2
def perception_node_2(state: AgentState) -> AgentState:
    logger.debug("Executing PerceptionNode2 with initial state: %s", state)

    if (
        state["perception_1"]["status"] != "data_found"
        or not state["perception_1"]["data"]
    ):
        logger.error("No valid data from PerceptionAgent1.")
        state["perception_2"] = {"status": "no_data", "data": pd.DataFrame()}
        return state

    summary_1 = state["perception_1"]["data"]
    summary_2, result_2 = perception_agent_2.extract_data(summary_1)

    if result_2.empty:
        state["perception_2"] = {"status": "no_data", "data": pd.DataFrame()}
    else:
        state["perception_2"] = {"status": "data_found", "data": summary_2}

    logger.debug("PerceptionNode2 result: %s", state["perception_2"])
    return state


# Integration Node
def integration_node(state: AgentState) -> AgentState:
    logger.debug("Executing IntegrationNode with initial state: %s", state)
    agent = IntegrationAgent()

    if (
        state["perception_1"]["status"] != "data_found"
        or state["perception_2"]["status"] != "data_found"
    ):
        logger.error("No valid perception data available for integration.")
        state["integration_result"] = {
            "status": "no_data",
            "message": "No relevant information found.",
        }
    else:
        perception_summaries = [
            state["perception_1"]["data"],
            state["perception_2"]["data"],
        ]
        query = state["messages"][-1]["content"]
        summaries = agent.synthesize_data(perception_summaries, query)

        state["integration_result"] = {
            "status": "data_integrated",
            "message": summaries,
        }

    logger.debug("Final integration result: %s", state["integration_result"])
    return state


# Initialize Models
load_sentence_transformer_model()
load_huggingface_model()

# Instantiate Perception Agents with DataFrames
sec_df = pd.read_csv("data/sec_docs.csv")
financial_df = pd.read_csv("data/financial_reports.csv")

perception_agent_1 = PerceptionAgent1(sec_df, "SEC_Perception", SEC_EMBEDDINGS_PATH)
perception_agent_2 = PerceptionAgent2(
    financial_df, "Financial_Perception", FINANCIAL_EMBEDDINGS_PATH
)
