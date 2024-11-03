import os
import pandas as pd
import logging
from typing import TypedDict, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import pickle

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global Variables for models
GLOBAL_SENTENCE_MODEL = None
GLOBAL_OPENAI_CLIENT = None
GLOBAL_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths for Embeddings
CASE_EMBEDDINGS_PATH = "/data/case_embeddings.pkl"
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
    perception_3: PerceptionResult
    integration_result: IntegrationResult


# Load SentenceTransformer model globally
def load_sentence_transformer_model(model_name="all-MiniLM-L6-v2"):
    global GLOBAL_SENTENCE_MODEL
    if GLOBAL_SENTENCE_MODEL is None:
        logger.debug("Loading SentenceTransformer model: %s", model_name)
        GLOBAL_SENTENCE_MODEL = SentenceTransformer(model_name)


def generate_embeddings(texts):
    if GLOBAL_SENTENCE_MODEL is None:
        raise ValueError("SentenceTransformer model not loaded.")
    logger.debug("Generating embeddings for texts.")
    embeddings = GLOBAL_SENTENCE_MODEL.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()


def compute_similarities(query_embedding, document_embeddings):
    logger.debug("Computing cosine similarities.")
    return cosine_similarity(query_embedding, document_embeddings).flatten()


# Initialize OpenAI GPT-3.5 Turbo
def initialize_openai():
    global GLOBAL_OPENAI_CLIENT
    if not GLOBAL_OPENAI_CLIENT:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found.")
        from openai import OpenAI

        GLOBAL_OPENAI_CLIENT = OpenAI(api_key=api_key)
        logger.debug("OpenAI GPT-3.5 Turbo client initialized.")


def openai_generate_response(messages, max_tokens=150):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=max_tokens
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}")
        return None


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


# New PerceptionAgent1 for `case_docs.csv`
class PerceptionAgent1(PerceptionAgentBase):
    def extract_data(self, query):
        logger.debug("PerceptionAgent1 extracting data for query: %s", query)
        query_embedding = generate_embeddings([query])
        top_k_documents = self.compute_similarity_and_retrieve(query_embedding)

        combined_text = " ".join(top_k_documents["chunked_text"].tolist())
        system_prompt = (
            "You are an expert summarizer. Based on the following information, answer the user query: "
            f"'{query}'. Summarize in 2 sentences focusing on the most relevant details."
        )

        messages = [{"role": "system", "content": system_prompt}]
        for doc in top_k_documents["chunked_text"]:
            messages.append({"role": "user", "content": doc})

        try:
            response = GLOBAL_OPENAI_CLIENT.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, max_tokens=550
            )
            summary = response.choices[0].message.content.strip()
            logger.debug("Generated summary for PerceptionAgent1: %s", summary)
            return summary, top_k_documents
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None, top_k_documents


# PerceptionAgent2 for sec_docs.csv (previously PerceptionAgent1)
class PerceptionAgent2(PerceptionAgentBase):
    def extract_data(self, query):
        logger.debug("PerceptionAgent2 extracting data for query: %s", query)
        query_embedding = generate_embeddings([query])
        top_k_documents = self.compute_similarity_and_retrieve(query_embedding)
        combined_text = " ".join(top_k_documents["chunked_text"].tolist())
        system_prompt = "Summarize in 2 sentences focusing on relevant people, dates, actions, and terms."

        messages = [{"role": "system", "content": system_prompt}]
        for doc in top_k_documents["chunked_text"]:
            messages.append({"role": "user", "content": doc})

        try:
            response = GLOBAL_OPENAI_CLIENT.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, max_tokens=550
            )
            summary = response.choices[0].message.content.strip()
            return summary, top_k_documents
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None, top_k_documents


# PerceptionAgent3 for financial_reports.csv (previously PerceptionAgent2)
class PerceptionAgent3(PerceptionAgentBase):
    def extract_data(self, summary_from_agent_2, query):
        query_embedding = generate_embeddings([summary_from_agent_2])
        top_k_documents = self.compute_similarity_and_retrieve(query_embedding)

        combined_text = " ".join(top_k_documents["chunked_text"].tolist())
        system_prompt = "Summarize in 2 sentences highlighting the most relevant financial facts and events."

        messages = [{"role": "system", "content": system_prompt}]
        for doc in top_k_documents["chunked_text"]:
            messages.append({"role": "user", "content": doc})

        try:
            response = GLOBAL_OPENAI_CLIENT.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, max_tokens=550
            )
            summary = response.choices[0].message.content.strip()
            return summary, top_k_documents
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None, top_k_documents


# IntegrationAgent for final summary
class IntegrationAgent:
    def synthesize_data(self, perception_summaries, query):
        combined_summaries = " ".join(
            self.clean_summary(summary) for summary in perception_summaries
        )

        system_prompt = (
            "Using the following details, answer the user query: "
            f"'{query}' in a concise, single-sentence summary."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_summaries},
        ]

        summary = self.synthesize_data_llm(
            system_prompt + "\n\nContext:\n" + combined_summaries, max_length=60
        )
        return [summary.strip()]

    def clean_summary(self, summary):
        cleaned = summary.split("Context:")[-1]
        return cleaned.strip()

    def synthesize_data_llm(self, input_text, max_length=150):
        messages = [
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": input_text},
        ]

        try:
            response = GLOBAL_OPENAI_CLIENT.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, max_tokens=max_length
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None


# Define Perception Nodes
def perception_node_1(state: AgentState) -> AgentState:
    query = state["messages"][-1]["content"]
    summary_1, result_1 = perception_agent_1.extract_data(query)

    state["perception_1"] = {
        "status": "data_found" if not result_1.empty else "no_data",
        "data": summary_1,
    }
    return state


def perception_node_2(state: AgentState) -> AgentState:
    query = state["messages"][-1]["content"]
    summary_2, result_2 = perception_agent_2.extract_data(query)

    state["perception_2"] = {
        "status": "data_found" if not result_2.empty else "no_data",
        "data": summary_2,
    }
    return state


def perception_node_3(state: AgentState) -> AgentState:
    query = state["messages"][-1]["content"]
    summary_3, result_3 = perception_agent_3.extract_data(
        state["perception_2"]["data"], query
    )

    state["perception_3"] = {
        "status": "data_found" if not result_3.empty else "no_data",
        "data": summary_3,
    }
    return state


def integration_node(state: AgentState) -> AgentState:
    agent = IntegrationAgent()

    perception_summaries = [
        state["perception_1"]["data"],
        state["perception_2"]["data"],
        state["perception_3"]["data"],
    ]

    query = state["messages"][-1]["content"]
    summaries = agent.synthesize_data(perception_summaries, query)
    state["integration_result"] = {"status": "data_integrated", "message": summaries}
    return state


# Initialize Models
load_sentence_transformer_model()
initialize_openai()

# Instantiate Perception Agents with DataFrames
case_df = pd.read_csv("data/case_docs.csv")
sec_df = pd.read_csv("data/sec_docs.csv")
financial_df = pd.read_csv("data/financial_reports.csv")

perception_agent_1 = PerceptionAgent1(case_df, "Case_Perception", CASE_EMBEDDINGS_PATH)
perception_agent_2 = PerceptionAgent2(sec_df, "SEC_Perception", SEC_EMBEDDINGS_PATH)
perception_agent_3 = PerceptionAgent3(
    financial_df, "Financial_Perception", FINANCIAL_EMBEDDINGS_PATH
)
