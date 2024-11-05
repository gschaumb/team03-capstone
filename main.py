import os
import pandas as pd
import logging
from typing import TypedDict, List, Dict, Optional, Callable
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
    def __init__(self, data_df, name, embeddings_path, retrieval_pipeline=None):
        self.data_df = data_df
        self.name = name
        self.embeddings_path = embeddings_path
        self.document_embeddings = self.load_or_generate_embeddings()
        self.retrieval_pipeline = retrieval_pipeline if retrieval_pipeline else []
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

    # Retrieval Option 1: Apply a similarity threshold
    def apply_similarity_threshold(self, similarities, threshold=0.5):
        logger.debug("Applying similarity threshold: %s", threshold)
        filtered_indices = [
            i for i, score in enumerate(similarities) if score >= threshold
        ]
        return filtered_indices

    # Retrieval Option 2: Select the top K documents
    def select_top_k(self, similarities, top_k=1):
        logger.debug("Selecting top %d documents by similarity", top_k)
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        return top_k_indices

    # Retrieval Option 3: Retrieve with context (neighbors before and after)
    def retrieve_with_context(self, selected_indices, context_window=1):
        logger.debug("Retrieving with context window: %d", context_window)
        extended_indices = set(selected_indices)
        for index in selected_indices:
            for offset in range(-context_window, context_window + 1):
                if 0 <= index + offset < len(self.data_df):
                    extended_indices.add(index + offset)
        return sorted(extended_indices)

    # Execute retrieval pipeline
    def compute_similarity_and_retrieve(self, query_embedding):
        logger.debug("Executing retrieval pipeline for %s", self.name)
        similarities = compute_similarities(query_embedding, self.document_embeddings)

        # Start with all document indices
        selected_indices = list(range(len(self.data_df)))

        # Apply each step in the pipeline
        for step in self.retrieval_pipeline:
            selected_indices = step(similarities, selected_indices)

        # Return DataFrame with retrieved documents, or empty if no indices meet criteria
        if not selected_indices:
            logger.warning(
                f"{self.name}: No documents retrieved above threshold or criteria."
            )
            return pd.DataFrame()

        return self.data_df.iloc[selected_indices]


# Each agent can now specify its own retrieval pipeline


# New PerceptionAgent1 for `case_docs.csv`
class PerceptionAgent1(PerceptionAgentBase):
    def __init__(self, data_df, name, embeddings_path):
        retrieval_pipeline = [
            lambda similarities, indices: self.apply_similarity_threshold(
                similarities, threshold=0.5
            ),
            lambda similarities, indices: self.select_top_k(similarities, top_k=1),
            lambda similarities, indices: self.retrieve_with_context(
                indices, context_window=1
            ),
        ]
        super().__init__(data_df, name, embeddings_path, retrieval_pipeline)

    def extract_data(self, query):
        logger.debug("PerceptionAgent1 extracting data for query: %s", query)
        query_embedding = generate_embeddings([query])
        top_k_documents = self.compute_similarity_and_retrieve(query_embedding)

        # Check if no documents were retrieved
        if top_k_documents.empty:
            logger.warning(
                "PerceptionAgent1 found no documents above the similarity threshold."
            )
            return (
                None,
                pd.DataFrame(),
            )  # Return None for summary, empty DataFrame for data

        # Summarize retrieved documents using GPT-3.5 Turbo
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


# PerceptionAgent2 (previously PerceptionAgent1)
class PerceptionAgent2(PerceptionAgentBase):
    def __init__(self, data_df, name, embeddings_path):
        retrieval_pipeline = [
            lambda similarities, indices: self.apply_similarity_threshold(
                similarities, threshold=0.6
            ),
            lambda similarities, indices: self.select_top_k(similarities, top_k=2),
        ]
        super().__init__(data_df, name, embeddings_path, retrieval_pipeline)

    def extract_data(self, query):
        logger.debug("PerceptionAgent2 extracting data for query: %s", query)
        query_embedding = generate_embeddings([query])
        top_k_documents = self.compute_similarity_and_retrieve(query_embedding)

        if top_k_documents.empty:
            logger.warning(
                "PerceptionAgent2 found no documents above the similarity threshold."
            )
            return None, pd.DataFrame()

        # Summarize retrieved documents using GPT-3.5 Turbo
        combined_text = " ".join(top_k_documents["chunked_text"].tolist())
        system_prompt = (
            "You are an expert summarizer. Based on the following information, answer the user query: "
            f"'{query}'. Summarize in 2 sentences focusing on relevant people, dates, actions, and terms."
        )

        messages = [{"role": "system", "content": system_prompt}]
        for doc in top_k_documents["chunked_text"]:
            messages.append({"role": "user", "content": doc})

        try:
            response = GLOBAL_OPENAI_CLIENT.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, max_tokens=550
            )
            summary = response.choices[0].message.content.strip()
            logger.debug("Generated summary for PerceptionAgent2: %s", summary)
            return summary, top_k_documents
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None, top_k_documents


# PerceptionAgent3 (previously PerceptionAgent2)
class PerceptionAgent3(PerceptionAgentBase):
    def __init__(self, data_df, name, embeddings_path):
        retrieval_pipeline = [
            lambda similarities, indices: self.select_top_k(similarities, top_k=1),
            lambda similarities, indices: self.retrieve_with_context(
                indices, context_window=2
            ),
        ]
        super().__init__(data_df, name, embeddings_path, retrieval_pipeline)

    def extract_data(self, query):
        logger.debug("PerceptionAgent3 extracting data for query: %s", query)
        query_embedding = generate_embeddings([query])
        top_k_documents = self.compute_similarity_and_retrieve(query_embedding)

        if top_k_documents.empty:
            logger.warning(
                "PerceptionAgent3 found no documents above the similarity threshold."
            )
            return None, pd.DataFrame()

        # Summarize retrieved documents using GPT-3.5 Turbo
        combined_text = " ".join(top_k_documents["chunked_text"].tolist())
        system_prompt = (
            "You are an expert in financial analysis. Based on the user query: "
            f"'{query}', summarize the following financial data in 2 sentences, "
            "highlighting the most relevant financial facts and key events."
        )

        messages = [{"role": "system", "content": system_prompt}]
        for doc in top_k_documents["chunked_text"]:
            messages.append({"role": "user", "content": doc})

        try:
            response = GLOBAL_OPENAI_CLIENT.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, max_tokens=550
            )
            summary = response.choices[0].message.content.strip()
            logger.debug("Generated summary for PerceptionAgent3: %s", summary)
            return summary, top_k_documents
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None, top_k_documents


# Instantiate Perception Agents with DataFrames
case_df = pd.read_csv("data/case_docs.csv")
sec_df = pd.read_csv("data/sec_docs.csv")
financial_df = pd.read_csv("data/financial_reports.csv")

perception_agent_1 = PerceptionAgent1(case_df, "Case_Perception", CASE_EMBEDDINGS_PATH)
perception_agent_2 = PerceptionAgent2(sec_df, "SEC_Perception", SEC_EMBEDDINGS_PATH)
perception_agent_3 = PerceptionAgent3(
    financial_df, "Financial_Perception", FINANCIAL_EMBEDDINGS_PATH
)
