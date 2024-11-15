from langchain_chroma import Chroma
import chromadb
import pickle
from openai import OpenAI
import os 
import pandas as pd 

#CONSTANTS
global HF_EMBEDDINGS_DB
HF_EMBEDDINGS_DB= "data/HF_embeddings.pkl"

class VectorStore:
    '''
    This class initializes the embeddings and vector store for email agent.
    '''
    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.vector_store = self.load_vectors(self.embeddings)

    def load_embeddings(embeddings_str=HF_EMBEDDINGS_DB):
        with open(HF_EMBEDDINGS_DB, "rb") as f:
            return pickle.load(f)
    
    def load_vectors(self,  embeddings):
        chroma_client = chromadb.HttpClient(host='ec2-54-252-65-170.ap-southeast-2.compute.amazonaws.com', port=8000)
        return Chroma(client=chroma_client, collection_name='chroma_db', embedding_function=embeddings)

class EmailAgent:
    '''
    This class interacts with the vector store to retrieve relevant emails based on a given query.
    It first extracts keyphrases from responses coming from Agent 1 and Agent 2 using OpenAI LLM (GPT 3.5), 
    and outputs top 10 emails by cosine similarity.
    '''
    def __init__(self, VectorStore):
        self.embeddings = VectorStore.embeddings
        self.vector_store = VectorStore.vector_store
        self.openai_client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
    
    def get_keyphrases(self, query):
        response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages = [
                        {"role": "system",
                        "content": "Extract phrases that can be searched in email text to find relevant emails. Your output should be key phrases separated by a comma."},
                        {"role": "user",
                        "content": query}
                        ],
                    max_tokens = 200,  # Maximum output length
                    temperature = 0
                    )
        return response.choices[0].message.content


    def retrieve_emails(self, query):
        keyphrases =  self.get_keyphrases(query)
        results = self.vector_store.similarity_search_with_score(keyphrases, k=20)
        return results, keyphrases
    
    def retrieve_emails_from_all_agents(self, current_state):
        results=[]
        keyphrases=[]
        for key, value in current_state.items():
            if (key.startswith('perception') and value['status']=='data_found'):
                email_response, keyphrases_from_agent= self.retrieve_emails(value['summary'])
                keyphrases.append({key:keyphrases_from_agent})
                email= email_response[0]
                for email in email_response:
                    results.append([email[0].metadata['date_sent'], email[0].metadata['from'], email[0].metadata['to'], email[0].page_content])

        results_df= pd.DataFrame(results, columns=['date_sent', 'from', 'to', 'content'])
        results_df.drop_duplicates(inplace=True)
        return results_df, keyphrases 


#initialize vectorstore
vector_store = VectorStore()

'''email_agent = EmailAgent(vector_store)
responses= email_agent.retrieve_emails("Raptor hedges were fraudulent hedging devices implemented at Enron by the CFO Andrew Fastow, which led to financial losses and raised concerns about the company's financial integrity.")
print (len(responses[0]))
'''

