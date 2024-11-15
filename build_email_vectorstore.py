#%%
import chromadb
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import time 


#%%
chroma_client = chromadb.HttpClient(host='ec2-13-236-71-16.ap-southeast-2.compute.amazonaws.com', port=8000)

#%%
collection = chroma_client.get_or_create_collection(name= 'chroma_db')

# %%
collection = chroma_client.get_collection('chroma_db')
# %%
def load_email_pickle(filepath="data/email_data_cleaned.pkl"):
    with open(filepath, "rb") as f:
        email_data = pickle.load(f)
        return email_data
#%%
email_data = load_email_pickle()
#%%
email_list=[]
for i, row in email_data.iterrows():
    email_list.append(Document(page_content=row['content_cleaned'], metadata={"from":row['from'], "to":row['to'], "date_sent":row['date_sent']}, id=i ) )
    
# %%
def get_embeddings():
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    with open("data/HF_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
#%%
with open("data/HF_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
#%%

chroma_db = Chroma(
        collection_name="chroma_db",
        embedding_function=embeddings,
        client=chroma_client
    )

# %%
step_size= 5_000
num_iterations = (len(email_data) // step_size) + 1
print (num_iterations)
# %%
for i in range(0, num_iterations):
    print (f'{i} of {num_iterations}')
    start_index= i*step_size
    end_index = (i*step_size) + step_size
    if end_index > len(email_data):
        end_index = len(email_data)
    start_time = time.time()
    chroma_db.add_documents(documents=email_list[start_index:end_index])
    time_taken = time.time() - start_time
    print(f"Step {i}: Time taken to add documents: {time_taken:.2f} seconds")