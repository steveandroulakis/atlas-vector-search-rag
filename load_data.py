from pymongo import MongoClient
from pymongo.operations import SearchIndexModel 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
import key_param

# Set the MongoDB URI, DB, Collection Names
client: MongoClient = MongoClient(key_param.MONGO_URI)
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

# Create the search index if it doesn't exist
def ensure_search_index():
    try:
        # Check if index exists
        existing_indexes = list(collection.list_indexes())
        index_exists = any(index.get('name') == 'default' for index in existing_indexes)
        
        if not index_exists:
            # Create the search index
            index_definition = {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "embedding": {
                            "dimensions": 1536,
                            "similarity": "cosine",
                            "type": "knnVector"
                        }
                    }
                }
            }
            
            # Create SearchIndexModel with definition and name
            search_index = SearchIndexModel(
                definition=index_definition,
                name="default"
            )
            
            # Create the index
            collection.create_search_index(search_index)
            print("Created vector search index 'default'")
        else:
            print("Search index 'default' already exists")
            
    except Exception as e:
        print(f"Error creating search index: {str(e)}")

ensure_search_index()

# Initialize the DirectoryLoader
loader = DirectoryLoader('./sample_files', glob="./*.txt", show_progress=True)
data = loader.load()

# Define the OpenAI Embedding Model
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # Specify the desired model
    # dimensions=1024  # Optional: specify dimensions if supported by the model
)

# Initialize the VectorStore and load documents
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    data, 
    embeddings, 
    collection=collection
)

print("Documents loaded and vectorized successfully")