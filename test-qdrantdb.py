""" Ingesting PDF """

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

local_path = "/media/player/karna1/RAG-chatwithpdf-localy/WEF_The_Global_Cooperation_Barometer_2024.pdf"

""" Local PDF file uploads """
if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
else:
    print("Upload a PDF file")

""" Preview first page """
print(data[0].page_content)

""" Vector Embeddings """

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the Ollama embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# Example text to determine embedding size
example_text = "sample text for dimension check"
example_embedding = embedding_model.embed_query(example_text)
embedding_size = len(example_embedding)

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = []
for doc in data:
    chunks.extend(text_splitter.split_documents([doc]))

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Create or recreate the collection with the correct vector dimension
client.create_collection(
    collection_name="local-rag",
    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE)  # Use dynamic size
)

# Loop over the document chunks and add them to Qdrant
for idx, chunk in enumerate(chunks):
    # Get the text content from each chunk
    text = chunk.page_content
    
    # Generate the embedding for the chunk
    embedding = embedding_model.embed_query(text)
    
    # Create a point structure with embedding and metadata
    point = PointStruct(
        id=idx,
        vector=embedding,
        payload={"text": text, **chunk.metadata}  # Include metadata if needed
    )
    
    # Upload the point to Qdrant
    client.upsert(collection_name="local-rag", points=[point])

print("Documents successfully added to Qdrant!")
