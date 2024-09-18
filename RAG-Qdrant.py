""" Ingesting PDF """

# Install required packages
# pip install --q unstructured langchain
# pip install --q "unstructured[all-docs]"

from langchain_community.document_loaders import UnstructuredPDFLoader

local_path = "/media/player/karna1/RAG-chatwithpdf-localy/English_Manual_vigor_.pdf"

""" Local PDF file uploads """
if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
else:
    print("Upload a PDF file")

""" Preview first page """
print(data[0].page_content)

""" Vector Embeddings """

# Install required packages
# pip install --q chromadb
# pip install --q langchain-text-splitters

from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize the Ollama embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

client = QdrantClient(url="http://localhost:6333")

# Get the dimension of embeddings by testing
sample_text = "Sample text for dimension testing."
sample_embedding = embedding_model._embed([sample_text])
dimension = len(sample_embedding[0])

# Create or recreate the collection with the correct vector dimension
try:
    try:
        client.delete_collection(collection_name="local-rag")  # Delete existing collection if any
    except Exception as e:
        print(f"An error occurred while deleting the collection: {e}")
    
    client.create_collection(
        collection_name="local-rag",
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )
except Exception as e:
    print(f"An error occurred while creating the collection: {e}")

# Add to vector database
try:
    # Prepare documents and embeddings
    documents = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model._embed(documents)
    
    # Prepare points for insertion
    points = [
        PointStruct(id=i, vector=embedding)
        for i, embedding in enumerate(embeddings)
    ]
    
    client.upsert(
        collection_name="local-rag",
        points=points
    )
except Exception as e:
    print(f"An error occurred while adding vectors to the database: {e}")

""" Retrieval """

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_qdrant import Qdrant  # Updated import for Qdrant

# LLM from Ollama
local_model = "llama3.1"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

try:
    # Initialize retriever without 'embedding' argument
    retriever = MultiQueryRetriever.from_llm(
        Qdrant(
            collection_name="local-rag"
        ).as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(chain.invoke("What is this about?"))
except Exception as e:
    print(f"An error occurred during retrieval: {e}")
