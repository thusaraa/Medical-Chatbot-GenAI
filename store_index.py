from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY =os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()

# Initialize Pinecone Client (New API)
pc = Pinecone(api_key=PINECONE_API_KEY)  # Replace with actual API key

index_name = "medicalbot"

# Create Index with `ServerlessSpec`
pc.create_index(
    name=index_name,
    dimension=384,  # Replace with your model dimensions
    metric="cosine",  # Replace with your model metric
    spec=ServerlessSpec(  # ✅ Required in Pinecone 6.0.1
        cloud="aws",
        region="us-east-1"
    )
)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)