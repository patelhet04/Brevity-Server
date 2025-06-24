import chromadb
import torch
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from typing import List, Dict, Any
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from app.db.article_crud import scan_recent_articles

class SelfRAGVectorStore:
    def __init__(self, collection_name: str = "news_articles"):
        """
        Initialize the Self-RAG Vector Store
        
        Args:
            collection_name: Name for the ChromaDB collection
        """
        self.collection_name = collection_name

        # Set device for embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en", device=device)

        self.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[.!?]+",
        )

        self.chroma_client = None
        self.vector_store = None
        self.index = None
        self.retriever = None
        
    def initialize_chroma_store(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
    def create_documents_from_articles(self, articles_data: List[Dict[str, Any]]) -> List[Document]:
        """
        Create LlamaIndex documents from DynamoDB articles
        
        Args:
            articles_data: List of articles from DynamoDB
            
        Returns:
            List of LlamaIndex Document objects
        """
        documents = []
        
        for article in articles_data:
            # Create document text (adjust based on your article structure)
            text_content = f"{article.get('title', '')}\n\n{article.get('summary', '')}"
            
            # Create metadata
            metadata = {
                "url": article.get('url', ''),
                "title": article.get('title', ''),
                "source": article.get('source_name', ''),
                "published_date": article.get('published_date', ''),
                "category": article.get('category', '')
            }
            
            # Create Document
            doc = Document(
                text=text_content,
                metadata=metadata,
                doc_id=metadata["url"]
            )
            documents.append(doc)
            
        return documents
    
    def build_vector_index(self, persist_directory: str = "./chroma_db"):
        """
        Build vector index from DynamoDB articles
        
        Args:
            scan_recent_articles_func: Function to fetch articles from DynamoDB
            persist_directory: Directory to persist ChromaDB data
        """
        # Initialize ChromaDB
        self.initialize_chroma_store(persist_directory)
        
        print("Fetching articles from DynamoDB...")
        
        # Fetch articles from DynamoDB
        api_response = scan_recent_articles()
        
        if not api_response.get("success", False):
            raise Exception("Failed to fetch articles from DynamoDB")
            
        articles_data = api_response.get("data", [])
        print(f"Fetched {len(articles_data)} articles")
        
        # Create documents
        print("Creating LlamaIndex documents...")
        documents = self.create_documents_from_articles(articles_data)
        
        # Parse documents into nodes
        print("Splitting documents into nodes...")
        nodes = self.node_parser.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} nodes")
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        try:
            existing_collection = self.chroma_client.get_collection(self.collection_name)
            doc_count = existing_collection.count()
            
            if doc_count > 0:
                print(f"Found existing vector store with {doc_count} documents. Loading existing index...")
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                    embed_model=self.embed_model
                )
                print("Existing vector index loaded successfully!")
            else:
                print("Empty vector store found. Building new index...")
                self.index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    show_progress=True
                )
                print("New vector index built successfully!")
                
        except Exception as e:
            print(f"Error checking existing collection: {e}")
            print("Building new vector index...")
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )
            print("New vector index built successfully!")
        # Create retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
            embed_model=self.embed_model
        )
        
        print("Vector index built successfully!")
        
    def retrieve_and_filter_nodes(self, query: str, similarity_threshold: float = 0.7):
        """
        Retrieve and filter nodes based on similarity score
        
        Args:
            query: User query string
            similarity_threshold: Minimum similarity score (0.7 by default)
            
        Returns:
            List of filtered nodes with similarity >= threshold
        """
        if not self.retriever:
            raise Exception("Vector index not built. Call build_vector_index() first.")
            
        # Retrieve nodes
        retrieved_nodes = self.retriever.retrieve(query)
        
        for node in retrieved_nodes:
            print("============================")
            print(node.text)
            print("\n")
            print(node.metadata)
            print("\n")
            print(node.score)
            print("\n")
            print("============================")
        # Filter based on similarity score
        filtered_nodes = [
            node for node in retrieved_nodes 
            if node.score >= similarity_threshold
        ]
        
        print(f"Retrieved {len(retrieved_nodes)} nodes, filtered to {len(filtered_nodes)} nodes with score >= {similarity_threshold}")
        
        return filtered_nodes
    
# Example usage
if __name__ == "__main__":
    
    # Initialize the vector store
    rag_store = SelfRAGVectorStore(collection_name="news_articles")
    
    # Build the vector index
    rag_store.build_vector_index()
    
    # Use the retriever
    query = "What are the latest developments in AI?"
    filtered_nodes = rag_store.retrieve_and_filter_nodes(query, similarity_threshold=0.7)
    
    print(f"Found {len(filtered_nodes)} relevant documents for query: '{query}'")