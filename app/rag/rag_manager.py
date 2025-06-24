from app.rag.SelfRAG_Workflow.advanced_workflow_orchestrator import SelfRAGWorkflow
from app.rag.SelfRAG_Workflow.vector_store_and_retrieval import SelfRAGVectorStore
from app.rag.SelfRAG_Workflow.web_search_synthesize import WebSearchAgent
import logging

logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self):
        self.orchestrator = None
    
    def initialize(self):
        """Initialize the RAG system"""
        if self.orchestrator is not None:
            return
            
        logger.info("✅ Initializing vector store...")
        vector_store = SelfRAGVectorStore(collection_name="news_articles")
        vector_store.build_vector_index()

        logger.info("✅ Initializing web search agent...")
        web_search_agent = WebSearchAgent()

        logger.info("✅ Initializing orchestrator...")
        self.orchestrator = SelfRAGWorkflow(
            vector_store=vector_store,
            web_search_agent=web_search_agent,
            timeout=60
        )
    
    def get_orchestrator(self):
        if self.orchestrator is None:
            raise RuntimeError("RAG system not initialized")
        return self.orchestrator

# Global instance
rag_manager = RAGManager()