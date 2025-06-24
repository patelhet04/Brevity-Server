import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import sys
from pathlib import Path
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step
)
from llama_index.llms.ollama import Ollama

# Import our custom modules
from vector_store_and_retrieval import SelfRAGVectorStore
from web_search_synthesize import WebSearchAgent
# from app.schemas.articles import ChatMessage, ChatResponse, ChatRequest
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from app.schemas.articles import ChatMessage, ChatResponse, ChatRequest

# Event Classes for Workflow
class RetrieverEvent(Event):
    """Event to trigger document retrieval"""
    query: str
    chat_history: List[ChatMessage]
    conversation_id: Optional[str]


class QueryRouterEvent(Event):
    """Event containing retrieved nodes for routing decision"""
    query: str
    chat_history: List[ChatMessage]
    conversation_id: Optional[str]
    retrieved_nodes: List[Any]


class LLMSynthesizerEvent(Event):
    """Event to trigger LLM synthesis with retrieved context"""
    query: str
    chat_history: List[ChatMessage]
    conversation_id: Optional[str]
    retrieved_nodes: List[Any]


class WebSearchEvent(Event):
    """Event to trigger web search when insufficient context"""
    query: str
    chat_history: List[ChatMessage]
    conversation_id: Optional[str]


class SelfRAGWorkflow(Workflow):
    """
    Self-RAG Workflow Implementation
    
    Flow:
    1. RetrieverEvent -> Query documents from vector store
    2. QueryRouterEvent -> Route based on number of retrieved documents
    3a. LLMSynthesizerEvent -> Generate response using retrieved context
    3b. WebSearchEvent -> Search web when insufficient context
    4. StopEvent -> Return final response
    """
    
    def __init__(
        self, 
        vector_store: SelfRAGVectorStore,
        web_search_agent: WebSearchAgent,
        llm: Optional[Ollama] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Use injected components
        self.vector_store = vector_store
        self.web_search_agent = web_search_agent
        
        # Initialize LLM (with default if not provided)
        self.llm = llm or Ollama(
            model="gemma2:2B",
            request_timeout=120.0,
            temperature=0.7,
            num_predict=512,
            num_ctx=4096
        )
        
        print("Self-RAG Workflow initialized with injected dependencies")
    
    @step
    async def retriever_step(self, ev: StartEvent) -> RetrieverEvent:
        """
        Step 1: Extract data from ChatRequest and trigger retrieval
        """
        chat_request = ev.get("chat_request")
        
        print(f"Starting retrieval for query: {chat_request.query}")
        
        return RetrieverEvent(
            query=chat_request.query,
            chat_history=chat_request.history,
            conversation_id=chat_request.conversation_id
        )
    
    @step
    async def retrieve_documents(self, ev: RetrieverEvent) -> QueryRouterEvent:
        """
        Step 2: Retrieve and filter documents from vector store
        """
        print(f"Retrieving documents for query: {ev.query}")
        
        try:
            # Retrieve and filter nodes
            retrieved_nodes = self.vector_store.retrieve_and_filter_nodes(
                query=ev.query,
                similarity_threshold=0.6
            )
            
            print(f"Retrieved {len(retrieved_nodes)} relevant documents")
            
            return QueryRouterEvent(
                query=ev.query,
                chat_history=ev.chat_history,
                conversation_id=ev.conversation_id,
                retrieved_nodes=retrieved_nodes
            )
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            # Return empty nodes on error
            return QueryRouterEvent(
                query=ev.query,
                chat_history=ev.chat_history,
                conversation_id=ev.conversation_id,
                retrieved_nodes=[]
            )
    
    @step
    async def query_router(self, ev: QueryRouterEvent) -> LLMSynthesizerEvent | WebSearchEvent:
        """
        Step 3: Route based on number of retrieved documents
        """
        num_documents = len(ev.retrieved_nodes)
        print(f"Query router: Found {num_documents} documents")
        
        if num_documents >= 3:
            ev.retrieved_nodes = ev.retrieved_nodes[:2] if len(ev.retrieved_nodes) > 2 else ev.retrieved_nodes
            print("Sufficient context found. Routing to LLM Synthesizer")
            return LLMSynthesizerEvent(
                query=ev.query,
                chat_history=ev.chat_history,
                conversation_id=ev.conversation_id,
                retrieved_nodes=ev.retrieved_nodes
            )
        else:
            print("Insufficient context. Routing to Web Search")
            return WebSearchEvent(
                query=ev.query,
                chat_history=ev.chat_history,
                conversation_id=ev.conversation_id
            )
    
    @step
    async def llm_synthesizer(self, ev: LLMSynthesizerEvent) -> StopEvent:
        """
        Step 4a: Generate response using LLM with retrieved context
        """
        print("Synthesizing response using retrieved context")
        
        try:
            # Format retrieved context
            context_text = "\n\n".join([
                f"Document {i+1}:\nTitle: {node.metadata.get('title', 'N/A')}\nContent: {node.text}"
                for i, node in enumerate(ev.retrieved_nodes)  # Use top 5 documents
            ])
            
            # Format chat history
            chat_context = ""
            if ev.chat_history:
                chat_context = "Previous conversation:\n"
                for msg in ev.chat_history[-4:]:  # Last 4 messages
                    chat_context += f"{msg.role}: {msg.content}\n"
                chat_context += "\n"
            
            # Create synthesis prompt
            synthesis_prompt = f"""{chat_context}Based on the following context from news articles, please answer the user's question:

Context:
{context_text}

User Question: {ev.query}

Provide a comprehensive answer based primarily on the context provided. If the context doesn't fully answer the question, mention what information is available."""
            
            # Generate response
            response = await self.llm.acomplete(synthesis_prompt)
            response_text = str(response)
            
            # Update chat history
            updated_history = ev.chat_history + [
                ChatMessage(role="user", content=ev.query),
                ChatMessage(role="assistant", content=response_text)
            ]
            
            # Generate conversation ID if not provided
            conversation_id = ev.conversation_id or str(uuid.uuid4())
            
            return StopEvent(
                result=ChatResponse(
                    response=response_text,
                    conversation_id=conversation_id,
                    history=updated_history
                )
            )
            
        except Exception as e:
            print(f"Error in LLM synthesis: {e}")
            error_response = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            
            updated_history = ev.chat_history + [
                ChatMessage(role="user", content=ev.query),
                ChatMessage(role="assistant", content=error_response)
            ]
            
            conversation_id = ev.conversation_id or str(uuid.uuid4())
            
            return StopEvent(
                result=ChatResponse(
                    response=error_response,
                    conversation_id=conversation_id,
                    history=updated_history
                )
            )
    
    @step
    async def web_search_step(self, ev: WebSearchEvent) -> StopEvent:
        """
        Step 4b: Perform web search when insufficient context available
        """
        print("Performing web search due to insufficient retrieved context")
        
        try:
            # Convert chat history to the format expected by web search agent
            web_chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in ev.chat_history
            ]
            
            # Perform web search
            search_result = self.web_search_agent.web_search_synthesize(
                query=ev.query,
                chat_history=web_chat_history if web_chat_history else None
            )
            
            if search_result["success"]:
                response_text = search_result["response"]
            else:
                response_text = f"I apologize, but I couldn't find current information about your query. Error: {search_result.get('error', 'Unknown error')}"
            
            # Update chat history
            updated_history = ev.chat_history + [
                ChatMessage(role="user", content=ev.query),
                ChatMessage(role="assistant", content=response_text)
            ]
            
            # Generate conversation ID if not provided
            conversation_id = ev.conversation_id or str(uuid.uuid4())
            
            return StopEvent(
                result=ChatResponse(
                    response=response_text,
                    conversation_id=conversation_id,
                    history=updated_history
                )
            )
            
        except Exception as e:
            print(f"Error in web search: {e}")
            error_response = f"I apologize, but I encountered an error while searching for current information: {str(e)}"
            
            updated_history = ev.chat_history + [
                ChatMessage(role="user", content=ev.query),
                ChatMessage(role="assistant", content=error_response)
            ]
            
            conversation_id = ev.conversation_id or str(uuid.uuid4())
            
            return StopEvent(
                result=ChatResponse(
                    response=error_response,
                    conversation_id=conversation_id,
                    history=updated_history
                )
            )


# Usage Example
async def main():
    """Example usage of the Self-RAG Workflow"""
    
    # Initialize components (this would be done in main.py)
    print("Initializing vector store...")
    vector_store = SelfRAGVectorStore(collection_name="news_articles")
    vector_store.build_vector_index()
    
    print("Initializing web search agent...")
    web_search_agent = WebSearchAgent()
    
    # Initialize workflow with injected dependencies
    workflow = SelfRAGWorkflow(
        vector_store=vector_store,
        web_search_agent=web_search_agent,
        timeout=60
    )
    
    # Example 1: Query that should find relevant documents
    chat_request1 = ChatRequest(
        query="What are the latest developments around Hollywood?",
        history=[],
        conversation_id=None
    )
    
    print("="*50)
    print("Example 1: Query with expected retrieved context")
    print("="*50)
    
    result1 = await workflow.run(chat_request=chat_request1)
    chat_response1 = result1
    
    print(f"Response: {chat_response1.response}")
    print(f"Conversation ID: {chat_response1.conversation_id}")
    print(f"History length: {len(chat_response1.history)}")
    
    # # Example 2: Follow-up query with chat history
    # chat_request2 = ChatRequest(
    #     query="Can you tell me more about the Google Meet's Functionality specifically?",
    #     history=chat_response1.history,
    #     conversation_id=chat_response1.conversation_id
    # )
    
    # print("\n" + "="*50)
    # print("Example 2: Follow-up query with chat history")
    # print("="*50)
    
    # result2 = await workflow.run(chat_request=chat_request2)
    # chat_response2 = result2
    
    # print(f"Response: {chat_response2.response}")
    # print(f"Conversation ID: {chat_response2.conversation_id}")
    # print(f"History length: {len(chat_response2.history)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())