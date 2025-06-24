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
from llama_index.llms.openai import OpenAI
from app.rag.SelfRAG_Workflow.vector_store_and_retrieval import SelfRAGVectorStore
from app.rag.SelfRAG_Workflow.web_search_synthesize import WebSearchAgent
from app.config import settings
import logging

logger = logging.getLogger(__name__)


# Uncomment this when runnning just this file to test the workflow
# import dotenv
# import os
# dotenv.load_dotenv()
# openai_api_key = os.getenv("OPENAI_KEY")
# project_root = Path(__file__).parent.parent.parent.parent
# sys.path.append(str(project_root))


from app.schemas.articles import ChatMessage, ChatResponse, ChatRequest


# Pydantic model for structured hallucination checking output
class HallucinationCheckResult(BaseModel):
    is_hallucinated: bool = Field(..., description="Whether the response contains hallucinations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level of the assessment (0.0 to 1.0)")
    reasoning: str = Field(..., description="Brief explanation of the decision")
    unsupported_claims: List[str] = Field(default=[], description="List of specific unsupported claims found")


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
    retry_count: int = 0


class WebSearchEvent(Event):
    """Event to trigger web search when insufficient context"""
    query: str
    chat_history: List[ChatMessage]
    conversation_id: Optional[str]


class HallucinationCheckEvent(Event):
    """Event to trigger hallucination checking of generated response"""
    query: str
    chat_history: List[ChatMessage]
    conversation_id: Optional[str]
    retrieved_nodes: List[Any]
    generated_response: str
    retry_count: int = 0


class SelfRAGWorkflow(Workflow):
    """
    Self-RAG Workflow Implementation
    
    Flow:
    1. RetrieverEvent -> Query documents from vector store
    2. QueryRouterEvent -> Route based on number of retrieved documents
    3a. LLMSynthesizerEvent -> Generate response using retrieved context
    4. HallucinationCheckEvent -> Check if response is grounded in facts
    5a. If hallucination detected -> Return to LLMSynthesizerEvent (retry)
    5b. If grounded -> StopEvent (return response)
    3b. WebSearchEvent -> Search web when insufficient context -> StopEvent
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
            request_timeout=60.0,
            temperature=0.7,
            num_predict=512,
            num_ctx=4096
        )
        
        # Initialize separate LLM for hallucination checking

        # self.hallucination_checker_llm = self.web_search_agent.llm
        # self.hallucination_checker_llm = Ollama(
        #     model="llama3.2:3b",
        #     request_timeout=60.0,
        #     temperature=0.1,  # Low temperature for consistent checking
        #     num_predict=256,
        #     num_ctx=4096
        # )

        self.hallucination_checker_llm = OpenAI(
            model="gpt-4o-mini",                # You can change this to gpt-4, gpt-3.5-turbo, etc.
            temperature=0.1,                    # Low temperature for consistent checking
            max_tokens=512,
            api_key=settings.openai_key
        )
        logger.info("✅ Self-RAG Workflow initialized with injected dependencies")
    
    @step
    async def retriever_step(self, ev: StartEvent) -> RetrieverEvent:
        """
        Step 1: Extract data from ChatRequest and trigger retrieval
        """
        chat_request = ev.get("chat_request")
        
        logger.info(f"🔎 Starting retrieval for query: {chat_request.query}")
        
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
        logger.info(f"📃 Retrieving documents for query: {ev.query}")
        
        try:
            # Retrieve and filter nodes
            retrieved_nodes = self.vector_store.retrieve_and_filter_nodes(
                query=ev.query,
                similarity_threshold=0.6
            )
            
            logger.info(f"✅ Retrieved {len(retrieved_nodes)} relevant documents")
            
            return QueryRouterEvent(
                query=ev.query,
                chat_history=ev.chat_history,
                conversation_id=ev.conversation_id,
                retrieved_nodes=retrieved_nodes
            )
            
        except Exception as e:
            logger.error(f"‼️ Error during retrieval: {e}")
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
        logger.info(f"ℹ️ Query router: Found {num_documents} documents")
        
        if num_documents >= 3:
            ev.retrieved_nodes = ev.retrieved_nodes[:2] if len(ev.retrieved_nodes) > 2 else ev.retrieved_nodes
            logger.info("Sufficient context found. Routing to LLM Synthesizer")
            return LLMSynthesizerEvent(
                query=ev.query,
                chat_history=ev.chat_history,
                conversation_id=ev.conversation_id,
                retrieved_nodes=ev.retrieved_nodes,
                retry_count=0
            )
        else:
            logger.info("❌ Insufficient context. Routing to Web Search")
            return WebSearchEvent(
                query=ev.query,
                chat_history=ev.chat_history,
                conversation_id=ev.conversation_id
            )
    
    @step
    async def llm_synthesizer(self, ev: LLMSynthesizerEvent) -> HallucinationCheckEvent:
        """
        Step 4a: Generate response using LLM with retrieved context
        """
        logger.info("💭 Synthesizing response using retrieved context")
        
        try:
            # Format retrieved context
            context_text = "\n\n".join([
                f"Document {i+1}:\nTitle: {node.metadata.get('title', 'N/A')}\nContent: {node.text}"
                for i, node in enumerate(ev.retrieved_nodes)
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
            
            logger.info(f"☑️ Generated response, now checking for hallucinations...")
            
            # Send to hallucination checker instead of returning StopEvent
            return HallucinationCheckEvent(
                query=ev.query,
                chat_history=ev.chat_history,
                conversation_id=ev.conversation_id,
                retrieved_nodes=ev.retrieved_nodes,
                generated_response=response_text,
                retry_count=ev.retry_count
            )
            
        except Exception as e:
            logger.error(f"‼️ Error in LLM synthesis: {e}")
            error_response = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            
            # Even error responses should go through hallucination check for consistency
            return HallucinationCheckEvent(
                query=ev.query,
                chat_history=ev.chat_history,
                conversation_id=ev.conversation_id,
                retrieved_nodes=ev.retrieved_nodes,
                generated_response=error_response,
                retry_count=ev.retry_count
            )
    
    @step
    async def hallucination_checker(self, ev: HallucinationCheckEvent) -> LLMSynthesizerEvent | StopEvent | WebSearchEvent:
        """
        Step 5: Check if generated response is grounded in retrieved facts using LLM Call and parsing the JSON response.
        """
        logger.info(f"🕵️ Checking for hallucinations (attempt {ev.retry_count + 1})")
        
        # Maximum retry limit to prevent infinite loops
        MAX_RETRIES = 2
        
        try:
            # Format retrieved facts for checking
            facts_text = "\n".join([
                f"Fact {i+1}: {node.text}..."  # Limit fact length for efficiency
                for i, node in enumerate(ev.retrieved_nodes)
            ])

            # Get structured hallucination check result
            # Use complete instead of structured_predict for better Ollama compatibility
            structured_prompt = f"""You are a fact-checker. Your task is to determine if a generated response is supported by the provided facts.

FACTS (Source Material):
{facts_text}

GENERATED RESPONSE:
{ev.generated_response}

USER QUESTION:
{ev.query}

Analyze whether the generated response is well-grounded in the provided facts. Look for:
1. Claims made without factual support
2. Information that contradicts the facts
3. Details that seem invented or not present in the source material

You MUST respond with ONLY a valid JSON object in this exact format:
{{
    "is_hallucinated": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of your decision",
    "unsupported_claims": ["list", "of", "specific", "unsupported", "claims"]
}}

JSON Response:"""

            # Use regular completion with JSON parsing for better compatibility
            check_response = await self.hallucination_checker_llm.acomplete(structured_prompt)
            check_result_text = str(check_response).strip()
            
            # Parse JSON response
            import json
            try:
                # Extract JSON from response
                start_idx = check_result_text.find('{')
                end_idx = check_result_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = check_result_text[start_idx:end_idx]
                    logger.info(f"JSON response: {json_text}")
                    parsed_result = json.loads(json_text)
                    
                    check_result = HallucinationCheckResult(
                        is_hallucinated=parsed_result.get("is_hallucinated", False),
                        confidence=float(parsed_result.get("confidence", 0.5)),
                        reasoning=parsed_result.get("reasoning", "No reasoning provided"),
                        unsupported_claims=parsed_result.get("unsupported_claims", [])
                    )
                else:
                    raise json.JSONDecodeError("No valid JSON found", check_result_text, 0)
                    
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"‼️ Failed to parse hallucination check JSON: {e}")
                logger.error(f"Raw response: {check_result_text}")
                # Default to assuming no hallucination if parsing fails
                check_result = HallucinationCheckResult(
                    is_hallucinated=False,
                    confidence=0.5,
                    reasoning="Failed to parse checker response",
                    unsupported_claims=[]
                )
            
            logger.info(f"ℹ️ Hallucination check result: {check_result.is_hallucinated} (confidence: {check_result.confidence})")
            logger.info(f"Reasoning: {check_result.reasoning}")
            if check_result.unsupported_claims:
                logger.info(f"Unsupported claims: {check_result.unsupported_claims}")
            
            # Decision making
            if check_result.is_hallucinated and check_result.confidence > 0.7 and ev.retry_count < MAX_RETRIES:
                logger.info(f"😵‍💫 Hallucination detected with high confidence. Retrying synthesis (attempt {ev.retry_count + 1}/{MAX_RETRIES})")
                
                # Return to LLM synthesizer for retry with updated retry count
                return LLMSynthesizerEvent(
                    query=ev.query,
                    chat_history=ev.chat_history,
                    conversation_id=ev.conversation_id,
                    retrieved_nodes=ev.retrieved_nodes,
                    retry_count=ev.retry_count + 1
                )
                
            else:
                # Either no hallucination detected, low confidence, or max retries reached
                if ev.retry_count >= MAX_RETRIES:
                    logger.info(f"😵 Max retries ({MAX_RETRIES}) reached. Proceeding with web search...")

                    return WebSearchEvent(
                        query=ev.query,
                        chat_history=ev.chat_history,
                        conversation_id=ev.conversation_id,
                    )
                else:
                    logger.info("✅ Response appears to be grounded. Proceeding to final output.")
                
                # Update chat history
                updated_history = ev.chat_history + [
                    ChatMessage(role="user", content=ev.query),
                    ChatMessage(role="assistant", content=ev.generated_response)
                ]
                
                # Generate conversation ID if not provided
                conversation_id = ev.conversation_id or str(uuid.uuid4())
                
                return StopEvent(
                    result=ChatResponse(
                        response=ev.generated_response,
                        conversation_id=conversation_id,
                        history=updated_history
                    )
                )
                
        except Exception as e:
            logger.error(f"‼️ Error in hallucination checking: {e}")
            logger.info("Proceeding with original response due to checker error.")
            
            # On error, proceed with original response
            updated_history = ev.chat_history + [
                ChatMessage(role="user", content=ev.query),
                ChatMessage(role="assistant", content=ev.generated_response)
            ]
            
            conversation_id = ev.conversation_id or str(uuid.uuid4())
            
            return StopEvent(
                result=ChatResponse(
                    response=ev.generated_response,
                    conversation_id=conversation_id,
                    history=updated_history
                )
            )
    
    @step
    async def web_search_step(self, ev: WebSearchEvent) -> StopEvent:
        """
        Step 4b: Perform web search when insufficient context available
        """
        logger.info("🕸️ Performing web search due to insufficient retrieved context")
        
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
            logger.error(f"‼️ Error in web search: {e}")
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


# Usage Example. Ran only when this is file is run directly to test the workflow
async def main():
    """Example usage of the Self-RAG Workflow"""
    
    # Initialize components (this would be done in main.py)
    logger.info("Initializing vector store...")
    vector_store = SelfRAGVectorStore(collection_name="news_articles")
    vector_store.build_vector_index()
    
    logger.info("Initializing web search agent...")
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
    
    logger.info("="*50)
    logger.info("Example 1: Query with expected retrieved context")
    logger.info("="*50)
    
    result1 = await workflow.run(chat_request=chat_request1)
    chat_response1 = result1
    
    logger.info(f"Response: {chat_response1.response}")
    logger.info(f"Conversation ID: {chat_response1.conversation_id}")
    logger.info(f"History length: {len(chat_response1.history)}")

    # # Example 2: Follow-up query with chat history
    # chat_request2 = ChatRequest(
    #     query="Can you tell me more about the Google Meet's Functionality specifically?",
    #     history=chat_response1.history,
    #     conversation_id=chat_response1.conversation_id
    # )
    
    # logger.info("\n" + "="*50)
    # logger.info("Example 2: Follow-up query with chat history")
    # logger.info("="*50)
    
    # result2 = await workflow.run(chat_request=chat_request2)
    # chat_response2 = result2
    
    # logger.info(f"Response: {chat_response2.response}")
    # logger.info(f"Conversation ID: {chat_response2.conversation_id}")
    # logger.info(f"History length: {len(chat_response2.history)}")
 

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())