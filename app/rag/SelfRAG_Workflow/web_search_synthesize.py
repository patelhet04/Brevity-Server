import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama


class WebSearchAgent:
    def __init__(self, llm_model: str = "gemma2:2B"):
        """
        Initialize the Web Search Agent
        
        Args:
            tavily_api_key: API key for Tavily search
            llm_model: LLM model to use (default: gemma3:1B via Ollama)
        """
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.llm_model = llm_model
        
        # Initialize components
        self.tavily_tool = None
        self.llm = None
        self.agent = None
        
        self._setup_tools()
        self._setup_llm()
        self._setup_agent()
    
    def _setup_tools(self):
        """Setup Tavily search tools"""
        try:
            # Initialize Tavily tool
            tavily_tool_spec = TavilyToolSpec(api_key=self.tavily_api_key)
            
            # Get the search tool from TavilyToolSpec
            self.tavily_tools = tavily_tool_spec.to_tool_list()
            
            print(f"Tavily tools initialized: {len(self.tavily_tools)} tools available")
            
        except Exception as e:
            print(f"Error setting up Tavily tools: {e}")
            raise
    
    def _setup_llm(self):
        """Setup the LLM (using Ollama as free alternative)"""
        try:
            # Using Ollama with Llama 3.1 (free, local)
            self.llm = Ollama(
                model=self.llm_model,
                request_timeout=120.0,
                temperature=0.5,
                num_predict=512,  # Limit response length
                num_ctx=4096    # High for diversity
            )
            
            print(f"LLM initialized: {self.llm_model}")
            
        except Exception as e:
            print(f"Error setting up LLM: {e}")
            # Fallback to a simpler model if needed
            try:
                self.llm = Ollama(model=self.llm_model, request_timeout=120.0, temperature=0.6, num_predict=256, num_ctx=4096)
                print("Fallback to gemma2:2B model")
            except:
                raise Exception("Unable to initialize any LLM model. Please ensure Ollama is running with gemma2:2B")
    
    def _setup_agent(self):
        """Setup the ReAct agent with tools and LLM"""
        try:
            # Create ReAct agent
            self.agent = ReActAgent.from_tools(
                tools=self.tavily_tools,
                llm=self.llm,
                verbose=True,
                max_iterations=5
            )
            
            print("ReAct agent initialized successfully")
            
        except Exception as e:
            print(f"Error setting up agent: {e}")
            raise
    
    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Format chat history for context
        
        Args:
            chat_history: List of chat messages with role and content
            
        Returns:
            Formatted chat history string
        """
        if not chat_history:
            return ""
        
        formatted_history = "Previous conversation context:\n"
        for message in chat_history[-6:]:  # Last 6 messages for context
            role = message.get("role", "")
            content = message.get("content", "")
            formatted_history += f"{role.capitalize()}: {content}\n"
        
        return formatted_history + "\n"
    
    def web_search_synthesize(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Search the web and synthesize results for a user query.
        
        Args:
            query: User's search query
            chat_history: Optional chat history for context
            
        Returns:
            Dictionary containing search results and synthesized response
        """
        try:
            # Format chat history if provided
            context = self._format_chat_history(chat_history) if chat_history else ""
            
            # Create enhanced prompt with context
            enhanced_query = f"""{context}Current question: {query}

Please search the web for current, relevant information to answer this question. 
Get only 6 results from web search. Do not call the web search tool again and again if it returned some result in the first attempt.
Provide a comprehensive and accurate response based on the search results. Do not make up answers and stick to the search results.
If the question relates to the previous conversation, consider that context in your response."""
            
            # Use agent to search and synthesize
            print(f"Processing query: {query}")
            response = self.agent.chat(enhanced_query)
            
            # print("Agent response:", response)
            
            return {
                "success": True,
                "query": query,
                "response": str(response),
                "has_context": bool(chat_history)
            }
            
        except Exception as e:
            print(f"Error in web search synthesis: {e}")
            return {
                "success": False,
                "query": query,
                "response": f"Error occurred during web search: {str(e)}",
                "has_context": bool(chat_history),
                "error": str(e)
            }


# Example usage and testing
if __name__ == "__main__":
    
    try:
        # Initialize the web search agent
        web_agent = WebSearchAgent()
        
        # Example 1: Simple search without context
        result1 = web_agent.web_search_synthesize("What are the latest developments in hollywood?")
        print("Search Result 1:")
        print(result1)
        # print("\n" + "="*50 + "\n")
        
        # # Example 2: Search with chat history context
        # chat_history = [
        #     {
        #         "role": "user",
        #         "content": "Tell me about recent AI breakthroughs"
        #     },
        #     {
        #         "role": "assistant", 
        #         "content": "Recent AI breakthroughs include advances in large language models, computer vision, and robotics..."
        #     }
        # ]
        
        # result2 = web_agent.web_search_synthesize(
        #     "How do these breakthroughs compare to previous years?", 
        #     chat_history=chat_history
        # )
        # print("Search Result 2 (with context):")
        # print(result2)
        
    except Exception as e:
        print(f"Error running example: {e}")
        print("Make sure to:")
        print("1. Install required packages: pip install llama-index-tools-tavily llama-index-llms-ollama")
        print("2. Have Ollama running locally with llama3.1 or llama3.2 model")
        print("3. Set your actual Tavily API key")