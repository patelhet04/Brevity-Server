from fastapi import FastAPI, HTTPException
import httpx
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    raise ValueError("NEWS API not set")


app = FastAPI(title="News Fetcher")

@app.get("/")
async def root():
    return {"message": "News Fetcher"}

@app.get("/fetch")
async def fetch_news():
    """Fetch top 500 news articles from the last 5 days"""
    
    # Calculate date from 5 days ago
    from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    # NewsAPI base URL
    base_url = "https://newsapi.org/v2/everything"
    
    # We need to paginate since NewsAPI limits to 100 articles per request
    page = 1

    params = {
                "apiKey": NEWS_API_KEY,
                "q": "news",  # General query to get broad news coverage
                "language": "en",
                "pageSize": 100,  # Maximum allowed by NewsAPI
                "page": page,
                "from": from_date,
                "sortBy": "publishedAt"  # Sort by popularity to get top news
            }
    
    # Articles storage
    articles_store = []

    async with httpx.AsyncClient() as client:
        while len(articles_store) < 500:

            #Set the updated page number
            params["page"]=page
            
            try:
                response = await client.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Add articles to our collection
                articles = data.get("articles", [])
                articles_store.extend(articles)
                
                # Check if we've reached the end of available articles and max storage capacity of storing 500 articles
                if len(articles) < 100 or len(articles_store) >= 500:
                    break
                    
                # Move to next page
                page += 1
                
            except httpx.HTTPError as e:
                raise HTTPException(status_code=e.response.status_code if hasattr(e, 'response') else 500, 
                                  detail=str(e))
    
    # Trim to max 500 articles
    return {"total": len(articles_store[:500]), "articles": articles_store[:500]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)