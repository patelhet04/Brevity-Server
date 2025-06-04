from fastapi import FastAPI, HTTPException
import httpx
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import aiofiles
import json
import asyncio
from newspaper import Article, Config
from typing import List, Dict, Any, Optional

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

async def extract_full_article_content(url: str) -> Dict[str, Any]:
    """
    Extract full article content using newspaper3k
    Returns dictionary with extracted content and metadata
    """
    try:
        def extract_article():
            """Synchronous function to extract article content"""
            config = Config()
            config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            config.number_threads = 1
            
            # Additional headers to look more like a real browser
            # config.headers = {
            #     #'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            #     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            #     'Accept-Language': 'en-US,en;q=0.5',
            #     'Accept-Encoding': 'gzip, deflate',
            #     'Connection': 'keep-alive',
            #     'Upgrade-Insecure-Requests': '1',
            # }
            article = Article(url, config=config)
            
            # Download and parse the article
            article.download()
            article.parse()
            
            return {
                'success': True,
                'title': article.title,
                'full_content': article.text,
                'url': article.url,
                'content_length': len(article.text),
                'extraction_method': 'newspaper3k'
            }
        
        # Run in separate thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(extract_article)

        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'full_content': '',
            'extraction_method': 'newspaper3k',
            'content_length': 0
        }

async def enhance_article_with_full_content(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance a single article with full content extraction
    """
    enhanced_article = article.copy()
    
    if article.get('url'):
        
        # Extract full content using newspaper3k
        extraction_result = await extract_full_article_content(article['url'])
        
        if extraction_result['success']:
            # Add extracted content to the article
            enhanced_article.update({
                'full_content': extraction_result['full_content'],
                'content_length': extraction_result['content_length'],
                'extraction_status': 'success'
            })
            
        else:
            # If extraction failed, use original content
            enhanced_article.update({
                'full_content': article.get('content', ''),
                'extraction_status': 'failed',
                'extraction_error': extraction_result.get('error', 'Unknown error'),
                'content_length': len(article.get('content', ''))
            })
    else:
        # No URL available
        enhanced_article.update({
            'full_content': article.get('content', ''),
            'extraction_status': 'no_url',
            'content_length': len(article.get('content', ''))
        })
    
    return enhanced_article

@app.get("/fetch")
async def fetch_news_basic():
    """Fetch top 500 news articles from the last 5 days
    Sample article/response format:
    {
    "total": 1,
    "articles": [
            {
            "source": {
                "id": null,
                "name": "Dramabeans.com"
                },
            "author": "DB Staff",
            "title": "For Eagle Brothers: Episodes 31-32 (Drama Hangout)",
            "description": "Welcome to the Drama Hangout for KBS weekender For Eagle Brothers, starring Eom Ji-won and Ahn Jae-wook. This is your place to chat about the drama as you watch. Beware of spoilers!   RELATED POSTS Eom Ji-won leads a brewery For Eagle Brothers Eom Ji-won take…",
            "url": "https://dramabeans.com/2025/05/for-eagle-brothers-episodes-31-32-drama-hangout/",
            "urlToImage": "https://d263ao8qih4miy.cloudfront.net/wp-content/uploads/2025/01/ForEsgleBrothers_reviewb.jpeg",
            "publishedAt": "2025-05-18T00:00:58Z",
            "content": "A woman who got married later in life loses her husband shortly after their marriage. However, she decides to take responsibility for her deceased husband's four adult brothers and the family brewery… [+220 chars]"
            }
        ]
    }"""
    
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

    # JSON file name for storing all the fetched articles
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"news_articles_{timestamp}.json"

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

@app.get("/fetch-news-enhanced")
async def fetch_news_enhanced():
    """Enhanced endpoint - includes full content extraction"""
    
    # First, get basic articles from NewsAPI
    basic_articles = await fetch_news_basic()
    articles_to_enhance = basic_articles['articles']
    
    enhanced_articles = []
    
    # Process articles in batches to avoid overwhelming servers
    batch_size = 5
    for i in range(0, len(articles_to_enhance), batch_size):
        batch = articles_to_enhance[i:i + batch_size]
        
        # Process batch concurrently
        tasks = [enhance_article_with_full_content(article) for article in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Error processing article: {str(result)}")
            else:
                if result.get('content_length', 0) >600 and result.get('extraction_status', "") != "failed":
                    enhanced_articles.append(result)
        
        # Small delay between batches to be respectful to servers
        await asyncio.sleep(1)
    
    return {
        "total": len(enhanced_articles),
        "enhanced_count": len([a for a in enhanced_articles if a.get('extraction_status') == 'success']),
        "articles": enhanced_articles
    }

@app.get("/fetch-and-save-enhanced")
async def fetch_and_save_enhanced():
    """Fetch enhanced articles and save to file"""
    # Get enhanced articles
    result = await fetch_news_enhanced()
    articles = result["articles"]
    
    # Save to file asynchronously
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"enhanced_news_articles_{timestamp}.json"
    
    try:
        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
            json_data = {
                "timestamp": datetime.now().isoformat(),
                "total": len(articles),
                "enhanced_count": result["enhanced_count"],
                "extraction_stats": {
                    "successful": len([a for a in articles if a.get('extraction_status') == 'success']),
                    "failed": len([a for a in articles if a.get('extraction_status') == 'failed']),
                    "no_url": len([a for a in articles if a.get('extraction_status') == 'no_url']),
                    "not_processed": len([a for a in articles if a.get('extraction_status') == 'not_processed'])
                },
                "articles": articles
            }
            
            json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
            await f.write(json_string)
        
        return {
            "message": f"Articles saved to {filename}",
            "filename": filename,
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

@app.get("/test-extraction")
async def test_extraction(url: str):
    """Test content extraction on a single URL"""
    result = await extract_full_article_content(url)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)