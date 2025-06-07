import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from app.db import article_crud
from app.schemas.articles import ArticleResponse, ArticleListResponse
from utils.news_fetcher import fetch_news_enhanced
from utils.summarizer import process_articles
import logging
import re
import json
import base64
from utils.news_fetcher import get_category_from_source
from datetime import datetime, timedelta, timezone
logger = logging.getLogger(__name__)

# ================================
# DATA TRANSFORMATION FUNCTIONS
# ================================


def clean_source_id(source_id: str) -> str:
    """Clean source_id to be database and URL friendly"""
    if not source_id:
        return "unknown"

    # Replace special characters with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', source_id)

    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)

    # Truncate to 50 characters
    if len(cleaned) > 50:
        cleaned = cleaned[:50].rstrip('_')

    return cleaned.lower()


def clean_text_field(text: str, max_length: int = None) -> str:
    """Clean text fields for database storage"""
    if not text:
        return ""

    # Remove extra whitespace
    cleaned = ' '.join(text.split())

    # Truncate if needed
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()

    return cleaned


def transform_summarized_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Transform article from summarizer format to DynamoDB storage format"""
    try:
        # Extract and validate URL
        url = article.get('url', '').strip()
        if not url:
            logger.warning("Article missing URL, skipping")
            return None

        # Extract source information
        source = article.get('source', {})
        source_name = source.get('name', 'Unknown')
        raw_source_id = source.get(
            'id') or source_name.lower().replace(' ', '_')

        # Clean the source_id (your existing logic)
        source_id = raw_source_id[:100] if len(
            raw_source_id) > 100 else raw_source_id

        # Get category from source mapping
        category = get_category_from_source(source.get('id'))  # Add this line

        # Clean text fields (your existing logic)
        title = article.get('title', '').strip()
        if not title:
            logger.warning(f"Article missing title: {url}")
            title = "Untitled Article"

        author = article.get('author')
        if author:
            author = author.strip()[:200] if len(
                author) > 200 else author.strip()
            if not author:
                author = None

        # Clean and validate summary
        summary = article.get('summary', '').strip()
        if not summary:
            logger.warning(f"Article missing summary: {url}")
            return None

        # Calculate summary length
        summary_length = len(summary)

        # Parse published date (your existing logic)
        published_date_str = article.get(
            'publishedAt') or article.get('published_date')
        if published_date_str:
            try:
                if isinstance(published_date_str, str):
                    if 'T' in published_date_str:
                        if published_date_str.endswith('Z'):
                            published_date_str = published_date_str[:-1] + '+00:00'
                        published_date = datetime.fromisoformat(
                            published_date_str)
                    else:
                        published_date = datetime.fromisoformat(
                            published_date_str)
                else:
                    published_date = published_date_str
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid date format for {url}: {published_date_str}, using current time")
                published_date = datetime.now(timezone.utc)
        else:
            published_date = datetime.now(timezone.utc)

        # Calculate TTL (30 days from now)
        ttl = int((datetime.now(timezone.utc) +
                  timedelta(days=30)).timestamp())

        # Create current timestamp
        created_at = datetime.now(timezone.utc)

        # Create DynamoDB format
        transformed_article = {
            "url": url,
            "published_date": published_date.isoformat(),
            "title": title,
            "author": author,
            "source_name": source_name,
            "source_id": source_id,
            "summary": summary,
            "summary_length": summary_length,
            "category": category,  # Add this line
            "created_at": created_at.isoformat(),
            "ttl": ttl,
            "published_date_key": published_date.strftime('%Y-%m-%d')
        }

        return transformed_article

    except Exception as e:
        logger.error(f"Error transforming article: {e}")
        return None


def transform_summarized_articles_batch(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform multiple articles from summarizer to DynamoDB format"""
    transformed_articles = []

    for article in articles:
        transformed = transform_summarized_article(article)
        # Only include valid articles with URLs
        if transformed and transformed.get('url'):
            transformed_articles.append(transformed)

    logger.info(
        f"Transformed {len(transformed_articles)} out of {len(articles)} articles")
    return transformed_articles

# ================================
# MAIN SERVICE FUNCTIONS
# ================================


async def process_and_store_news() -> Dict[str, Any]:
    """
    Complete pipeline: Fetch → Summarize → Transform → Store
    This is your main API 1 function
    """
    try:
        logger.info("Starting news processing pipeline...")

        # Step 1: Fetch enhanced articles
        logger.info("Fetching articles from NewsAPI...")
        news_result = await fetch_news_enhanced()
        raw_articles = news_result.get('articles', [])

        if not raw_articles:
            return {
                "success": False,
                "message": "No articles fetched from NewsAPI",
                "articles_processed": 0
            }

        logger.info(f"Fetched {len(raw_articles)} articles")

        # Step 2: Summarize articles
        logger.info("Summarizing articles...")
        summarized_articles = await process_articles(raw_articles, concurrency=3)

        # Step 3: Transform for DynamoDB
        logger.info("Transforming articles for storage...")
        db_ready_articles = transform_summarized_articles_batch(
            summarized_articles)

        if not db_ready_articles:
            return {
                "success": False,
                "message": "No valid articles after transformation",
                "articles_processed": 0
            }

        # Step 4: Store in DynamoDB
        logger.info(
            f"Storing {len(db_ready_articles)} articles in DynamoDB...")
        storage_result = await store_articles_batch(db_ready_articles)

        if storage_result["success"]:
            return {
                "success": True,
                "message": "Articles processed and stored successfully",
                "articles_fetched": len(raw_articles),
                "articles_summarized": len(summarized_articles),
                "articles_stored": len(db_ready_articles),
                "storage_details": storage_result
            }
        else:
            return {
                "success": False,
                "message": "Failed to store articles",
                "error": storage_result.get("error"),
                "articles_processed": len(db_ready_articles)
            }

    except Exception as e:
        logger.error(f"Error in process_and_store_news: {e}")
        return {
            "success": False,
            "message": f"Pipeline failed: {str(e)}",
            "articles_processed": 0
        }


async def store_articles_batch(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Store multiple articles in DynamoDB using batch operation"""
    try:
        # Use the batch_put_articles from your CRUD layer
        result = article_crud.batch_put_articles(articles)

        if result["success"]:
            logger.info(f"Successfully stored {len(articles)} articles")
            return {
                "success": True,
                "articles_stored": len(articles),
                "message": "Batch storage successful"
            }
        else:
            logger.error(f"Batch storage failed: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error"),
                "articles_attempted": len(articles)
            }

    except Exception as e:
        logger.error(f"Error in store_articles_batch: {e}")
        return {
            "success": False,
            "error": str(e),
            "articles_attempted": len(articles)
        }

# ================================
# RETRIEVAL SERVICE FUNCTIONS
# ================================


async def get_articles_by_date(date_str: str, limit: int = 50, last_key: Optional[str] = None) -> Dict[str, Any]:
    """Get articles by date using DateIndex GSI"""
    try:
        # Convert string pagination key back to dict for DynamoDB
        dynamodb_last_key = deserialize_pagination_key(last_key)

        # Pass the dict to CRUD function (you might need to update CRUD functions to accept this)
        result = article_crud.query_by_date_index(date_str, limit)

        if result["success"]:
            # Convert DynamoDB dict pagination key to base64 string
            serialized_key = serialize_pagination_key(
                result.get("last_evaluated_key"))

            return {
                "success": True,
                "articles": result["data"],
                "count": result["count"],
                "last_evaluated_key": serialized_key,
                "has_more": result.get("last_evaluated_key") is not None
            }
        else:
            return {
                "success": False,
                "error": result.get("error"),
                "articles": []
            }

    except Exception as e:
        logger.error(f"Error getting articles by date: {e}")
        return {
            "success": False,
            "error": str(e),
            "articles": []
        }


async def get_articles_by_source(source_name: str, limit: int = 50, last_key: Optional[str] = None) -> Dict[str, Any]:
    """Get articles by source using SourceIndex GSI"""
    try:
        # Convert pagination key
        dynamodb_last_key = deserialize_pagination_key(last_key)

        result = article_crud.query_by_source_index(source_name, limit)

        if result["success"]:
            # Serialize pagination key
            serialized_key = serialize_pagination_key(
                result.get("last_evaluated_key"))

            return {
                "success": True,
                "articles": result["data"],
                "count": result["count"],
                "last_evaluated_key": serialized_key,
                "has_more": result.get("last_evaluated_key") is not None
            }
        else:
            return {
                "success": False,
                "error": result.get("error"),
                "articles": []
            }

    except Exception as e:
        logger.error(f"Error getting articles by source: {e}")
        return {
            "success": False,
            "error": str(e),
            "articles": []
        }


async def get_articles_general(limit: int = 50, last_key: Optional[str] = None) -> Dict[str, Any]:
    """Get articles with general pagination"""
    try:
        # Convert pagination key for DynamoDB
        dynamodb_last_key = deserialize_pagination_key(last_key)

        result = article_crud.scan_recent_articles(limit, dynamodb_last_key)

        if result["success"]:
            # Serialize for API response
            serialized_key = serialize_pagination_key(
                result.get("last_evaluated_key"))

            return {
                "success": True,
                "articles": result["data"],
                "count": result["count"],
                "last_evaluated_key": serialized_key,
                "has_more": result.get("last_evaluated_key") is not None
            }
        else:
            return {
                "success": False,
                "error": result.get("error"),
                "articles": []
            }

    except Exception as e:
        logger.error(f"Error getting articles: {e}")
        return {
            "success": False,
            "error": str(e),
            "articles": []
        }


def serialize_pagination_key(last_key: Optional[Dict[str, Any]]) -> Optional[str]:
    """Convert DynamoDB pagination key to base64 string (industry standard)"""
    if last_key is None:
        return None

    try:
        # Convert dict to JSON string
        json_str = json.dumps(last_key, sort_keys=True, separators=(',', ':'))
        # Encode to base64
        base64_bytes = base64.b64encode(json_str.encode('utf-8'))
        return base64_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Error serializing pagination key: {e}")
        return None


def deserialize_pagination_key(key_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Convert base64 string back to DynamoDB pagination key"""
    if not key_str:
        return None

    try:
        # Decode from base64
        json_bytes = base64.b64decode(key_str.encode('utf-8'))
        json_str = json_bytes.decode('utf-8')
        # Parse JSON back to dict
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error deserializing pagination key: {e}")
        return None


# ================================
# EXPORTS
# ================================


__all__ = [
    'process_and_store_news',
    'get_articles_by_date',
    'get_articles_by_source',
    'get_articles_general',
    'store_articles_batch',
    'transform_summarized_article'
]
