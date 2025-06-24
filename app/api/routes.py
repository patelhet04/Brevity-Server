from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from app.db import article_crud
import logging

from app.services.article_service import (
    process_and_store_news,
    get_articles_by_date,
    get_articles_by_source,
    get_articles_general
)
from app.schemas.articles import (
    ArticleListResponse,
    SuccessResponse
)
from pydantic import BaseModel
from typing import Optional


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/articles", tags=["articles"])

# ================================
# API 1: PROCESS AND STORE
# ================================


@router.post("/process-and-store", response_model=SuccessResponse)
async def process_and_store_articles():
    """
    API 1: Complete pipeline - Fetch news → Summarize → Store in DynamoDB

    This endpoint:
    1. Fetches articles from NewsAPI
    2. Extracts full content
    3. Generates AI summaries
    4. Transforms data for DynamoDB
    5. Stores in your ArticleSummaries table

    Returns: Success message with processing statistics
    """
    try:
        logger.info("Starting article processing pipeline...")

        # Call your service function
        result = await process_and_store_news()

        if result["success"]:
            return SuccessResponse(
                message=f"Successfully processed {result['articles_stored']} articles",
                # You can add additional data here if needed
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Processing failed")
            )

    except Exception as e:
        logger.error(f"Error in process_and_store_articles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# ================================
# API 2: RETRIEVE ARTICLES
# ================================


@router.get("/", response_model=ArticleListResponse)
async def get_articles(
    # Query parameters
    date: Optional[str] = Query(
        None, description="Filter by date (YYYY-MM-DD)", regex=r'^\d{4}-\d{2}-\d{2}$'),
    source_name: Optional[str] = Query(
        None, description="Filter by news source"),
    limit: int = Query(
        50, ge=1, le=100, description="Number of articles to return"),
    # Changed from last_key to cursor
    cursor: Optional[str] = Query(
        None, description="Pagination cursor (base64 encoded)")
):
    """
    API 2: Retrieve articles with filtering options

    Pagination: Use the 'cursor' from the previous response to get the next page
    """
    try:
        # Route to appropriate service function
        if date:
            logger.info(f"Fetching articles by date: {date}")
            result = await get_articles_by_date(date, limit, cursor)

        elif source_name:
            logger.info(f"Fetching articles by source: {source_name}")
            result = await get_articles_by_source(source_name, limit, cursor)

        else:
            logger.info("Fetching articles with general pagination")
            result = await get_articles_general(limit, cursor)

        if result["success"]:
            return ArticleListResponse(
                articles=result["articles"],
                count=result["count"],
                # This is now base64 string
                last_evaluated_key=result.get("last_evaluated_key"),
                has_more=result.get("has_more", False)
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to retrieve articles")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_articles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# ================================
# SPECIFIC ENDPOINT ALTERNATIVES
# ================================


@router.get("/by-date/{date}", response_model=ArticleListResponse)
async def get_articles_by_date_endpoint(
    date: str,
    limit: int = Query(50, ge=1, le=100),
    sort_order: str = Query("desc", regex="^(asc|desc)$")
):
    """
    Alternative endpoint: Get articles by specific date
    Uses DateIndex GSI for efficient querying
    """
    try:
        result = await get_articles_by_date(date, limit)

        if result["success"]:
            return ArticleListResponse(
                articles=result["articles"],
                count=result["count"],
                last_evaluated_key=result.get("last_evaluated_key"),
                has_more=result.get("has_more", False)
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_articles_by_date_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-source/{source_name}", response_model=ArticleListResponse)
async def get_articles_by_source_endpoint(
    source_name: str,
    limit: int = Query(50, ge=1, le=100)
):
    """
    Alternative endpoint: Get articles by news source
    Uses SourceIndex GSI for efficient querying
    """
    try:
        result = await get_articles_by_source(source_name, limit)

        if result["success"]:
            return ArticleListResponse(
                articles=result["articles"],
                count=result["count"],
                last_evaluated_key=result.get("last_evaluated_key"),
                has_more=result.get("has_more", False)
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_articles_by_source_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# HEALTH CHECK
# ================================


@router.get("/health")
async def health_check():
    """Simple health check for the articles API"""
    return {"status": "healthy", "service": "articles_api"}

# ================================
# EXPORT ROUTERS
# ================================

# Export router to be included in main.py
__all__ = ['router']
