from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict
from typing import Optional, List
from datetime import datetime


class ArticleBase(BaseModel):
    """Base article schema matching your storage format"""
    url: HttpUrl = Field(..., description="Article URL")
    published_date: datetime = Field(...,
                                     description="Article publication timestamp")
    title: str = Field(..., min_length=1, max_length=500,
                       description="Article title")
    author: Optional[str] = Field(
        None, max_length=200, description="Article author")
    source_name: str = Field(..., min_length=1,
                             max_length=100, description="News source name")
    source_id: str = Field(..., min_length=1, max_length=100,
                           description="News source identifier")
    summary: str = Field(..., min_length=10,
                         description="AI-generated article summary")
    summary_length: int = Field(..., ge=0,
                                description="Length of generated summary")
    category: str = Field(...,
                          description="Article category from NewsAPI sources")

    @field_validator('summary_length')
    @classmethod
    def validate_summary_length(cls, v: int, info) -> int:
        """Ensure summary_length matches actual summary length"""
        if 'summary' in info.data and info.data['summary']:
            actual_length = len(info.data['summary'])
            if v != actual_length:
                return actual_length  # Auto-correct the length
        return v


class ArticleResponse(ArticleBase):
    """Complete article response - matches your storage exactly"""
    created_at: datetime = Field(..., description="When summary was created")
    ttl: Optional[int] = Field(
        None, description="Unix timestamp for auto-deletion")

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class ArticleCreateRequest(BaseModel):
    """Schema for manual article creation"""
    url: HttpUrl
    published_date: datetime
    title: str = Field(..., min_length=1, max_length=500)
    author: Optional[str] = Field(None, max_length=200)
    source_name: str = Field(..., min_length=1, max_length=100)
    source_id: str = Field(..., min_length=1, max_length=50)
    summary: str = Field(..., min_length=10)

    # Note: summary_length will be auto-calculated in services layer
    # Note: created_at and ttl will be auto-generated in services layer


class ArticleSummary(BaseModel):
    """Minimal article info for list responses"""
    url: HttpUrl
    title: str
    source_name: str
    published_date: datetime
    summary_length: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SortOrder(str, Enum):
    """Sort order options"""
    ASC = "asc"
    DESC = "desc"


class ArticleQueryParams(BaseModel):
    """Query parameters for filtering articles"""
    # Pagination
    limit: int = Field(default=50, ge=1, le=100,
                       description="Number of articles to return")
    last_evaluated_key: Optional[str] = Field(
        None, description="Pagination token")

    # Sorting
    sort_order: SortOrder = Field(
        default=SortOrder.DESC, description="Sort order for results")

    # Date filtering
    date: Optional[str] = Field(
        None, pattern=r'^\d{4}-\d{2}-\d{2}$', description="Filter by date (YYYY-MM-DD)")
    start_date: Optional[datetime] = Field(
        None, description="Filter from this date")
    end_date: Optional[datetime] = Field(
        None, description="Filter until this date")

    # Source filtering
    source_name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Filter by news source")
    source_id: Optional[str] = Field(
        None, min_length=1, max_length=50, description="Filter by source ID")

    # Text search (for future implementation)
    search_text: Optional[str] = Field(
        None, min_length=1, max_length=200, description="Search in title/summary")

    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure end_date is after start_date"""
        if v and 'start_date' in info.data and info.data['start_date']:
            if v <= info.data['start_date']:
                raise ValueError('end_date must be after start_date')
        return v


class SourceQueryParams(BaseModel):
    """Query parameters specific to source-based filtering"""
    source_name: str = Field(..., min_length=1,
                             max_length=100, description="News source name")
    limit: int = Field(default=50, ge=1, le=100,
                       description="Number of articles to return")
    start_date: Optional[datetime] = Field(
        None, description="Filter from this date")
    end_date: Optional[datetime] = Field(
        None, description="Filter until this date")
    sort_order: SortOrder = Field(
        default=SortOrder.DESC, description="Sort order")


class DateQueryParams(BaseModel):
    """Query parameters specific to date-based filtering"""
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$',
                      description="Filter by date (YYYY-MM-DD)")
    limit: int = Field(default=50, ge=1, le=100,
                       description="Number of articles to return")
    sort_order: SortOrder = Field(
        default=SortOrder.DESC, description="Sort order")


class ArticleListResponse(BaseModel):
    """Response schema for multiple articles"""
    articles: List[ArticleResponse] = Field(...,
                                            description="List of articles")
    count: int = Field(..., ge=0,
                       description="Number of articles in this response")
    # total_count: Optional[int] = Field(
    #     None, ge=0, description="Total available articles (if known)")
    last_evaluated_key: Optional[str] = Field(
        None, description="Pagination token for next page")
    has_more: bool = Field(...,
                           description="Whether more results are available")

    model_config = ConfigDict(from_attributes=True)


class ArticleSummaryListResponse(BaseModel):
    """Response schema for article summaries (lighter payload)"""
    articles: List[ArticleSummary] = Field(...,
                                           description="List of article summaries")
    count: int = Field(..., ge=0,
                       description="Number of articles in this response")
    last_evaluated_key: Optional[str] = Field(
        None, description="Pagination token for next page")
    has_more: bool = Field(...,
                           description="Whether more results are available")

    model_config = ConfigDict(from_attributes=True)


class ArticleStatsResponse(BaseModel):
    """Response schema for article statistics"""
    total_articles: int = Field(..., ge=0,
                                description="Total number of articles")
    sources_count: int = Field(..., ge=0,
                               description="Number of unique sources")
    latest_article_date: Optional[datetime] = Field(
        None, description="Most recent article date")
    oldest_article_date: Optional[datetime] = Field(
        None, description="Oldest article date")

    model_config = ConfigDict(from_attributes=True)


class ErrorDetail(BaseModel):
    """Individual error detail"""
    field: Optional[str] = Field(
        None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Specific error code")


class ErrorResponse(BaseModel):
    """Standard error response format"""
    success: bool = Field(
        default=False, description="Operation success status")
    message: str = Field(..., description="Main error message")
    details: Optional[List[ErrorDetail]] = Field(
        None, description="Detailed error information")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp")

    model_config = ConfigDict(from_attributes=True)


class SuccessResponse(BaseModel):
    """Standard success response format"""
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(..., description="Success message")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp")

    model_config = ConfigDict(from_attributes=True)


class ChatMessage(BaseModel):
    role: str = Field(...,
                      description="Role of the message sender (e.g., 'user' or 'assistant')")
    content: str = Field(..., description="Content of the message")


class ChatTurnRequest(BaseModel):
    query: str = Field(..., description="The current user's message/query")
    history: List[ChatMessage] = Field(
        [], description="Previous messages in the conversation")
    conversation_id: Optional[str] = Field(
        None, description="Unique ID for the conversation thread")


class ChatTurnResponse(BaseModel):
    response: str = Field(..., description="The chatbot's generated response")
    conversation_id: str = Field(...,
                                 description="Unique ID for the conversation thread")
    history: List[ChatMessage] = Field(...,
                                       description="Updated conversation history")


__all__ = [
    # Base schemas
    'ArticleBase',
    'ArticleResponse',
    'ArticleCreateRequest',
    'ArticleUpdateRequest',
    'ArticleSummary',

    # Query schemas
    'ArticleQueryParams',
    'SourceQueryParams',
    'DateQueryParams',
    'SortOrder',

    # Response schemas
    'ArticleListResponse',
    'ArticleSummaryListResponse',
    'ArticleStatsResponse',

    # Utility schemas
    'ErrorResponse',
    'ErrorDetail',
    'SuccessResponse'
]
