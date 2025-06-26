from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from app.config import settings, validate_settings
from app.api.routes import router as articles_router
from app.db.database import health_check
from app.schemas.articles import (
    ChatRequest, ChatResponse
)
from datetime import datetime, timezone
from app.services.redis_service import redis_store
from utils.news_fetcher import initialize_sources
from app.rag.rag_manager import rag_manager
from mangum import Mangum

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format,
    filename=settings.log_file if settings.log_file else None
)
logger = logging.getLogger(__name__)


# ================================
# APPLICATION LIFECYCLE
# ================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""

    # ================================
    # STARTUP
    # ================================
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    try:
        # Validate environment configuration
        validate_settings()
        logger.info("✅ Environment configuration validated")

        # Initialize news sources
        await initialize_sources()
        logger.info("✅ News sources initialized")

        # Check database connectivity
        if health_check():
            logger.info("✅ DynamoDB connection established")
        else:
            logger.error("❌ DynamoDB connection failed")
            raise Exception("Database health check failed")

        rag_manager.initialize()
        logger.info("✅ RAG system initialized")

        logger.info("🚀 Application startup complete")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield

    # ================================
    # SHUTDOWN
    # ================================
    logger.info("🛑 Application shutdown initiated")

# ================================
# CREATE FASTAPI APP
# ================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    **Brevity** - AI-Powered News Summarization API
    
    This API provides two main functionalities:
    
    1. **Process and Store** (`POST /api/v1/articles/process-and-store`)
       - Fetches latest news articles from NewsAPI
       - Extracts full article content
       - Generates AI-powered summaries
       - Stores processed articles in DynamoDB
    
    2. **Retrieve Articles** (`GET /api/v1/articles/`)
       - Fetch stored articles with filtering options
       - Support for date-based and source-based queries
       - Pagination support for large datasets
       - Efficient querying using DynamoDB GSI indexes
    
    ## Features
    - 🤖 AI-powered summarization using DistilBART
    - 🗃️ Efficient storage with DynamoDB
    - 🔍 Advanced filtering and pagination
    - 📊 RESTful API design
    - 🚀 High-performance async processing
    
    ## Getting Started
    1. Configure your environment variables
    2. Call the process endpoint to fetch and store articles
    3. Use the retrieve endpoint to access summarized content
    """,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
    lifespan=lifespan
)

# ================================
# MIDDLEWARE CONFIGURATION
# ================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Security middleware (optional - add trusted hosts if needed)
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.yourdomain.com", "localhost"]
    )

# ================================
# CUSTOM MIDDLEWARE
# ================================


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with timing information"""
    start_time = time.time()

    # Log incoming request
    logger.info(
        f"📨 {request.method} {request.url.path} - Client: {request.client.host}")

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log response
    logger.info(
        f"📤 {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")

    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)

    return response

# ================================
# GLOBAL EXCEPTION HANDLERS
# ================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format"""
    logger.error(
        f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")

    error_response = {
        "success": False,
        "message": exc.detail,
        "details": [{
            "field": None,
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}"
        }],
        # Convert to string
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        f"Unhandled exception: {str(exc)} - Path: {request.url.path}", exc_info=True)

    error_message = "Internal server error" if settings.environment == "production" else str(
        exc)

    error_response = {
        "success": False,
        "message": error_message,
        "details": [{
            "field": None,
            "message": error_message,
            "error_code": "INTERNAL_SERVER_ERROR"
        }],
        # Convert to string
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return JSONResponse(
        status_code=500,
        content=error_response
    )

# ================================
# ROUTE REGISTRATION
# ================================

# Include article routes with API prefix
app.include_router(
    articles_router,
    prefix=settings.api_v1_prefix,
    tags=["articles"]
)


# ================================
# ROOT ENDPOINTS
# ================================


@app.get("/", tags=["root"])
async def root():
    """API root endpoint with basic information"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "environment": settings.environment,
        "docs_url": settings.docs_url,
        "endpoints": {
            "process_articles": f"{settings.api_v1_prefix}/articles/process-and-store",
            "get_articles": f"{settings.api_v1_prefix}/articles/",
            "health": "/health",
        }
    }


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint - checks database, Redis, and RAG system"""
    try:
        # Check DynamoDB connectivity
        db_healthy = health_check()

        # Check Redis connectivity
        redis_healthy = False
        redis_error = None
        try:
            redis_store.redis.ping()
            redis_healthy = True
        except Exception as e:
            redis_error = str(e)
            logger.warning(f"Redis health check failed: {e}")

        # Check RAG system
        rag_healthy = False
        rag_error = None
        try:
            # Simple check to see if RAG manager is initialized
            from app.rag.rag_manager import rag_manager
            orchestrator = rag_manager.get_orchestrator()
            rag_healthy = orchestrator is not None
        except Exception as e:
            rag_error = str(e)
            logger.warning(f"RAG health check failed: {e}")

        # Determine overall system health
        all_healthy = db_healthy and redis_healthy and rag_healthy

        health_status = {
            "status": "healthy" if all_healthy else "degraded",
            "version": settings.app_version,
            "environment": settings.environment,
            "components": {
                "database": "connected" if db_healthy else "disconnected",
                "redis": "connected" if redis_healthy else "disconnected",
                "rag_system": "initialized" if rag_healthy else "not_initialized"
            },
            "timestamp": datetime.now(timezone.utc).timestamp()
        }

        # Add error details if any component is unhealthy
        if not all_healthy:
            health_status["errors"] = {}
            if not db_healthy:
                health_status["errors"]["database"] = "Connection failed"
            if not redis_healthy:
                health_status["errors"]["redis"] = redis_error or "Connection failed"
            if not rag_healthy:
                health_status["errors"]["rag_system"] = rag_error or "Not initialized"

        # Add Redis-specific info if healthy
        if redis_healthy:
            try:
                # Get Redis info (optional additional details)
                redis_info = redis_store.redis.info('server')
                health_status["redis_info"] = {
                    "version": redis_info.get('redis_version'),
                    "uptime_seconds": redis_info.get('uptime_in_seconds'),
                    "connected_clients": redis_store.redis.info('clients').get('connected_clients', 0)
                }
            except Exception:
                # Don't fail health check if we can't get Redis info
                pass

        # Return appropriate status code
        if all_healthy:
            status_code = 200
        elif db_healthy:  # Core system (DB) works, others degraded
            status_code = 200  # Still functional
        else:
            status_code = 503  # Core system down

        return JSONResponse(content=health_status, status_code=status_code)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).timestamp()
            },
            status_code=503
        )


handler = Mangum(app)

# ================================
# DEVELOPMENT SERVER
# ================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers,
        log_level=settings.log_level.lower()
    )
