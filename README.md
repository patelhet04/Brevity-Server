# Brevity Server

**AI-Powered News Summarization & Smart Chat System**

Brevity automatically finds, reads, and creates short summaries of news articles from around the web. It also includes an intelligent chatbot that can answer questions about current events by searching through stored articles and the internet in real-time.

## 🎯 Project Overview

Brevity transforms the way users consume news by:

- **Automated News Processing**: Fetches articles from NewsAPI and extracts full content using Newspaper3k
- **AI-Powered Summarization**: Generates concise, high-quality summaries using AWS Bedrock foundation models
- **Intelligent Storage**: Efficiently stores processed articles in DynamoDB with GSI indexes and automatic TTL
- **Smart Retrieval**: Provides advanced filtering, pagination, and search capabilities via optimized DynamoDB queries
- **RAG-Enhanced Chat**: Offers contextual conversations using LlamaIndex orchestration with ChromaDB vector embeddings and Tavily web search
- **Scalable Architecture**: Built for high-performance async processing with FastAPI and AWS Lambda deployment

## 🏗️ System Architecture

```
                    📰 News Processing Pipeline
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   NewsAPI       │    │   Content       │    │   AWS Bedrock   │
    │   Fetcher       │──→ │   Extraction    │──→ │   Foundation    │
    │                 │    │   (Newspaper3k) │    │   Models        │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                  │                        │
                                  │                        ▼
                            ┌─────────────────┐    ┌─────────────────┐
                            │   Article       │    │   AI-Generated  │
                            │   Validation    │◄── │   Summaries     │
                            │   & Transform   │    │   & Embeddings  │
                            └─────────────────┘    └─────────────────┘
                                  │
                                  ▼
                    💾 Storage & Retrieval Layer
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   DynamoDB      │◄── │   FastAPI       │──→ │   ChromaDB      │
    │   Articles      │    │   Application   │    │   Vector Store  │
    │   + GSI Indexes │    │   Server        │    │   + Embeddings  │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                  │
                                  ▼
                    🤖 Intelligent Chat Interface
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Redis         │◄── │   RAG System    │──→ │   Web Search    │
    │   Session       │    │   Orchestrator  │    │   Integration   │
    │   Management    │    │   + Chat Logic  │    │   (Tavily)      │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                  │
                                  ▼
                            ┌─────────────────┐
                            │   AWS Bedrock   │
                            │   LLM Response  │
                            │   Generation    │
                            └─────────────────┘
```

### Core Technologies

| Component              | Technology   | Purpose                                         |
| ---------------------- | ------------ | ----------------------------------------------- |
| **API Framework**      | FastAPI      | High-performance async web server               |
| **Database**           | AWS DynamoDB | Scalable article storage with GSI indexes       |
| **AI Models**          | AWS Bedrock  | Summarization and embeddings                    |
| **RAG Framework**      | LlamaIndex   | Intelligent question-answering orchestration    |
| **Vector Store**       | ChromaDB     | Semantic search and similarity matching         |
| **Cache**              | Redis        | Session management and performance optimization |
| **Content Extraction** | Newspaper3k  | Full article text extraction                    |
| **Web Search**         | Tavily API   | Real-time internet information retrieval        |
| **Deployment**         | AWS Lambda   | Serverless cloud hosting                        |

### Data Schema

**ArticleSummaries Table (DynamoDB)**

- **Partition Key**: `url` (String) - Unique article URL (Primary access pattern)
- **Attributes**:
  - `title` (String) - Article headline
  - `author` (String, Optional) - Article author
  - `source_name` (String) - News source name (e.g., "BBC News", "CNN")
  - `source_id` (String) - NewsAPI source identifier
  - `published_date` (String, ISO 8601) - Publication timestamp for sorting
  - `summary` (String) - AWS Bedrock-generated article summary
  - `summary_length` (Number) - Character count of generated summary
  - `category` (String) - Article category from NewsAPI sources
  - `created_at` (String, ISO 8601) - Processing timestamp
  - `ttl` (Number, Unix timestamp) - Auto-deletion after 30 days

**Global Secondary Indexes (GSI)**:

- **DateIndex**:
  - Partition Key: `published_date` (String)
  - Sort Key: `created_at` (String)
  - Purpose: Efficient date-based queries with chronological ordering
- **SourceIndex**:
  - Partition Key: `source_name` (String)
  - Sort Key: `published_date` (String)
  - Purpose: Source-based filtering with date sorting

**Access Patterns**:

- Get article by URL (Primary Key)
- List articles by date range (DateIndex GSI)
- List articles by source (SourceIndex GSI)
- Scan for RAG vector embeddings (Full table scan for ChromaDB sync)

## 🚀 Key Features

### News Processing

- **Automated Collection**: 500+ articles daily from NewsAPI sources
- **Content Extraction**: Full article text via Newspaper3k
- **AI Summarization**: AWS Bedrock foundation models generate concise summaries
- **Batch Processing**: Configurable concurrency with semantic chunking for long articles

### Smart Retrieval & Chat

- **Flexible Queries**: Date, source, and topic-based filtering with pagination
- **RAG-Powered Chat**: LlamaIndex orchestrates context-aware conversations
- **Hybrid Search**: Combines stored articles with real-time web search (Tavily)
- **Vector Similarity**: ChromaDB enables semantic article matching

### Performance & Reliability

- **Async Architecture**: High-throughput non-blocking processing
- **Auto-scaling**: DynamoDB and Lambda scale with demand
- **Health Monitoring**: Comprehensive logging and error handling
- **Data Management**: 30-day TTL with optimized GSI indexes

## 🚢 Infrastructure & Deployment

### Current Setup

- **Runtime**: FastAPI with AWS Lambda serverless deployment
- **Storage**: DynamoDB (articles) + ChromaDB (vectors) + Redis (cache)
- **AI Services**: AWS Bedrock foundation models + OpenAI integration

### 🚧 Ongoing Infrastructure Upgrades

- **Docker Containerization**: Multi-stage builds with health checks
- **AWS ECR Integration**: Automated image building and security scanning
- **Terraform IaC**: Automated infrastructure provisioning and management
- **CI/CD Pipeline**: Zero-downtime deployments with auto-scaling
- **Monitoring**: CloudWatch metrics, X-Ray tracing, centralized logging

## 📝 Configuration & Monitoring

### System Settings

- **AI Models**: AWS Bedrock foundation models with configurable parameters
- **Processing**: Parallel batch processing with 500-7000 character content limits
- **Storage**: 30-day TTL with optimized GSI indexing
- **RAG System**: LlamaIndex orchestration with adjustable similarity thresholds

### Health & Performance

- **Monitoring**: Database connectivity, AI model status, cache performance
- **Metrics**: Response times, processing throughput, query efficiency
- **Logging**: Structured logs with request tracking and error detection
- **Alerting**: Automated issue identification and performance profiling

**Note**: This project is actively under development with ongoing infrastructure improvements. The Docker, ECR, and Terraform integration will provide enhanced scalability, reliability, and deployment automation.
