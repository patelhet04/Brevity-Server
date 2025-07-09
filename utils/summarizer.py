import logging
from .news_fetcher import fetch_news_enhanced, test_extraction
from app.config import settings
import asyncio
import json
import os
from typing import Dict, Any, List
import boto3
from botocore.exceptions import ClientError
import sys
from pathlib import Path

# Add the project root to the path so we can import from app
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


class BedrockSummarizer:
    """Amazon Bedrock Titan Text G1 - Lite summarizer with batch processing"""

    def __init__(self):
        """Initialize the Bedrock summarizer"""
        self.bedrock_config = settings.get_bedrock_config()
        self.client = self._setup_bedrock_client()
        self.model_id = self.bedrock_config["summarization_model"]
        self.max_tokens = self.bedrock_config["max_tokens"]
        self.temperature = self.bedrock_config["temperature"]
        self.top_p = self.bedrock_config["top_p"]
        self.max_input_chars = 6000
        self.batch_size = 5  # Process 5 articles at a time
        self.delay_between_batches = 1.0  # 1 second delay between batches

    def _setup_bedrock_client(self):
        """Setup the Bedrock client with credentials"""
        try:
            return boto3.client(
                'bedrock-runtime',
                aws_access_key_id=self.bedrock_config["aws_access_key_id"],
                aws_secret_access_key=self.bedrock_config["aws_secret_access_key"],
                region_name=self.bedrock_config["region_name"]
            )
        except Exception as e:
            logger.error(f"Error setting up Bedrock client: {e}")
            raise

    def invoke(self, input_text: str) -> str:
        """Invoke the Bedrock model for summarization"""
        try:
            prompt = f"Please provide a concise summary of the following article:\n\n{input_text}\n\nSummary:"

            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": min(self.max_tokens, 512),
                    "temperature": self.temperature,
                    "topP": self.top_p
                }
            }

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())

            if 'results' in response_body and len(response_body['results']) > 0:
                result = response_body['results'][0]
                if 'outputText' in result:
                    summary = result['outputText'].strip()
                    if summary.startswith("Summary:"):
                        summary = summary[8:].strip()
                    return summary
                else:
                    logger.error(f"No outputText in result: {result}")
                    return "Error: No outputText in response"
            else:
                logger.error(
                    f"No results in Bedrock response. Response: {response_body}")
                return "Error: No summary generated"

        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error invoking Bedrock model: {e}")
            return f"Error: {str(e)}"

    async def process_batch(self, articles_batch: List[Dict]) -> List[Dict]:
        """Process a batch of articles"""
        results = []

        for article in articles_batch:
            try:
                bedrock_config = settings.get_bedrock_config()
                content_min_chars = bedrock_config.get(
                    "content_min_chars", 500)
                max_chars = bedrock_config.get("max_chars", 7000)

                content = article.get("full_content", "")
                title = article.get("title", "")

                if len(content) < content_min_chars:
                    results.append(
                        {**article, "summary": "Content too short to summarize"})
                    continue

                input_text = f"{title}. {content}" if title else content

                if len(input_text) > max_chars:
                    input_text = input_text[:max_chars]

                # Add small delay between individual requests in batch
                if len(results) > 0:
                    await asyncio.sleep(0.2)  # 200ms delay between requests

                summary = self.invoke(input_text)
                results.append({**article, "summary": summary})

            except Exception as e:
                logger.error(f"Error summarizing article: {e}")
                results.append({**article, "summary": f"Error: {str(e)}"})

        return results


def get_llm(model_name: str = None):
    """Get the Bedrock summarizer"""
    return BedrockSummarizer()


async def process_articles_in_batches(articles: List[Dict], batch_size: int = 5) -> List[Dict]:
    """Process articles in batches to avoid rate limits"""
    summarizer = get_llm()
    all_results = []

    # Split articles into batches
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(articles) + batch_size - 1) // batch_size

        logger.info(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} articles)")

        # Process the batch
        batch_results = await summarizer.process_batch(batch)
        all_results.extend(batch_results)

        # Add delay between batches (except for the last batch)
        if i + batch_size < len(articles):
            logger.info(
                f"Waiting {summarizer.delay_between_batches}s before next batch...")
            await asyncio.sleep(summarizer.delay_between_batches)

    return all_results


async def process_articles(articles, concurrency=None):
    """Process articles with batch processing instead of high concurrency"""
    # Get batch size from config
    bedrock_config = settings.get_bedrock_config()
    batch_size = bedrock_config.get("batch_size", 5)

    logger.info(
        f"Processing {len(articles)} articles in batches of {batch_size}")

    return await process_articles_in_batches(articles, batch_size)


async def summarize_single_article_async():
    """Async function to summarize a single article"""
    single_article = await test_extraction("https://www.foxsports.com/stories/mlb/yordan-alvarezs-return-delayed-newly-discovered-hand-fracture")
    articles = [single_article]
    return await process_articles(articles)


def single_article_test():
    """Test with a single article"""
    return asyncio.run(summarize_single_article_async())


async def summarize_articles_async():
    """Async function to summarize fetched articles"""
    articles_data = await fetch_news_enhanced()
    articles = articles_data.get("articles", [])

    if not articles:
        logger.warning("No articles to process")
        return []

    return await process_articles(articles)


def main():
    """Main entry point"""
    return asyncio.run(summarize_articles_async())


# # Execution point
# if __name__ == "__main__":
#     summarized_articles = main()

#     # Save results to file
#     try:
#         with open("summarized_articles_batch.json", "w", encoding="utf-8") as f:
#             json.dump(summarized_articles, f, ensure_ascii=False, indent=2)
#         print(f"Results saved to summarized_articles_batch.json")
#         print(f"Processed {len(summarized_articles)} articles")
#     except Exception as e:
#         print(f"Error saving results: {e}")
