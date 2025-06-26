import asyncio
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.asyncio import tqdm_asyncio
from .news_fetcher import fetch_news_enhanced, test_extraction
import json
import torch
import gc
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_llm(model_name: str):
    # Initialize the model
    model_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Create summarization pipeline
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_length=250,  # Adjust based on desired summary length
        min_length=150,
        do_sample=False,  # For deterministic summaries
        early_stopping=False,     # Don't stop at first EOS
        num_beams=4,             # Better quality generation
    )

    # Wrap with LangChain
    return HuggingFacePipeline(pipeline=summarizer)


async def summarize_article(article, summarizer, semaphore):
    """Summarize a single article with semaphore for concurrency control"""

    async with semaphore:
        try:
            # Extract content from your article JSON
            content = article.get("full_content", "")
            title = article.get("title", "")

            # Check if content is long enough to summarize
            if len(content) < 500:
                return {**article, "summary": "Content too short to summarize"}

            # Create prompt with title context if available
            input_text = f"{title}. {content}" if title else content

            max_chars = 7000  # Approximate character limit
            if len(input_text) > max_chars:
                input_text = input_text[:max_chars]

            # Generate summary
            summary = summarizer.invoke(input_text)

            # Add summary to article dictionary
            return {**article, "summary": summary}

        except Exception as e:
            print(f"Error summarizing article: {e}")
            return {**article, "summary": f"Error: {str(e)}"}


def semantic_chunking(text, tokenizer, max_tokens=512):
    """Split text based on paragraphs and sections
    This is used for map-reduce strategy"""
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
        tokens = tokenizer.encode(test_chunk, add_special_tokens=True)

        if len(tokens) <= max_tokens:
            current_chunk = test_chunk
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    validated_chunks = []
    for chunk in chunks:
        chunk_tokens = tokenizer.encode(chunk, add_special_tokens=True)
        if len(chunk_tokens) <= max_tokens:
            validated_chunks.append(chunk)
        else:
            print(
                f"Warning: Chunk still too long ({len(chunk_tokens)} tokens), truncating...")

    return chunks


def efficient_semantic_chunking(text, tokenizer, max_tokens=512):
    """Efficient semantic chunking with guaranteed size limits"""

    safe_max_tokens = max_tokens - 5
    chunks = []

    # Split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    current_chunk = ""

    for paragraph in paragraphs:
        test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

        if len(tokenizer.encode(test_chunk, add_special_tokens=True)) <= safe_max_tokens:
            current_chunk = test_chunk
        else:
            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk)

            # Handle oversized paragraph
            if len(tokenizer.encode(paragraph, add_special_tokens=True)) > safe_max_tokens:
                # Split paragraph into sentences and build chunks
                sentences = paragraph.replace('. ', '.|').split('|')
                para_chunk = ""

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    test_sentence = para_chunk + " " + sentence if para_chunk else sentence

                    if len(tokenizer.encode(test_sentence, add_special_tokens=True)) <= safe_max_tokens:
                        para_chunk = test_sentence
                    else:
                        if para_chunk:
                            chunks.append(para_chunk)

                        para_chunk = sentence

                current_chunk = para_chunk if para_chunk else ""
            else:
                current_chunk = paragraph

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Final safety check with truncation
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = tokenizer.encode(chunk, add_special_tokens=True)
        if len(chunk_tokens) > safe_max_tokens:
            print(
                f"Warning: Chunk still too long ({len(chunk_tokens)} tokens)")

        final_chunks.append(chunk)

    return final_chunks


async def map_reduce_summarization(article, summarizer, semaphore, max_tokens=1024):
    """Use map-reduce pattern for long texts using semantic_chunking technique"""
    async with semaphore:
        try:
            tokenizer = summarizer.pipeline.tokenizer
            if 'intermediate_summary' in article:
                full_article = article.get('intermediate_summary')
            else:
                full_article = article.get('full_content', "")

            if len(tokenizer.encode(full_article, add_special_tokens=True)) <= max_tokens:
                print("here")
                final_summary = summarizer.invoke(full_article)
                clear_gpu_memory()
                return {**article, "summary": final_summary}

            # Step 1: Map - split into chunks and summarize each
            chunks = efficient_semantic_chunking(full_article, tokenizer)
            chunk_summaries = []

            for chunk in chunks:
                summary = summarizer.invoke(chunk)
                chunk_summaries.append(summary)

            # Step 2: Reduce - combine summaries
            combined_summary = " ".join(chunk_summaries)

            # If combined summary is still too long, summarize again
            combined_tokens = tokenizer.encode(
                combined_summary, add_special_tokens=True)
            if len(combined_tokens) > max_tokens:
                article['intermediate_summary'] = combined_summary
                clear_gpu_memory()
                print("here2")
                return await map_reduce_summarization(article, summarizer, semaphore)

            return {**article, "summary": combined_summary}

        except Exception as e:
            print(f"Error summarizing article: {e}")
            return {**article, "summary": f"Error: {str(e)}"}

# Process articles in batches with controlled concurrency


async def process_articles(articles, concurrency=2):
    """Process articles with controlled concurrency"""
    # Create semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(concurrency)

    """Setup the summarizer model. Model options are "sshleifer/distilbart-cnn-12-6" and "google/bigbird-pegasus-large-arxiv". Distil-Bart has token limit of 1024 so we have to use chunking and map-reduce strategy to make it work.
    Whereas BigBird_Pegasus has token limit of 4096 and thus can be used using normal summarizer pipeline without chunking and so forth.
    BigBird gave terrible results as its trained for scientific data summarization.
    """
    summarizer = get_llm("sshleifer/distilbart-cnn-12-6")

    results = []

    async def summarize_and_update(article):
        result = await map_reduce_summarization(article, summarizer, semaphore)
        return result

    # Create tasks with the wrapper
    tasks = [summarize_and_update(article) for article in articles]

    # Process tasks
    results = await asyncio.gather(*tasks)

    return results


async def summarize_single_article_async(concurrency=2):
    """Async main function to summarize fetched articles"""
    # Properly await the fetch_news coroutine

    single_article = await test_extraction("https://www.foxsports.com/stories/mlb/yordan-alvarezs-return-delayed-newly-discovered-hand-fracture")
    articles = list()
    articles.append(single_article)
    # Process the article
    return await process_articles(articles, concurrency)


def single_article_test():
    """Entry point that runs the async functions for sumamrizing a single fetched article"""
    return asyncio.run(summarize_single_article_async(concurrency=5))


async def summarize_articles_async(concurrency=5):
    """Async main function to summarize fetched articles"""
    # Properly await the fetch_news coroutine
    articles = await fetch_news_enhanced()

    # Process the articles
    return await process_articles(articles.get("articles", ""), concurrency)


def main():
    """Entry point that runs the async functions"""
    return asyncio.run(summarize_articles_async(concurrency=5))


# # Execution point
# if __name__ == "__main__":
#     clear_gpu_memory()
#     summarized_articles = main()

#     # summarized_articles = single_article_test()

#     # Save results to file instead of printing
#     try:
#         with open("summarized_articles_single.json", "w", encoding="utf-8") as f:
#             json.dump(summarized_articles, f, ensure_ascii=False, indent=2)
#         print(f"Results saved to summarized_articles.json")
#     except Exception as e:
#         print(f"Error saving results: {e}")
