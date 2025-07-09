from app.db.database import table, logger
from botocore.exceptions import ClientError
from typing import Dict, Any, Optional, List
from boto3.dynamodb.conditions import Key

# ================================
# BASIC CRUD OPERATIONS
# ================================


def put_article(item_data: Dict[str, Any]) -> Dict[str, Any]:
    """Pure DynamoDB put_item operation"""
    try:
        response = table.put_item(Item=item_data)
        logger.info(f"Article stored: {item_data.get('url', 'unknown')}")

        return {
            "success": True,
            "data": item_data,
            "dynamodb_response": response
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Put item failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }


def get_article_by_url(url: str) -> Dict[str, Any]:
    """Pure DynamoDB get_item by URL (partition key)"""
    try:
        response = table.get_item(Key={'url': url})

        if 'Item' in response:
            logger.info(f"Article retrieved: {url}")
            return {
                "success": True,
                "data": response['Item'],
                "found": True
            }
        else:
            logger.info(f"Article not found: {url}")
            return {
                "success": True,
                "data": None,
                "found": False
            }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Get item failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }


def delete_article(url: str) -> Dict[str, Any]:
    """Pure DynamoDB delete_item operation"""
    try:
        response = table.delete_item(Key={'url': url})

        logger.info(f"Article deleted: {url}")
        return {
            "success": True,
            "data": {"deleted_url": url}
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Delete item failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }


def update_article(url: str, published_date: str, update_expression: str, expression_values: Dict[str, Any]) -> Dict[str, Any]:
    """Pure DynamoDB update_item operation with composite key"""
    try:
        response = table.update_item(
            Key={
                'url': url,
                'published_date': published_date  # Add sort key
            },
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values,
            ReturnValues="ALL_NEW"
        )

        logger.info(f"Article updated: {url}")
        return {
            "success": True,
            "data": response.get('Attributes', {})
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Update item failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }

# ================================
# GSI QUERY OPERATIONS
# ================================


def query_by_date_index(date_key: str, limit: int = 50) -> Dict[str, Any]:
    """Query DateIndex GSI by date partition key"""
    try:
        response = table.query(
            IndexName='DateIndex',
            KeyConditionExpression=Key('published_date_key').eq(date_key),
            Limit=limit,
            ScanIndexForward=False  # Latest first (by created_at sort key)
        )

        logger.info(
            f"DateIndex query: {date_key}, found {len(response['Items'])} items")
        return {
            "success": True,
            "data": response['Items'],
            "count": len(response['Items']),
            "last_evaluated_key": response.get('LastEvaluatedKey')
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"DateIndex query failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }


def query_by_source_index(source_name: str, limit: int = 50) -> Dict[str, Any]:
    """Query SourceIndex GSI by source partition key"""
    try:
        response = table.query(
            IndexName='SourceIndex',
            KeyConditionExpression=Key('source_name').eq(source_name),
            Limit=limit,
            ScanIndexForward=False  # Latest first (by published_date sort key)
        )

        logger.info(
            f"SourceIndex query: {source_name}, found {len(response['Items'])} items")
        return {
            "success": True,
            "data": response['Items'],
            "count": len(response['Items']),
            "last_evaluated_key": response.get('LastEvaluatedKey')
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(
            f"SourceIndex query failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }


def query_source_by_date_range(source_name: str, start_date: str, end_date: str, limit: int = 50) -> Dict[str, Any]:
    """Query SourceIndex GSI with date range on sort key"""
    try:
        response = table.query(
            IndexName='SourceIndex',
            KeyConditionExpression=Key('source_name').eq(source_name) &
            Key('published_date').between(start_date, end_date),
            Limit=limit,
            ScanIndexForward=False
        )

        logger.info(
            f"SourceIndex date range query: {source_name} from {start_date} to {end_date}")
        return {
            "success": True,
            "data": response['Items'],
            "count": len(response['Items']),
            "last_evaluated_key": response.get('LastEvaluatedKey')
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(
            f"SourceIndex date range query failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }

# ================================
# SCAN OPERATIONS
# ================================


def scan_recent_articles(limit: int = 100, last_evaluated_key: Optional[Dict] = None) -> Dict[str, Any]:
    """Pure DynamoDB scan operation with pagination"""
    try:
        scan_params = {
            'Limit': limit,
            'Select': 'ALL_ATTRIBUTES'
        }

        if last_evaluated_key:
            scan_params['ExclusiveStartKey'] = last_evaluated_key

        response = table.scan(**scan_params)

        logger.info(f"Scan operation: found {len(response['Items'])} items")
        return {
            "success": True,
            "data": response['Items'],
            "count": len(response['Items']),
            "last_evaluated_key": response.get('LastEvaluatedKey')
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Scan operation failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }

# ================================
# BATCH OPERATIONS
# ================================


def batch_put_articles(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure DynamoDB batch_writer operation"""
    try:
        with table.batch_writer() as batch:
            for article in articles:
                batch.put_item(Item=article)

        logger.info(f"Batch put: {len(articles)} articles stored")
        return {
            "success": True,
            "data": {"items_processed": len(articles)}
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Batch put failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }


def batch_get_articles(urls: List[str]) -> Dict[str, Any]:
    """Pure DynamoDB batch_get_item operation"""
    try:
        response = table.meta.client.batch_get_item(
            RequestItems={
                table.name: {
                    'Keys': [{'url': url} for url in urls]
                }
            }
        )

        items = response['Responses'][table.name]
        logger.info(
            f"Batch get: retrieved {len(items)} out of {len(urls)} requested")

        return {
            "success": True,
            "data": items,
            "count": len(items),
            "requested_count": len(urls)
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Batch get failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }

# ================================
# UTILITY OPERATIONS
# ================================


def clear_all_articles() -> Dict[str, Any]:
    """Clear all articles by deleting and recreating the table"""
    try:
        from app.db.database import dynamodb, TABLE_NAME, create_table_if_not_exists

        logger.warning(f"DELETING table {TABLE_NAME} to clear all data...")

        # Delete the table
        table.delete()
        logger.info(f"Table {TABLE_NAME} deletion initiated")

        # Wait for table to be deleted
        logger.info("Waiting for table deletion to complete...")
        table.wait_until_not_exists()
        logger.info(f"Table {TABLE_NAME} successfully deleted")

        # Recreate the table with same schema
        logger.info(f"Recreating table {TABLE_NAME}...")
        new_table = create_table_if_not_exists()

        logger.info(f"✅ Table {TABLE_NAME} cleared and recreated successfully")
        return {
            "success": True,
            "data": {"action": "table_recreated"},
            "message": f"Table {TABLE_NAME} cleared by recreation"
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Clear table failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }
    except Exception as e:
        logger.error(f"Clear table failed: {str(e)}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


def article_exists(url: str) -> Dict[str, Any]:
    """Check if article exists (lightweight operation)"""
    try:
        response = table.get_item(
            Key={'url': url},
            ProjectionExpression='#url',  # Only get URL field
            ExpressionAttributeNames={'#url': 'url'}
        )

        exists = 'Item' in response
        logger.info(f"Article exists check: {url} = {exists}")

        return {
            "success": True,
            "exists": exists
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Exists check failed: {error_code} - {error_message}")

        return {
            "success": False,
            "error": error_message,
            "error_code": error_code
        }

# ================================
# EXPORTS
# ================================


__all__ = [
    'put_article',
    'get_article_by_url',
    'delete_article',
    'query_by_date_index',
    'query_by_source_index',
    'query_source_by_date_range',
    'scan_recent_articles',
    'batch_put_articles',
    'batch_get_articles',
    'clear_all_articles',
    'article_exists'
]
