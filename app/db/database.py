import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from app.config import settings
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DynamoDB configuration from settings
AWS_REGION = settings.aws_default_region
TABLE_NAME = settings.dynamodb_table_name

# Create DynamoDB resource
try:
    dynamodb = boto3.resource(
        'dynamodb',
        region_name=AWS_REGION,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key
    )
    logger.info(f"DynamoDB resource created for region: {AWS_REGION}")
except NoCredentialsError:
    logger.error("AWS credentials not found")
    raise


def create_table_if_not_exists():
    """Create DynamoDB table with proper schema and GSIs if it doesn't exist"""
    try:
        # Check if table exists
        table = dynamodb.Table(TABLE_NAME)
        table.load()
        logger.info(f"Table {TABLE_NAME} already exists")
        return table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            logger.info(f"Table {TABLE_NAME} does not exist. Creating...")

            # Create table with proper schema
            table = dynamodb.create_table(
                TableName=TABLE_NAME,
                KeySchema=[
                    {
                        'AttributeName': 'url',
                        'KeyType': 'HASH'  # Partition key
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'url',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'published_date_key',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'created_at',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'source_name',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'published_date',
                        'AttributeType': 'S'
                    }
                ],
                GlobalSecondaryIndexes=[
                    {
                        'IndexName': 'DateIndex',
                        'KeySchema': [
                            {
                                'AttributeName': 'published_date_key',
                                'KeyType': 'HASH'
                            },
                            {
                                'AttributeName': 'created_at',
                                'KeyType': 'RANGE'
                            }
                        ],
                        'Projection': {
                            'ProjectionType': 'ALL'
                        },
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    },
                    {
                        'IndexName': 'SourceIndex',
                        'KeySchema': [
                            {
                                'AttributeName': 'source_name',
                                'KeyType': 'HASH'
                            },
                            {
                                'AttributeName': 'published_date',
                                'KeyType': 'RANGE'
                            }
                        ],
                        'Projection': {
                            'ProjectionType': 'ALL'
                        },
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    }
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )

            # Wait for table to be created
            logger.info(f"Waiting for table {TABLE_NAME} to be created...")
            table.wait_until_exists()

            # Wait a bit more for GSIs to be active
            logger.info("Waiting for Global Secondary Indexes to be active...")
            time.sleep(10)

            logger.info(
                f"✅ Table {TABLE_NAME} created successfully with GSIs!")
            return table
        else:
            logger.error(f"Error creating table: {e}")
            raise


# Get table reference with auto-creation
def get_table():
    """Get DynamoDB table reference with auto-creation if needed"""
    try:
        return create_table_if_not_exists()
    except ClientError as e:
        logger.error(f"Error accessing table {TABLE_NAME}: {e}")
        raise


# Create table instance
table = get_table()


def health_check():
    """Validate DynamoDB connection and table accessibility"""
    try:
        # Load table metadata
        table.load()

        # Check table status
        if table.table_status == 'ACTIVE':
            logger.info(f"Table {TABLE_NAME} is active and accessible")
            return True
        else:
            logger.warning(f"Table {TABLE_NAME} status: {table.table_status}")
            return False

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceNotFoundException':
            logger.error(f"Table {TABLE_NAME} does not exist")
            return False
        elif error_code == 'UnauthorizedOperation':
            logger.error(
                f"Access denied to table {TABLE_NAME} - check AWS credentials")
            return False
        else:
            logger.error(f"AWS error checking table health: {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error in health check: {e}")
        return False


# Export for use in other modules
__all__ = ['dynamodb', 'table', 'health_check',
           'TABLE_NAME', 'create_table_if_not_exists']
