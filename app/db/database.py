import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from app.config import settings
import logging

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


# Get table reference (lazy loading approach)
def get_table():
    """Get DynamoDB table reference with error handling"""
    try:
        table = dynamodb.Table(TABLE_NAME)
        return table
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
            logger.error(
                f"Table {TABLE_NAME} does not exist - please create it in AWS Console")
        elif error_code == 'UnauthorizedOperation':
            logger.error(
                f"Access denied to table {TABLE_NAME} - check AWS credentials")
        else:
            logger.error(f"AWS error checking table health: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in health check: {e}")
        return False


    # Export for use in other modules
__all__ = ['dynamodb', 'table', 'health_check', 'TABLE_NAME']
