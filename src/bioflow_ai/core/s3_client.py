import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger

from bioflow_ai.core.config import get_settings

s3_client = boto3.resource(
    's3',
    endpoint_url=get_settings().persistent.storage.s3.ENDPOINT,
    aws_access_key_id=get_settings().persistent.storage.s3.ACCESS_KEY.get_secret_value(),
    aws_secret_access_key=get_settings().persistent.storage.s3.SECRET_KEY.get_secret_value(),
    config=Config(signature_version='s3v4'),
    region_name='us-east-1',
    verify=False,
)


def init_bucket():
    settings = get_settings().persistent.storage.s3
    bucket_name = settings.BUCKET
    bucket = s3_client.Bucket(bucket_name)

    try:
        s3_client.meta.client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' already exists.")
    except ClientError as e:
        error_code = str(e.response.get('Error', {}).get('Code', ''))
        if error_code in ('404', 'NoSuchBucket', 'NotFound'):
            logger.info(f"Bucket '{bucket_name}' does not exist, creating a new one.")
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            logger.exception(e)

    return bucket
