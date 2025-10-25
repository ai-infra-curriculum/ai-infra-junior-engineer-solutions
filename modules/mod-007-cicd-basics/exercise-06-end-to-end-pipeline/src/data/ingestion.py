"""Data ingestion module for loading data from various sources."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import boto3
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class DataIngester:
    """Load data from various sources (CSV, S3, databases)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data ingester.

        Args:
            config: Configuration dictionary with source-specific settings
        """
        self.config = config or {}
        logger.info("DataIngester initialized")

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.ParserError: If CSV parsing fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.info(f"Loading data from CSV: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from CSV")

        return df

    def load_from_s3(self, s3_uri: str, aws_profile: Optional[str] = None) -> pd.DataFrame:
        """Load data from S3.

        Args:
            s3_uri: S3 URI (s3://bucket/key)
            aws_profile: AWS profile name (optional)

        Returns:
            DataFrame with loaded data

        Raises:
            ValueError: If S3 URI is invalid
            boto3.exceptions.S3Error: If S3 access fails
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        # Parse S3 URI
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        logger.info(f"Loading data from S3: {s3_uri}")

        # Create S3 client
        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        s3 = session.client('s3')

        # Download file
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])

        logger.info(f"Loaded {len(df)} rows from S3")
        return df

    def load_from_database(
        self,
        connection_string: str,
        query: Optional[str] = None,
        table_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Load data from database.

        Args:
            connection_string: SQLAlchemy connection string
            query: SQL query to execute (optional)
            table_name: Table name to load (optional, used if query not provided)

        Returns:
            DataFrame with loaded data

        Raises:
            ValueError: If neither query nor table_name provided
            sqlalchemy.exc.SQLAlchemyError: If database access fails
        """
        if not query and not table_name:
            raise ValueError("Either query or table_name must be provided")

        logger.info("Connecting to database")
        engine = create_engine(connection_string)

        if query:
            logger.info(f"Executing query: {query[:100]}...")
            df = pd.read_sql_query(query, engine)
        else:
            logger.info(f"Loading table: {table_name}")
            df = pd.read_sql_table(table_name, engine)

        logger.info(f"Loaded {len(df)} rows from database")
        return df

    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from any source based on source string.

        Args:
            source: Source identifier (file path, S3 URI, or connection string)
            **kwargs: Additional arguments for specific loaders

        Returns:
            DataFrame with loaded data
        """
        if source.startswith("s3://"):
            return self.load_from_s3(source, **kwargs)
        elif source.startswith(("postgresql://", "mysql://", "sqlite://")):
            return self.load_from_database(source, **kwargs)
        else:
            return self.load_from_csv(source, **kwargs)
