import os
import urllib
import urllib.parse

import databases
import sqlalchemy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_database_url():
    # Required environment variables with no defaults
    required_vars = {
        "db_password": os.environ.get("DB_PASSWORD"),
        "db_username": os.environ.get("DB_USERNAME"),
        "host_server": os.environ.get("DB_HOST"),
    }

    # Check for missing required variables
    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Optional variables with safe defaults
    db_server_port = urllib.parse.quote_plus(str(os.environ.get("DB_PORT", "5432")))
    database_name = os.environ.get("DB_NAME", "postgres")
    ssl_mode = urllib.parse.quote_plus(str(os.environ.get("DB_SSL_MODE", "require")))

    # Safely quote credentials
    db_username = urllib.parse.quote_plus(str(required_vars["db_username"]))
    db_password = urllib.parse.quote_plus(str(required_vars["db_password"]))
    host_server = urllib.parse.quote_plus(str(required_vars["host_server"]))

    return f"postgresql://{db_username}:{db_password}@{host_server}:{db_server_port}/{database_name}?sslmode={ssl_mode}"


# Use the function to get the DATABASE_URL
try:
    DATABASE_URL = get_database_url()
except EnvironmentError as e:
    raise RuntimeError(f"Database configuration error: {str(e)}")

# Database instance and SQLAlchemy metadata
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()
