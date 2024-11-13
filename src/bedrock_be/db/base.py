import os
import urllib
import databases
import sqlalchemy

# Database connection string construction
host_server = os.environ.get('host_server', 'techx-kal-dev-postgres.coadbyfuowjn.ap-southeast-1.rds.amazonaws.com')
db_server_port = urllib.parse.quote_plus(str(os.environ.get('db_server_port', '5432')))
database_name = os.environ.get('database_name', 'postgres')
db_username = urllib.parse.quote_plus(str(os.environ.get('db_username', 'bedrock-llm')))
db_password = urllib.parse.quote_plus(str(os.environ.get('db_password', 'JpjLbHhPsFnJiNdC')))
ssl_mode = urllib.parse.quote_plus(str(os.environ.get('ssl_mode', 'require')))

DATABASE_URL = f'postgresql://{db_username}:{db_password}@{host_server}:{db_server_port}/{database_name}?sslmode={ssl_mode}'

# Database instance and SQLAlchemy metadata
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()
