import sqlalchemy

from .base import metadata

# Table schema definition
notes = sqlalchemy.Table(
    "notes",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("text", sqlalchemy.String),
    sqlalchemy.Column("completed", sqlalchemy.Boolean),
)

# Engine setup and table creation
from .base import DATABASE_URL

engine = sqlalchemy.create_engine(DATABASE_URL, pool_size=3, max_overflow=0)
metadata.create_all(engine)
