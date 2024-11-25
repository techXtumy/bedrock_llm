"""Database utility functions."""

from typing import Any, Dict, Optional

from sqlalchemy import Table, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import Select

from bedrock_be.settings import settings


def build_query(
    table: Table,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> Select:
    """Build a query with filters, limit, and offset."""
    query = table.select()

    if filters:
        for key, value in filters.items():
            query = query.where(getattr(table.c, key) == value)

    if limit is not None:
        query = query.limit(limit)

    if offset is not None:
        query = query.offset(offset)

    return query


async def create_database() -> None:
    """Create a database."""
    db_url = make_url(str(settings.db_url.with_path("/postgres")))
    engine = create_async_engine(db_url, isolation_level="AUTOCOMMIT")

    async with engine.connect() as conn:
        sql = f"SELECT 1 FROM pg_database WHERE datname='{settings.db_base}'"
        database_existance = await conn.execute(text(sql))
        database_exists = database_existance.scalar() == 1

    if database_exists:
        await drop_database()

    async with engine.connect() as conn:
        sql = (
            f'CREATE DATABASE "{settings.db_base}" '
            'ENCODING "utf8" TEMPLATE template1'
        )
        await conn.execute(text(sql))


async def drop_database() -> None:
    """Drop current database."""
    db_url = make_url(str(settings.db_url.with_path("/postgres")))
    engine = create_async_engine(db_url, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        disc_users = (
            "SELECT pg_terminate_backend(pg_stat_activity.pid) "
            "FROM pg_stat_activity "
            f"WHERE pg_stat_activity.datname = '{settings.db_base}' "
            "AND pid <> pg_backend_pid();"
        )
        await conn.execute(text(disc_users))
        await conn.execute(text(f'DROP DATABASE "{settings.db_base}"'))
