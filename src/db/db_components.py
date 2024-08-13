from __future__ import annotations

import logging
import re
from io import StringIO

import pandas as pd
import psycopg2.pool
from sqlalchemy import create_engine


class DatabaseConnection:
    def __init__(self, admin_creds: dict, db_info: dict):
        self.admin_creds = admin_creds
        self.db_info = db_info

        self.engine, self.conn = self.create_my_engine()
        self.pool = self.create_my_pool()
        self.log_database_info()

    def create_my_pool(self):
        """Initialize connection pool"""
        pool = psycopg2.pool.SimpleConnectionPool(
            1,
            10,
            user=self.admin_creds["user"],
            password=self.admin_creds["password"],
            host=self.admin_creds["host"],
            port=self.admin_creds["port"],
            database=self.db_info["database"],
        )
        return pool

    def create_my_engine(self):
        """Get the database engine."""
        engine = create_engine(
            f'postgresql+psycopg2://{self.admin_creds["user"]}:{self.admin_creds["password"]}@{self.admin_creds["host"]}:{self.admin_creds["port"]}/{self.db_info["database"]}'
        )
        conn = psycopg2.connect(
            f'dbname={self.db_info["database"]} user={self.admin_creds["user"]} host={self.admin_creds["host"]} password={self.admin_creds["password"]}'
        )
        return engine, conn

    def log_database_info(self):
        """Log connection and database information."""
        conn = self.pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT current_database();")
            db_name = cur.fetchone()[0]
            cur.execute(
                "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('information_schema', 'pg_catalog');"
            )
            tables = cur.fetchall()
            logging.info(f"Connected to database: {db_name}")
            logging.info("Tables in the database:")
            for schema, table in tables:
                logging.info(f"Schema: {schema}, Table: {table}\n")
        except (Exception, psycopg2.DatabaseError) as error:
            logging.error(f"Failed to fetch database information: {error}")
        finally:
            cur.close()
            self.pool.putconn(conn)

    def close_pool(self):
        """Close the connection pool on exit."""
        self.pool.closeall()
        logging.info("Connection pool closed.")


class DatabaseOperations:
    def __init__(self, connection: DatabaseConnection, schema: str, table: str):
        self.connection = connection
        self.schema = schema
        self.table = table

    def create_table_if_not_exists(self, df: pd.DataFrame) -> None:
        """Create table if it does not exist."""
        conn = self.connection.pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(f"SET search_path TO {self.schema}")

            clean_columns = [self._clean_column_name(col) for col in df.columns]

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                {', '.join([f"{col} {self._map_dtype(dtype)}" for col, dtype in zip(clean_columns, df.dtypes)])});
            """
            logging.info(f"Creating table with SQL: {create_table_sql}")
            cur.execute(create_table_sql)
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            conn.rollback()
            logging.error(f"Failed to create table: {error}")
        finally:
            cur.close()
            self.connection.pool.putconn(conn)

    @staticmethod
    def _clean_column_name(column_name: str) -> str:
        """Clean column name to be SQL-friendly. Replaces non-alphanumeric characters with _"""
        clean_name = re.sub(r"[^\w]", "_", column_name)
        if clean_name[0].isdigit():
            clean_name = f"col_{clean_name}"
        return clean_name

    @staticmethod
    def _map_dtype(dtype):
        """Map pandas/numpy dtype to SQL dtype."""
        if pd.api.types.is_integer_dtype(dtype):
            return "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            return "FLOAT"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        else:
            return "TEXT"


class DataHandler:
    def __init__(self, connection: DatabaseConnection, schema: str, table: str, batch_size: int):
        self.connection = connection
        self.schema = schema
        self.table = table
        self.batch_size = batch_size

    def insert_batches_to_db(self, df: pd.DataFrame) -> None:
        """Insert data into the database in batches using COPY command."""
        conn = self.connection.pool.getconn()
        try:
            num_rows = len(df)
            for start in range(0, num_rows, self.batch_size):
                end = start + self.batch_size
                batch_data = df.iloc[start:end]
                logging.info(
                    f"Inserting batch {start // self.batch_size + 1}: rows {start} to {end - 1}"
                )
                self._copy_from_stringio(conn, batch_data)
        except (Exception, psycopg2.DatabaseError) as error:
            logging.error(f"Failed to insert batches: {error}")
        finally:
            self.connection.pool.putconn(conn)

    def _copy_from_stringio(self, conn, df: pd.DataFrame) -> None:
        """Using COPY command to bulk load data."""
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        cursor = conn.cursor()
        try:
            cursor.execute(f"SET search_path TO {self.schema};")
            cursor.copy_from(buffer, self.table, sep=",", null="")
            conn.commit()
            logging.info(f"SUCCESS: Data copied to {self.table} using COPY command.")
        except (Exception, psycopg2.DatabaseError) as error:
            conn.rollback()
            logging.error(f"Failed to copy data using COPY command: {error}")

    def fetch_data(self, query: str = None) -> pd.DataFrame:
        """Fetch data from the database."""
        conn = self.connection.pool.getconn()
        try:
            cur = conn.cursor()
            if query is None:
                query = f"""SELECT * FROM {self.schema}.{self.table};"""
            cur.execute(query)

            chunks = []
            while True:
                rows = cur.fetchmany(self.batch_size)
                if not rows:
                    break
                chunks.append(pd.DataFrame(rows, columns=[desc[0] for desc in cur.description]))

            return pd.concat(chunks, ignore_index=True)

        except (Exception, psycopg2.DatabaseError) as error:
            logging.error(f"Failed to fetch data: {error}")
            return pd.DataFrame()
        finally:
            cur.close()
            self.connection.pool.putconn(conn)
