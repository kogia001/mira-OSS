"""
PostgreSQL client with explicit connection management and dict-style results.

General-purpose Postgres client that can be used anywhere in the application.
Replaces SQLAlchemy with raw SQL for performance and explicit memory management.
"""

import psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector.psycopg2 import register_vector
import json
import logging
import threading
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union, Tuple
from uuid import UUID
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)

# Track if JSONB has been registered globally
_jsonb_registered = False

class PostgresClient:
    """Raw SQL client with connection pooling, user isolation via RLS, and automatic JSON serialization."""
    
    # Class-level connection pools shared across all instances
    _connection_pools: Dict[str, psycopg2.pool.ThreadedConnectionPool] = {}
    _pools_lock = threading.RLock()  # Thread safety for pool creation
    
    @classmethod
    def reset_all_pools(cls):
        """
        Reset all connection pools. Used primarily for testing.
        
        Properly closes all connections before clearing pools to avoid
        leaving dangling database connections.
        """
        with cls._pools_lock:
            # Close sync pools
            for pool in cls._connection_pools.values():
                try:
                    pool.closeall()
                except Exception as e:
                    logger.warning(f"Error closing sync pool: {e}")
            cls._connection_pools.clear()
            
            logger.debug("All PostgresClient connection pools reset")
    
    def __init__(self, database_name: str, user_id: Optional[str] = None):
        self.database_name = database_name
        self.user_id = user_id
        from clients.vault_client import get_database_url
        self._database_url = get_database_url(database_name)
        self._ensure_connection_pool()
    
    def _needs_vector(self) -> bool:
        """mira_service stores embeddings/vectors (memories, entities, messages)."""
        return self.database_name == 'mira_service'
    
    def _ensure_connection_pool(self):
        """Ensure connection pool exists for this database."""
        with self._pools_lock:
            if self.database_name not in self._connection_pools:
                try:
                    pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=2,  # Minimum connections
                        maxconn=50,  # Maximum connections
                        dsn=self._database_url,
                        connect_timeout=30
                    )
                    
                    # Register JSONB deserialization globally (works for all connections)
                    # This only needs to be done once per process
                    global _jsonb_registered
                    if not _jsonb_registered:
                        psycopg2.extras.register_default_jsonb(globally=True)
                        _jsonb_registered = True
                        logger.info("Registered JSONB deserialization globally")
                    
                    # Register pgvector type only for databases that need it
                    if self._needs_vector():
                        for i in range(pool.minconn):
                            conn = pool.getconn()
                            register_vector(conn)
                            pool.putconn(conn)
                    
                    self._connection_pools[self.database_name] = pool
                    logger.info(f"Connection pool created: {self.database_name} (min={pool.minconn}, max={pool.maxconn})")
                    
                    if self._needs_vector():
                        logger.info(f"pgvector registered for {self.database_name}")
                        
                except Exception as e:
                    logger.error(f"Connection pool creation failed for {self.database_name}: {e}")
                    raise
    
    @contextmanager
    def get_connection(self):
        """Gets pooled connection and sets app.current_user_id for Row Level Security."""
        # Ensure pool exists (thread-safe check)
        if self.database_name not in self._connection_pools:
            self._ensure_connection_pool()
        pool = self._connection_pools[self.database_name]
        conn = None
        try:
            conn = pool.getconn()
            if conn is None:
                raise Exception(f"Could not get connection from pool for {self.database_name}")
            
            if self._needs_vector():
                register_vector(conn)
            
            # ALWAYS set or clear the user context to prevent inheriting from pooled connections
            with conn.cursor() as cur:
                if self.user_id:
                    cur.execute("SET app.current_user_id = %s", (str(self.user_id),))
                else:
                    # Clear any previous user context to prevent data leaks
                    cur.execute("RESET app.current_user_id")
            yield conn
        except psycopg2.pool.PoolError as e:
            logger.error(f"Connection pool exhausted for {self.database_name}: {e}")
            raise Exception(f"Database connection pool exhausted for {self.database_name}")
        finally:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
                pool.putconn(conn)
    
    
    
    
    def _convert_uuid_params(self, params: Optional[Union[Dict, Tuple]]) -> Optional[Union[Dict, Tuple]]:
        """Convert UUID objects to strings in parameters for database compatibility."""

        def _convert(value: Any) -> Any:
            """Recursively convert UUID instances nested in query parameters."""
            if isinstance(value, UUID):
                return str(value)

            if isinstance(value, list):
                return [_convert(item) for item in value]

            if isinstance(value, tuple):
                return tuple(_convert(item) for item in value)

            if isinstance(value, dict):
                return {key: _convert(item) for key, item in value.items()}

            if isinstance(value, set):
                # psycopg2 can't bind sets; convert to list after conversion
                return [_convert(item) for item in value]

            return value

        return _convert(params) if params is not None else None
    
    def execute_query(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> List[Dict]:
        """Execute a SELECT query and return rows as list of dictionaries."""
        params = self._convert_uuid_params(params)
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                if cur.description:
                    return [dict(row) for row in cur.fetchall()]
                else:
                    conn.commit()
                    return []
    
    def execute_returning(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> List[Dict]:
        params = self._convert_uuid_params(params)
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                conn.commit()
                return [dict(row) for row in cur.fetchall()]
    
    def execute_single(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> Optional[Dict]:
        """Execute a query and return the first row as a dictionary or None."""
        results = self.execute_query(query, params)
        return results[0] if results else None
    
    def execute_scalar(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> Any:
        """Execute a query and return the first value of the first row."""
        params = self._convert_uuid_params(params)
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                return result[0] if result else None
    
    def execute_insert(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> None:
        """Execute a single INSERT query."""
        params = self._convert_uuid_params(params)
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()
    
    def execute_bulk_insert(self, query: str, params_list: List[Union[Dict, Tuple]]) -> int:
        """Execute multiple INSERT queries as a batch."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Convert UUIDs in each parameter set
                converted_params = [self._convert_uuid_params(params) for params in params_list]
                cur.executemany(query, converted_params)
                conn.commit()
                return len(converted_params)
    
    def execute_update(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> int:
        """Execute an UPDATE query and return the number of affected rows."""
        params = self._convert_uuid_params(params)
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()
                return cur.rowcount
    
    def execute_delete(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> int:
        """Execute a DELETE query and return the number of affected rows."""
        params = self._convert_uuid_params(params)
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()
                return cur.rowcount
    
    def json_insert(self, table_name: str, data: Dict[str, Any], 
                   json_columns: Optional[List[str]] = None,
                   returning: Optional[str] = None) -> Optional[Dict]:
        """Inserts with JSON serialization, user_id injection, and auto-timestamps for memories table."""
        # Add user_id and timestamps if user_id is set
        data = data.copy()
        if self.user_id:
            data['user_id'] = self.user_id
        
        # Only add created_at/updated_at for memories table
        if table_name == 'memories':
            data['created_at'] = utc_now()
            data['updated_at'] = utc_now()
        
        # Serialize JSON columns
        if json_columns:
            for col in json_columns:
                if col in data and data[col] is not None:
                    data[col] = json.dumps(data[col])
        
        columns = list(data.keys())
        placeholders = [f"%({col})s" for col in columns]
        
        query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        if returning:
            query += f" RETURNING {returning}"
            return self.execute_single(query, data)
        else:
            self.execute_query(query, data)
            return None
    
    def json_update(self, table_name: str, data: Dict[str, Any], 
                   where_clause: str, where_params: Optional[Dict] = None,
                   json_columns: Optional[List[str]] = None,
                   returning: Optional[str] = None) -> Optional[List[Dict]]:
        """Updates with JSON serialization and auto-timestamps. User isolation handled by RLS policies."""
        # Add updated timestamp
        data = data.copy()
        data['updated_at'] = utc_now()
        
        # Serialize JSON columns
        if json_columns:
            for col in json_columns:
                if col in data and data[col] is not None:
                    data[col] = json.dumps(data[col])
        
        set_clauses = [f"{col} = %({col})s" for col in data.keys()]
        
        query = f"""
        UPDATE {table_name}
        SET {', '.join(set_clauses)}
        WHERE {where_clause}
        """
        
        # Combine data and where parameters
        params = data.copy()
        if where_params:
            params.update(where_params)
        
        if returning:
            query += f" RETURNING {returning}"
            return self.execute_query(query, params)
        else:
            self.execute_update(query, params)
            return None
    
    def json_select(self, table_name: str, where_clause: Optional[str] = None,
                   where_params: Optional[Dict] = None, json_columns: Optional[List[str]] = None,
                   order_by: Optional[str] = None, limit: Optional[int] = None,
                   columns: str = "*") -> List[Dict]:
        """Selects with JSON deserialization. User isolation handled by RLS policies."""
        query = f"SELECT {columns} FROM {table_name}"
        params = where_params or {}
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        rows = self.execute_query(query, params)
        
        # Deserialize JSON columns
        # Note: JSONB columns auto-deserialize via psycopg2.extras.register_default_jsonb
        # This handles TEXT columns with JSON strings
        if json_columns:
            for row in rows:
                for col in json_columns:
                    if col in row and row[col]:
                        # Only deserialize if still a string (not already deserialized by JSONB handler)
                        if isinstance(row[col], str):
                            row[col] = json.loads(row[col])  # Raises JSONDecodeError if malformed
        
        return rows
    
    def json_delete(self, table_name: str, where_clause: str, 
                   where_params: Optional[Dict] = None) -> int:
        """Deletes rows matching WHERE clause. User isolation handled by RLS policies."""
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        params = where_params or {}
        
        # Use execute_transaction for synchronous DELETE operation
        results = self.execute_transaction([(query, params)])
        return results[0] if results else 0
    
    def table_exists(self, table_name: str, schema: str = "public") -> bool:
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = %s
        )
        """
        return self.execute_scalar(query, (schema, table_name))
    
    def create_table_if_not_exists(self, create_sql: str):
        self.execute_query(create_sql)
    
    def get_table_schema(self, table_name: str, schema: str = "public") -> List[Dict]:
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """
        return self.execute_query(query, (schema, table_name))
    
    def execute_transaction(self, operations: List[Tuple[str, Optional[Union[Dict, Tuple]]]]) -> List[Any]:
        """Executes multiple operations atomically - all succeed or all rollback."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                results = []
                for query, params in operations:
                    cur.execute(query, params)
                    if cur.description:
                        results.append([dict(row) for row in cur.fetchall()])
                    else:
                        results.append(cur.rowcount)
                conn.commit()
                return results
    
    def close_pool(self):
        if self.database_name in self.__class__._connection_pools:
            pool = self.__class__._connection_pools[self.database_name]
            pool.closeall()
            del self.__class__._connection_pools[self.database_name]
            logger.info(f"Connection pool closed: {self.database_name}")
    
    @classmethod
    def close_all_pools(cls):
        for db_name, pool in cls._connection_pools.items():
            pool.closeall()
            logger.info(f"Connection pool closed: {db_name}")
        cls._connection_pools.clear()
