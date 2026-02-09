"""
SQLite client with explicit connection management and dict-style results.

Simple tool storage with raw SQL for performance
and explicit memory management.
"""

import sqlite3
import json
import logging
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)

# Global cache for SQLiteClient instances per user
_client_cache: Dict[str, 'SQLiteClient'] = {}

class SQLiteClient:
    """
    Raw SQL client with explicit per-request connections (no pooling) and automatic user isolation.
    
    IMPORTANT: SQLite does not support Row Level Security (RLS) like PostgreSQL.
    Each user has their own separate SQLite database file, but we still need manual
    user_id filtering in queries to maintain consistency with the PostgreSQL patterns
    and to prevent accidental cross-user data access if database paths are misconfigured.
    
    This manual filtering is NOT redundant - it's the ONLY mechanism for user isolation
    in SQLite, unlike PostgreSQL where it would be redundant with RLS policies.
    """
    
    def __init__(self, db_path: str, user_id: str):
        self.db_path = db_path
        self.user_id = user_id
        self._ensure_db_directory()
        logger.info(f"SQLite client initialized: {db_path} for user {user_id}")
    
    def _ensure_db_directory(self):
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Creates fresh connection per request with sqlite3.Row factory for dict-like access."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logger.error(f"SQLite connection failed for {self.db_path}: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or {})
            
            if cursor.description:
                return [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                return []
    
    def execute_insert(self, query: str, params: Optional[Dict] = None) -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or {})
            conn.commit()
            return str(cursor.lastrowid)
    
    def execute_bulk_insert(self, query: str, params_list: List[Dict]) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    def execute_update(self, query: str, params: Optional[Dict] = None) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or {})
            conn.commit()
            return cursor.rowcount
    
    def execute_delete(self, query: str, params: Optional[Dict] = None) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or {})
            conn.commit()
            return cursor.rowcount
    
    def create_table(self, table_name: str, columns: List[str], if_not_exists: bool = True):
        if_not_exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        columns_str = ", ".join(columns)
        
        query = f"""
        CREATE TABLE {if_not_exists_clause}{table_name} (
            {columns_str}
        )
        """
        
        self.execute_query(query)
        logger.info(f"Table created: {table_name}")
    
    def json_insert(self, table_name: str, data: Dict[str, Any], 
                   json_columns: Optional[List[str]] = None) -> str:
        """Inserts with JSON serialization, auto user_id injection, and timestamps."""
        data = data.copy()
        data['user_id'] = self.user_id
        data['created_at'] = format_utc_iso(utc_now())
        data['updated_at'] = format_utc_iso(utc_now())
        
        # Serialize JSON columns
        if json_columns:
            for col in json_columns:
                if col in data and data[col] is not None:
                    data[col] = json.dumps(data[col])
        
        columns = list(data.keys())
        placeholders = [f":{col}" for col in columns]
        
        query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        return self.execute_insert(query, data)
    
    def json_update(self, table_name: str, data: Dict[str, Any], 
                   where_clause: str, where_params: Optional[Dict] = None,
                   json_columns: Optional[List[str]] = None) -> int:
        """
        Updates with JSON serialization, timestamps, and user isolation via AND user_id.
        
        NOTE: Manual user_id filtering is REQUIRED for SQLite as it doesn't support RLS.
        This is the only mechanism for user isolation in SQLite databases.
        """
        data = data.copy()
        data['updated_at'] = format_utc_iso(utc_now())
        
        # Serialize JSON columns
        if json_columns:
            for col in json_columns:
                if col in data and data[col] is not None:
                    data[col] = json.dumps(data[col])
        
        set_clauses = [f"{col} = :{col}" for col in data.keys()]
        
        query = f"""
        UPDATE {table_name}
        SET {', '.join(set_clauses)}
        WHERE {where_clause} AND user_id = :user_id
        """
        
        # Combine data and where parameters
        params = data.copy()
        params['user_id'] = self.user_id
        if where_params:
            params.update(where_params)
        
        return self.execute_update(query, params)
    
    def json_select(self, table_name: str, where_clause: Optional[str] = None,
                   where_params: Optional[Dict] = None, json_columns: Optional[List[str]] = None,
                   order_by: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Selects with JSON deserialization and automatic user isolation via WHERE user_id.
        
        NOTE: Manual user_id filtering is REQUIRED for SQLite as it doesn't support RLS.
        This is the only mechanism for user isolation in SQLite databases.
        """
        query = f"SELECT * FROM {table_name} WHERE user_id = :user_id"
        params = {'user_id': self.user_id}
        
        if where_clause:
            query += f" AND {where_clause}"
            if where_params:
                params.update(where_params)
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        rows = self.execute_query(query, params)
        
        # Deserialize JSON columns (SQLite stores JSON as TEXT)
        if json_columns:
            for row in rows:
                for col in json_columns:
                    if col in row and row[col]:
                        try:
                            row[col] = json.loads(row[col])
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                "Invalid JSON for column '%s' in table '%s'; returning None",
                                col,
                                table_name,
                            )
                            row[col] = None
        
        return rows
    
    def json_delete(self, table_name: str, where_clause: str, 
                   where_params: Optional[Dict] = None) -> int:
        """
        Deletes rows matching WHERE clause with user isolation.
        
        NOTE: Manual user_id filtering is REQUIRED for SQLite as it doesn't support RLS.
        This is the only mechanism for user isolation in SQLite databases.
        """
        query = f"""
        DELETE FROM {table_name}
        WHERE {where_clause} AND user_id = :user_id
        """
        
        params = {'user_id': self.user_id}
        if where_params:
            params.update(where_params)
        
        return self.execute_delete(query, params)
    
    def table_exists(self, table_name: str) -> bool:
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=:table_name
        """
        result = self.execute_query(query, {'table_name': table_name})
        return len(result) > 0
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)


def get_sqlite_client(db_path: str, user_id: str) -> SQLiteClient:
    """
    Get a singleton SQLiteClient instance for the given user.
    
    Args:
        db_path: Path to the SQLite database file
        user_id: User ID for isolation
        
    Returns:
        SQLiteClient instance (singleton per user)
    """
    cache_key = f"{user_id}:{db_path}"
    
    if cache_key not in _client_cache:
        _client_cache[cache_key] = SQLiteClient(db_path, user_id)
        logger.debug(f"Created new SQLiteClient singleton for user {user_id}")
    else:
        logger.debug(f"Reusing existing SQLiteClient singleton for user {user_id}")
    
    return _client_cache[cache_key]
