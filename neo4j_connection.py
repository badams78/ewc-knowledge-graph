#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j Connection Utility Module

Provides a reusable connection class for Neo4j Aura database.
Connection details:
- URI: neo4j+ssc://b86da93d.databases.neo4j.io
- Instance: MacroRoundup (b86da93d)

Usage:
    from neo4j_connection import Neo4jConnection
    
    with Neo4jConnection() as conn:
        results = conn.query("MATCH (n) RETURN count(n) as count")
        print(results)
"""

import os
import sys
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use defaults or environment variables

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: Neo4j driver not installed.")
    print("Please install it with: pip install neo4j")
    sys.exit(1)


class Neo4jConnection:
    """
    Neo4j database connection manager.
    
    Can be used as a context manager for automatic connection handling:
    
        with Neo4jConnection() as conn:
            results = conn.query("MATCH (n) RETURN n LIMIT 5")
    
    Or manually:
    
        conn = Neo4jConnection()
        conn.connect()
        results = conn.query("MATCH (n) RETURN n LIMIT 5")
        conn.close()
    """
    
    # Default connection parameters for MacroRoundup database
    DEFAULT_URI = "neo4j+ssc://b86da93d.databases.neo4j.io"
    DEFAULT_USERNAME = "neo4j"
    DEFAULT_PASSWORD = "wPFOAcs_nnaabmA3F-X6Hue72S_kWhPIDj8XHY2BnFU"
    DEFAULT_DATABASE = "neo4j"
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        Initialize Neo4j connection.
        
        Parameters are loaded in this priority:
        1. Explicit parameters passed to __init__
        2. Environment variables (NEO4J_URI, NEO4J_USERNAME, etc.)
        3. Default values for MacroRoundup database
        
        Args:
            uri: Neo4j connection URI
            username: Database username
            password: Database password
            database: Database name
        """
        self.uri = uri or os.getenv("NEO4J_URI", self.DEFAULT_URI)
        self.username = username or os.getenv("NEO4J_USERNAME", self.DEFAULT_USERNAME)
        self.password = password or os.getenv("NEO4J_PASSWORD", self.DEFAULT_PASSWORD)
        self.database = database or os.getenv("NEO4J_DATABASE", self.DEFAULT_DATABASE)
        
        self.driver = None
        self._connected = False
    
    def connect(self) -> bool:
        """
        Establish connection to Neo4j database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
            self._connected = True
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            self._connected = False
            return False
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected and self.driver is not None
    
    def query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_all: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            cypher: Cypher query string
            parameters: Optional query parameters
            fetch_all: If True, fetch all results. If False, return iterator.
        
        Returns:
            List of dictionaries containing query results
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to database. Call connect() first.")
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, parameters or {})
            if fetch_all:
                return [dict(record) for record in result]
            return result
    
    def query_single(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a Cypher query and return a single result.
        
        Args:
            cypher: Cypher query string
            parameters: Optional query parameters
        
        Returns:
            Single result dictionary or None
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to database. Call connect() first.")
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, parameters or {})
            record = result.single()
            return dict(record) if record else None
    
    def execute(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Execute a Cypher query without returning results.
        Useful for CREATE, UPDATE, DELETE operations.
        
        Args:
            cypher: Cypher query string
            parameters: Optional query parameters
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to database. Call connect() first.")
        
        with self.driver.session(database=self.database) as session:
            session.run(cypher, parameters or {})
    
    @contextmanager
    def session(self):
        """
        Get a session for more complex transactions.
        
        Usage:
            with conn.session() as session:
                with session.begin_transaction() as tx:
                    tx.run("CREATE (n:Node {name: $name})", name="Test")
                    tx.commit()
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to database. Call connect() first.")
        
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()
    
    def __enter__(self):
        """Context manager entry - connects to database."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes connection."""
        self.close()
        return False


def get_connection() -> Neo4jConnection:
    """
    Factory function to get a connected Neo4j connection.
    
    Returns:
        Connected Neo4jConnection instance
    
    Raises:
        RuntimeError: If connection fails
    """
    conn = Neo4jConnection()
    if not conn.connect():
        raise RuntimeError("Failed to connect to Neo4j database")
    return conn


# Quick test if run directly
if __name__ == "__main__":
    print("Testing Neo4j connection...")
    print(f"URI: {Neo4jConnection.DEFAULT_URI}")
    print(f"Database: {Neo4jConnection.DEFAULT_DATABASE}")
    print()
    
    with Neo4jConnection() as conn:
        # Test connection
        result = conn.query_single("MATCH (n) RETURN count(n) as node_count")
        print(f"[OK] Connected! Total nodes: {result['node_count']:,}")
        
        # Show node labels
        results = conn.query("""
            MATCH (n) 
            RETURN labels(n) as labels, count(*) as count 
            ORDER BY count DESC 
            LIMIT 5
        """)
        print("\nTop 5 node labels:")
        for r in results:
            labels = ":".join(r['labels']) if r['labels'] else "(no label)"
            print(f"  - {labels}: {r['count']:,}")
