#!/usr/bin/env python3
"""
Simple GPU test for HeavyDB
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.heavydb_connector import (
    get_connection,
    execute_query,
    get_execution_mode
)


def test_simple_gpu():
    """Test basic GPU functionality"""
    print("\n🚀 Simple HeavyDB GPU Test")
    print("="*50)
    
    # Test connection
    conn = get_connection(force_new=True)
    
    if not conn:
        print("❌ Failed to connect to HeavyDB")
        return False
    
    print("✅ Connected to HeavyDB")
    
    # Check execution mode
    mode = get_execution_mode()
    print(f"Execution Mode: {mode.upper()}")
    
    # Create a simple test table
    try:
        cursor = conn.cursor()
        
        # Drop if exists
        cursor.execute("DROP TABLE IF EXISTS simple_test")
        
        # Create table
        cursor.execute("""
            CREATE TABLE simple_test (
                id INTEGER,
                val DOUBLE,
                name TEXT
            ) WITH (fragment_size=1000000)
        """)
        print("✅ Created test table")
        
        # Insert some data
        values = []
        for i in range(10):
            values.append(f"({i}, {i * 1.5}, 'test_{i}')")
        
        insert_sql = f"INSERT INTO simple_test VALUES {', '.join(values)}"
        cursor.execute(insert_sql)
        print("✅ Inserted test data")
        
        # Test GPU query
        start_time = time.time()
        result = execute_query("SELECT SUM(val), COUNT(*) FROM simple_test")
        query_time = time.time() - start_time
        
        if result is not None:
            print(f"✅ Query executed in {query_time:.3f}s")
            print(f"   Result: sum={result.iloc[0, 0]}, count={result.iloc[0, 1]}")
        
        # Test more complex GPU operation
        print("\nTesting GPU aggregation...")
        cursor.execute("""
            CREATE TABLE test_agg AS
            SELECT 
                id % 3 as group_id,
                SUM(val) as total,
                COUNT(*) as cnt
            FROM simple_test
            GROUP BY id % 3
        """)
        print("✅ GPU aggregation completed")
        
        # Verify results
        result = execute_query("SELECT * FROM test_agg ORDER BY group_id")
        if result is not None:
            print("   Aggregation results:")
            print(result)
        
        # Clean up
        cursor.execute("DROP TABLE simple_test")
        cursor.execute("DROP TABLE test_agg")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_correlation_simple():
    """Test simple correlation calculation"""
    print("\n\nTesting Simple Correlation...")
    print("="*50)
    
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create test table with a few strategies
        cursor.execute("DROP TABLE IF EXISTS corr_test")
        
        cursor.execute("""
            CREATE TABLE corr_test (
                trading_day INTEGER,
                strat_1 DOUBLE,
                strat_2 DOUBLE,
                strat_3 DOUBLE
            ) WITH (fragment_size=1000000)
        """)
        
        # Insert test data
        values = []
        for i in range(20):
            v1 = np.sin(i * 0.5) * 100
            v2 = np.cos(i * 0.5) * 100
            v3 = v1 * 0.8 + np.random.randn() * 10
            values.append(f"({i}, {v1}, {v2}, {v3})")
        
        insert_sql = f"INSERT INTO corr_test VALUES {', '.join(values)}"
        cursor.execute(insert_sql)
        print("✅ Created correlation test data")
        
        # Calculate correlations
        start_time = time.time()
        result = execute_query("""
            SELECT 
                CORR(strat_1, strat_2) as corr_1_2,
                CORR(strat_1, strat_3) as corr_1_3,
                CORR(strat_2, strat_3) as corr_2_3
            FROM corr_test
        """)
        
        if result is not None:
            corr_time = time.time() - start_time
            print(f"✅ Correlations calculated in {corr_time:.3f}s")
            print(f"   Corr(1,2): {result.iloc[0, 0]:.4f}")
            print(f"   Corr(1,3): {result.iloc[0, 1]:.4f}")
            print(f"   Corr(2,3): {result.iloc[0, 2]:.4f}")
        
        # Clean up
        cursor.execute("DROP TABLE corr_test")
        
        return True
        
    except Exception as e:
        print(f"❌ Correlation test failed: {e}")
        return False


if __name__ == "__main__":
    test_simple_gpu()
    test_correlation_simple()
    
    print("\n✅ Simple GPU tests complete!")