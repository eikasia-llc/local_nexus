
import sys
import os
import time

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.unified_engine import create_engine_from_defaults

def test_engine_hang():
    print("Initializing Engine...")
    engine = create_engine_from_defaults()
    
    print("Engine Initialized. Running Query...")
    start_time = time.time()
    
    try:
        # Test a simple query first
        response = engine.query("Hello, are you there?")
        print(f"Response: {response.answer}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        
        # Test a complex query that might trigger all paths
        print("\nRunning Complex Query...")
        response = engine.query("How many sales in 2024?")
        print(f"Response: {response.answer}")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_engine_hang()
