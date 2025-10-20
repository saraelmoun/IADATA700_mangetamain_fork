#!/usr/bin/env python3
"""Script to run tests for the IADATA700_mangetamain project."""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run the test suite."""
    project_root = Path(__file__).parent
    
    print("🧪 Running InteractionsAnalyzer tests...")
    print("=" * 50)
    
    # Run pytest with uv
    cmd = [
        "uv", "run", "pytest", 
        "tests/test_interactions_analyzer.py",
        "-v",
        "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with return code {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())