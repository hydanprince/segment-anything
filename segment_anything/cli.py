"""CLI entry point for segment-anything."""
import sys
import os

# Add parent directory to path so we can import the root-level segment module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment import main

if __name__ == "__main__":
    main()
