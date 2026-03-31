import sys
from pathlib import Path

# Allow tests to import modules from the data/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
