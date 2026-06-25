import sys
import pathlib

backend_dir = str(pathlib.Path(__file__).parent / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
