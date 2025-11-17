from fastapi import UploadFile
from pathlib import Path

from demo_app import state

# simulate staged entries
entries = [
    {"filename": "foo.txt", "original_name": "Foo.txt"},
    {"filename": "bar.txt", "original_name": "Bar.txt"},
]
state.add_staged_files(entries)
print(state.read_staged_manifest())
