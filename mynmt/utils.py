import gzip
import os
from pathlib import Path


def smart_open(file_path: str):
    ending = os.path.splitext(file_path)[1]
    if ending == '.gz':
        return gzip.open(file_path, 'rt')
    else:
        return open(file_path, 'r')

