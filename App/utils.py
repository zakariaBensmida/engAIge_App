# app/utils.py
import os

def ensure_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

