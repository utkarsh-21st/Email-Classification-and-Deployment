from flask import Flask
from pathlib import Path
import os

app = Flask(__name__)
app.secret_key = '201438'
app_dir = Path(os.getcwd()) / 'app'

temp_folder = '__temp__'
temp_dir = app_dir / 'data' / temp_folder
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

result_folder = 'classified_messages'
result_dir = app_dir / result_folder

from app import views
