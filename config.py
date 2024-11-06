# config.py for scraping player info from college softball rosters

import os
from dotenv import load_dotenv

load_dotenv()

# INPUT_EXCEL_FILE = os.getenv('INPUT_EXCEL_FILE')
INPUT_EXCEL_FILE=r"C:\Users\dell3\source\repos\school-data-scraper-5\Freelancer_Data_Mining_Project_Softball.xlsx"
GEMINI_API_KEYS = os.getenv('GEMINI_API_KEYS').split(',')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')
# OXYLABS_USERNAME = os.getenv('OXYLABS_USERNAME')
# OXYLABS_PASSWORD = os.getenv('OXYLABS_PASSWORD')