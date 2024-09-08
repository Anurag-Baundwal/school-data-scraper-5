import asyncio
import re
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
import json
import google.generativeai as genai
import psutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor
from config import GEMINI_API_KEYS
import random
import logging
from datetime import datetime
import os
from urllib.parse import urlparse
from fuzzywuzzy import fuzz
import json5
import ssl
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import sys
import signal
import pymongo
import isoweek

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==== MONGO DB STUFF =====
# 2 databases - 
# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017", maxPoolSize=10)
print(f"Connection pool size: {client.maxPoolSize}")
historic_db = client["historic_data_softball"]
current_db = client["current_data_softball"]

# Collections (created if they don't exist)
coaches_collection_current = current_db["coaches"]
# debug_info_collection_current = current_db["debug_info"]

# Insert test documents
# coaches_collection_current.insert_one({"test_key": "test_value"})
# debug_info_collection_current.insert_one({"test_key": "test_value"})

print("Data inserted, check MongoDB Compass.")

# ==== MAKE DIR TO STORE OUTPUT ON FILESYSTEM ====
# Create the directory if it doesn't exist
os.makedirs('coaches3_v2', exist_ok=True)

# Add a file handler for DEBUG level
file_handler = logging.FileHandler('coaches3_v2/logs.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class WebDriverPool:
    def __init__(self, pool_size=15):
        self.pool_size = pool_size
        self.semaphore = asyncio.Semaphore(pool_size)
        self.drivers = []
        self.initialize_drivers()

    def initialize_drivers(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        chrome_options.add_argument('--log-level=3') # reduce clutter in the terminal

        service = Service(ChromeDriverManager().install())

        for _ in range(self.pool_size):
            driver = webdriver.Chrome(service=service, options=chrome_options)
            self.drivers.append(driver)

    async def get_driver(self):
        await self.semaphore.acquire()
        return self.drivers.pop()

    async def return_driver(self, driver):
        await asyncio.to_thread(self.drivers.append, driver)
        self.semaphore.release()

    def close_all(self):
        for driver in self.drivers:
            driver.quit()

    async def shutdown(self):
        """Waits for all drivers to be returned and then closes them."""
        await asyncio.gather(*[self.return_driver(driver) for driver in self.drivers])  
        self.close_all() 

# Global WebDriver pool
global_driver_pool = WebDriverPool(pool_size=10)

def log_raw_gemini_output(school_name, raw_output, sheet_name):
    logger.debug(f"Raw Gemini output for {school_name}:\n{raw_output}")
    
    output_dir = "coaches3_v2/raw_gemini_outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{sheet_name}_raw_gemini_outputs.txt")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Raw Gemini output for {school_name}:\n")
        f.write(raw_output)
        f.write("\n\n")

    logger.info(f"Raw Gemini output for {school_name} saved to {file_path}")

def save_body_html_content(school_name, body_html, sheet_name):
    output_dir = "coaches3_v2/body_html_contents"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{sheet_name}_body_html_contents.txt")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Body HTML content for {school_name}:\n")
        f.write(body_html)
        f.write("\n\n" + "="*50 + "\n\n")

    logger.info(f"Body HTML content for {school_name} saved to {file_path}")

    # # Save to MongoDB
    # current_week = isoweek.Week.thisweek().format("%Y-W%W")
    # debug_data = {
    #     "school": school_name,
    #     "sheet_name": sheet_name,
    #     "body_html": body_html,
    #     "timestamp": datetime.now(),
    #     "weekNumber": int(current_week[-2:]),
    #     "year": int(current_week[:4])
    # }
    # debug_info_collection_current.insert_one(debug_data)

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scroll_to_bottom(driver):
    def get_scroll_height():
        return driver.execute_script("return document.body.scrollHeight")

    last_height = get_scroll_height()
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for page to load

        new_height = get_scroll_height()
        if new_height == last_height:
            break
        last_height = new_height

    # Scroll back to top and then to bottom again to trigger any lazy-loaded images
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

# class APIKeyManager:
#     def __init__(self, api_keys):
#         self.api_keys = api_keys
#         self.current_index = 0

#     def get_next_key(self):
#         key = self.api_keys[self.current_index]
#         self.current_index = (self.current_index + 1) % len(self.api_keys)
#         return key

# pick a random key each time
class APIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys

    def get_next_key(self): # get_random_key
        return random.choice(self.api_keys)

api_key_manager = APIKeyManager(GEMINI_API_KEYS)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, str):
            return obj.encode('utf-8').decode('unicode_escape')
        return super().default(obj)


def validate_coach_data(coaches, body_html):
    valid_coaches = []
    body_html_lowercase = body_html.lower()
    
    for coach in coaches:
        if coach.get('name'):
            name_lowercase = coach['name'].lower()
            
            # First, try an exact match for the name (case-insensitive)
            if coach['name'].lower() in body_html_lowercase:
                logger.debug(f"Exact match found for {coach['name']}")
                valid_coach = {'name': coach['name']}
                
                # Validate other attributes
                for attr in ['title', 'email', 'phone', 'twitter']:
                # for attr in ['title', 'email', 'twitter']: # removed phone number validation for now because of dashes issue # lower the threshold?
                    if coach.get(attr):
                        if attr == 'phone':
                            valid_coach[attr] = coach[attr]
                        else:
                            attr_value = str(coach[attr]).lower()
                            if attr_value in body_html_lowercase:
                                valid_coach[attr] = coach[attr]
                            else:
                                # Try fuzzy matching for the attribute
                                match_ratio = fuzz.partial_ratio(attr_value, body_html_lowercase)
                                if match_ratio >= 85:
                                    valid_coach[attr] = coach[attr]
                                else:
                                    valid_coach[attr] = None
                    else:
                        # If the attribute is missing or null, consider it valid
                        valid_coach[attr] = None
                
                valid_coaches.append(valid_coach)
            else:
                # If no exact match for name, try fuzzy matching
                match_ratio = fuzz.partial_ratio(name_lowercase, body_html_lowercase)
                logger.debug(f"Fuzzy match ratio for {coach['name']}: {match_ratio}")

                if match_ratio >= 85:
                    valid_coach = {'name': coach['name']}
                    
                    # Validate other attributes
                    for attr in ['title', 'email', 'phone', 'twitter']:
                        if coach.get(attr):
                            attr_value = str(coach[attr]).lower()
                            if attr_value in body_html_lowercase:
                                valid_coach[attr] = coach[attr]
                            else:
                                # Try fuzzy matching for the attribute
                                attr_match_ratio = fuzz.partial_ratio(attr_value, body_html_lowercase)
                                if attr_match_ratio >= 85:
                                    valid_coach[attr] = coach[attr]
                                else:
                                    valid_coach[attr] = None
                        else:
                            # If the attribute is missing or null, consider it valid
                            valid_coach[attr] = None
                    
                    valid_coaches.append(valid_coach)
                else:
                    logger.debug(f"Coach {coach['name']} rejected: Name not found in body text")
        else:
            logger.debug(f"Coach rejected: Missing name. Coach data: {coach}")
    
    logger.info(f"Total coaches: {len(coaches)}, Valid coaches: {len(valid_coaches)}")
    return valid_coaches

def normalize_url(url):
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    return normalized.lower()

async def load_excel_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        return xls
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        return None

async def fetch_with_retry(url, headers, max_retries=4, timeout=20):
    driver = await global_driver_pool.get_driver()
    try:
        for attempt in range(max_retries):
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(driver.get, url),
                    timeout=timeout * (3**attempt) # x1, x3, x9
                )
                await asyncio.to_thread(scroll_to_bottom, driver)
                await asyncio.to_thread(scroll_to_bottom, driver)
                await asyncio.to_thread(scroll_to_bottom, driver)
                html_content = driver.page_source
                return html_content, None
            except asyncio.TimeoutError:
                logger.error(f"Timeout error fetching {url} - Attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    return None, "Max retries reached (timeout)"
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(f"Error fetching {url}: {error_msg}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    return None, error_msg
    finally:
        await global_driver_pool.return_driver(driver)
    return None, "Max retries reached"

async def gemini_based_scraping(url, school_name, sheet_name):
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'
    ]
    
    headers = {'User-Agent': random.choice(user_agents)}

    html_content, fetch_error = await fetch_with_retry(url, headers, max_retries=3, timeout=20)  # Changed: Removed driver_pool parameter
    if html_content is None:
        return None, False, 0, 0, fetch_error

    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.body
    if body is None:
        logger.warning(f"No body tag found in the HTML from {url}")
        return None, False, 0, 0
    body_html = str(body)
    save_body_html_content(school_name, body_html, sheet_name)

    for attempt in range(4):
        try:
            # Always use the full body_html now
            relevant_html = body_html

            genai.configure(api_key=api_key_manager.get_next_key())
            generation_config = {
                "temperature": 0.5,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192*16,
            }

            model = genai.GenerativeModel(
                # model_name="gemini-1.5-flash",
                model_name="gemini-1.5-flash-exp-0827",
                generation_config=generation_config
            )
            
            prompt = f"""
            Analyze the HTML content of the coaching staff webpage for {school_name} and extract information ONLY for softball *coaches* (head coach and assistant coaches - sometimes you'll see interim coaches too. Include those). They will typically be found under a softball section, and will usually only be 3-4 in number. Do not include coaches from other sports or general staff members. Extract the following information for each softball coach:
            - Name
            - Title
            - Email address (if available)
            - Phone number (If available. Sometimes it will be in the section heading (eg:Softball - Phone: 828-262-7310))
            - Twitter/X handle (if available)
            Note: Phone number is always 10 digits. If some part is in the section heading and some part is in the row for the particular coach, piece together the information to find the full phone number.
            Format the output as a JSON string with the following structure:
            {{
                "success": true/false,
                "reason": "reason for failing to scrape data" (or null if success),
                "coachingStaff": [
                    {{
                        "name": "...",
                        "title": "...",
                        "email": null,
                        "phone": null,
                        "twitter": null
                    }},
                    ...
                ]
            }}
            If you can find any softball coaching staff information, even if incomplete, set "success" to true and include the available data. If no softball coaches are found, set "success" to false and provide the reason "no softball coaches found".
            Important: Ensure all names, including those with non-English characters, are preserved exactly as they appear in the HTML. Do not escape or modify any special characters in names or other fields.
            The response should be a valid JSON string only, without any additional formatting or markdown syntax.
            """

            token_response = model.count_tokens(prompt + relevant_html)
            input_tokens = token_response.total_tokens

            response = await model.generate_content_async([prompt, relevant_html])
            
            output_tokens = model.count_tokens(response.text).total_tokens

            log_raw_gemini_output(school_name, response.text, sheet_name)

            cleaned_response = response.text.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()

            try:
                result = json.loads(cleaned_response)
                if result['success'] and 'coachingStaff' in result:
                    valid_coaches = validate_coach_data(result['coachingStaff'], body_html)
                    result['coachingStaff'] = valid_coaches
                    result['coach_count'] = len(valid_coaches)

                    if len(valid_coaches) == 0:
                        result['success'] = False
                        result['reason'] = "No valid coaches found after validation"
                else:
                    # Ensure coachingStaff and coach_count are set even if not present in the original response
                    result['coachingStaff'] = []
                    result['coach_count'] = 0
                
                # Ensure all expected fields are present
                result.setdefault('success', False)
                result.setdefault('reason', "Unknown error")
                result.setdefault('coachingStaff', [])
                result.setdefault('coach_count', 0)

                return result, result['success'], input_tokens, output_tokens, result.get('reason')
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to parse JSON from Gemini response for {school_name}: {json_error}")
                logger.debug(f"Raw response: {cleaned_response}")
                return {
                    'success': False,
                    'reason': f"Failed to parse JSON: {str(json_error)}",
                    'coachingStaff': [],
                    'coach_count': 0
                }, False, input_tokens, output_tokens, f"Failed to parse JSON: {str(json_error)}"

        except Exception as e:
            if attempt == 3:
                logger.error(f"Error in Gemini-based scraping for {school_name} after all attempts: {str(e)}")
                return None, False, 0, 0

    return None, False, 0, 0

async def process_school(school_data, sheet_name, staff_directory_column, coaches_url_column):
    school_name = school_data['School']
    max_retries = 3
    base_delay = 5
    final_result = {}
    reasons = []
    total_input_tokens = total_output_tokens = 0

    for attempt in range(max_retries):
        try:
            if staff_directory_column and pd.notna(school_data[staff_directory_column]):
                url = school_data[staff_directory_column]
                logger.info(f"Processing {school_name} (Staff Directory URL: {url}) - Attempt {attempt + 1}")
                result, success, input_tokens, output_tokens, error_msg = await gemini_based_scraping(url, school_name, sheet_name)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                if success:
                    logger.info(f"Successfully scraped and validated data for {school_name} from Staff Directory")
                    staff_directory_result = {
                        'school': school_name,
                        'url': url,
                        'success': True,
                        'reason': None,
                        'coachingStaff': result['coachingStaff'],
                        'coach_count': result['coach_count'],
                        'input_tokens': total_input_tokens,
                        'output_tokens': total_output_tokens,
                        'total_tokens': total_input_tokens + total_output_tokens
                    }
                    final_result = staff_directory_result
                else:
                    reason = error_msg if error_msg else (result['reason'] if result and 'reason' in result else 'Unknown error')
                    reasons.append(f"Staff Directory - Attempt {attempt + 1}: {reason}")
                    logger.warning(f"Scraping failed for {school_name} (Staff Directory) - Attempt {attempt + 1}: {reason}")

            if coaches_url_column and pd.notna(school_data[coaches_url_column]):
                url = school_data[coaches_url_column]
                logger.info(f"Processing {school_name} (Coaches URL: {url}) - Attempt {attempt + 1}")
                result, success, input_tokens, output_tokens, error_msg = await gemini_based_scraping(url, school_name, sheet_name)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                if success:
                    logger.info(f"Successfully scraped and validated data for {school_name} from Coaches URL")
                    coaches_url_result = {
                        'school': school_name,
                        'url': url,
                        'success': True,
                        'reason': None,
                        'coachingStaff': result['coachingStaff'],
                        'coach_count': result['coach_count'],
                        'input_tokens': total_input_tokens,
                        'output_tokens': total_output_tokens,
                        'total_tokens': total_input_tokens + total_output_tokens
                    }
                    final_result = coaches_url_result
                else:
                    reason = error_msg if error_msg else (result['reason'] if result and 'reason' in result else 'Unknown error')
                    reasons.append(f"Coaches URL - Attempt {attempt + 1}: {reason}")
                    logger.warning(f"Scraping failed for {school_name} (Coaches URL) - Attempt {attempt + 1}: {reason}")

            if final_result.get('success'):
                break 
            else:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay) 
        except Exception as e:  # <--- ADDED EXCEPT BLOCK
            reason = f"Error: {str(e)}"
            reasons.append(f"Attempt {attempt + 1}: {reason}")
            logger.error(f"Error in processing {school_name}: {reason} - Attempt {attempt + 1}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    if not final_result or not final_result['success']:
        final_result = {
            'school': school_name,
            'url': 'N/A',
            'success': False,
            'reason': '; '.join(reasons) if reasons else 'Unknown error',
            'coachingStaff': [],
            'coach_count': 0,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens
        }

    # ==== SAVE TO BOTH CURRENT AND HISTORIC DBs ====
    current_week = isoweek.Week.thisweek()
    current_week_str = f"{current_week.year}-W{current_week.week:02d}"
    historic_collection_name = f"coaches_{current_week_str[:4]}_{current_week_str[-2:]}"
    historic_collection = historic_db[historic_collection_name]
    
    coach_data = {
        "school": final_result['school'],
        "division": sheet_name, 
        "url": final_result['url'],
        "coachingStaff": final_result['coachingStaff'] if final_result['success'] else [],  
        "timestamp": datetime.now(),
        "weekNumber": int(current_week_str[-2:]),
        "year": int(current_week_str[:4])
    }
    coaches_collection_current.insert_one(coach_data)
    historic_collection.insert_one(coach_data) 

    # ==== CHANGE TRACKING ====
    track_changes(coach_data, sheet_name, current_week_str)
    return final_result

def track_changes(coach_data, sheet_name, current_week_str):
    school_name = coach_data['school']
    current_coaches = set(c['name'] for c in coach_data['coachingStaff'])

    # Get previous week's data
    if current_week_str.endswith("W01"):
        previous_week_obj = isoweek.Week(int(current_week_str[:4]) - 1, isoweek.Week.last_week_of_year(int(current_week_str[:4]) - 1).week)
    else:
        previous_week_obj = isoweek.Week(int(current_week_str[:4]), int(current_week_str[-2:]) - 1)
    previous_week_str = f"{previous_week_obj.year}-W{previous_week_obj.week:02d}"
    previous_week_collection_name = f"coaches_{previous_week_str[:4]}_{previous_week_str[-2:]}"

    previous_coaches = set()
    if previous_week_collection_name in historic_db.list_collection_names():
        previous_week_data = historic_db[previous_week_collection_name].find_one({"school": school_name})
        if previous_week_data:
            previous_coaches = set(c['name'] for c in previous_week_data['coachingStaff'])

    # Determine new hires and departures only if previous week data exists
    if previous_coaches:
        new_hires = current_coaches - previous_coaches
        departures = previous_coaches - current_coaches
    else:
        new_hires = None  # Set to None if no previous week data
        departures = None  # Set to None if no previous week data

    # Always mention both New Hires and Departures, even if None
    change_report_filename = f"coaches3_v2/change_report_{current_week_str}.txt"  # Filename includes week number
    with open(change_report_filename, "a", encoding="utf-8") as f:
        f.write(f"\n==== Change Tracking for Sheet: {sheet_name} ====\n")
        f.write(f"School: {school_name}\n")
        f.write(f"  New Hires: {', '.join(new_hires) if new_hires else 'None'}\n") 
        f.write(f"  Departures: {', '.join(departures) if departures else 'None'}\n")

async def fill_missing_data(coach_data, current_week):
    current_week_obj = isoweek.Week.thisweek()
    if current_week_obj.week == 1:
        previous_week_obj = isoweek.Week(current_week_obj.year - 1, isoweek.Week.last_week_of_year(current_week_obj.year - 1).week)
    else:
        previous_week_obj = isoweek.Week(current_week_obj.year, current_week_obj.week - 1)

    previous_week_str = f"{previous_week_obj.year}-W{previous_week_obj.week:02d}"
    previous_week_collection_name = f"coaches_{previous_week_str[:4]}_{previous_week_str[-2:]}"

    # Check if the previous week's collection exists
    if previous_week_collection_name in historic_db.list_collection_names():
        previous_week_collection = historic_db[previous_week_collection_name]

        for coach in coach_data["coachingStaff"]:
            # Find matching coach in the previous week's data
            matching_coach = previous_week_collection.find_one({"school": coach_data["school"], "coach.name": coach["name"]})
            if matching_coach:
                # Fill in missing attributes
                for attr in ["title", "email", "phone", "twitter"]:
                    if not coach.get(attr) and matching_coach["coach"].get(attr):
                        coach[attr] = matching_coach["coach"][attr]

def combine_coach_data(staff_directory_coaches, coaches_url_coaches):
    # Prioritize staff directory data, fill in missing attributes from coaches URL
    combined_coaches = staff_directory_coaches.copy()
    for coach_url in coaches_url_coaches:
        for coach_sd in combined_coaches:
            if coach_url['name'] == coach_sd['name']:
                for attr in ['title', 'email', 'phone', 'twitter']:
                    if not coach_sd.get(attr) and coach_url.get(attr):
                        coach_sd[attr] = coach_url[attr]
                break
        else:
            # Coach from coaches URL not found in staff directory, add them
            combined_coaches.append(coach_url)
    return combined_coaches

# version for semaphores
async def process_sheet(sheet_name, df):
    all_results = []
    total_tokens_used = 0
    
    staff_directory_column = next((col for col in df.columns if 'Staff Directory' in col), None)
    coaches_url_column = next((col for col in df.columns if 'Coaches URL' in col), None)

    if staff_directory_column or coaches_url_column:
        logger.info(f"\nProcessing URLs for sheet: {sheet_name}")

        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

        async def process_with_semaphore(row):
            async with semaphore:
                return await process_school(row, sheet_name, staff_directory_column, coaches_url_column)

        tasks = [process_with_semaphore(row) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)

        successful_scrapes = sum(1 for r in results if r['success'])
        failed_scrapes = len(results) - successful_scrapes
        tokens_used = sum(r['total_tokens'] for r in results)
        total_tokens_used += tokens_used

        logger.info(f"\nResults for {sheet_name}:")
        logger.info(f"Successful scrapes: {successful_scrapes}")
        logger.info(f"Failed scrapes: {failed_scrapes}")
        logger.info(f"Tokens used: {tokens_used}")

        save_results(results, f"coaches3_v2/scraping-results/{sheet_name}_results.json")
        save_failed_schools(results, f"coaches3_v2/scraping-results/{sheet_name}_failed_schools.txt")

        all_results.extend(results)
    else:
        logger.warning(f"No 'Staff Directory' or 'Coaches URL' column found in sheet: {sheet_name}")

    logger.info(f"\nTotal tokens used for {sheet_name}: {total_tokens_used}")
    return all_results, total_tokens_used

# version for asyncio.gather
# async def process_sheet(sheet_name, df):
#     all_results = []
#     total_tokens_used = 0

#     staff_directory_column = next((col for col in df.columns if 'Staff Directory' in col), None)
#     coaches_url_column = next((col for col in df.columns if 'Coaches URL' in col), None)

#     if staff_directory_column or coaches_url_column:
#         logger.info(f"\nProcessing URLs for sheet: {sheet_name}")
        
#         tasks = [process_school(row, sheet_name, staff_directory_column, coaches_url_column) 
#                  for _, row in df.iterrows()]
#         results = await asyncio.gather(*tasks)

#         successful_scrapes = sum(1 for r in results if r['success'])
#         failed_scrapes = len(results) - successful_scrapes
#         tokens_used = sum(r['total_tokens'] for r in results)
#         total_tokens_used += tokens_used

#         logger.info(f"\nResults for {sheet_name}:")
#         logger.info(f"Successful scrapes: {successful_scrapes}")
#         logger.info(f"Failed scrapes: {failed_scrapes}")
#         logger.info(f"Tokens used: {tokens_used}")

#         save_results(results, f"coaches3_v2/scraping-results/{sheet_name}_results.json")
#         save_failed_schools(results, f"coaches3_v2/scraping-results/{sheet_name}_failed_schools.txt")

#         all_results.extend(results)
#     else:
#         logger.warning(f"No 'Staff Directory' or 'Coaches URL' column found in sheet: {sheet_name}")

#     logger.info(f"\nTotal tokens used for {sheet_name}: {total_tokens_used}")
#     return all_results, total_tokens_used

def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)

def save_failed_schools(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    failed_schools = [f"{r['school']}: {r['url']} - Reason: {r['reason']}" for r in results if not r['success']]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(failed_schools))

def delete_old_output(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    print(f"Deleted old output in {directory}")

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

# Save terminal output to a file with a timestamp
output_filename = f"coaches3_v2/terminal_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
tee = Tee(output_filename)
sys.stdout = tee

async def main():
    global api_key_manager
    input_file = r"C:\Users\dell3\source\repos\school-data-scraper-4\Freelancer_Data_Mining_Project_mini.xlsx"

    delete_old_output("coaches3_v2/scraping-results")
    delete_old_output("coaches3_v2/raw_gemini_outputs")
    delete_old_output("coaches3_v2/body_html_contents")

    api_key_manager = APIKeyManager(GEMINI_API_KEYS)
    total_tokens_used = 0

    try:
        # ==== CLEAR CURRENT DB's COACHES COLLECTION ====
        coaches_collection_current.delete_many({})

        # ==== CLEAR HISTORIC DB's COLLECTION FOR CURRENT WEEK (IF EXISTS) ====
        current_week = isoweek.Week.thisweek()
        current_week_str = f"{current_week.year}-W{current_week.week:02d}"
        historic_collection_name = f"coaches_{current_week_str[:4]}_{current_week_str[-2:]}"
        if historic_collection_name in historic_db.list_collection_names():
            historic_db[historic_collection_name].delete_many({})
        
        # ==== PROCESS SHEETS ====
        xls = await load_excel_data(input_file)
        if xls is not None:
            for sheet_name in xls.sheet_names:
                if sheet_name in ["Softball Conferences", "Baseball Conferences", "DNU_States", "Freelancer Data"]:
                    continue
                logger.info(f"\nProcessing sheet: {sheet_name}")
                df = pd.read_excel(xls, sheet_name=sheet_name)
                _, sheet_tokens = await process_sheet(sheet_name, df)  # Get results and tokens directly
                total_tokens_used += sheet_tokens

            logger.info(f"\nTotal tokens used across all sheets: {total_tokens_used}")
        else:
            logger.error("Failed to load Excel file. Exiting.")
        
        # # ==== PROCESS SHEETS ====
        # xls = await load_excel_data(input_file)
        # if xls is not None:
        #     sheet_tasks = []
        #     for sheet_name in xls.sheet_names:
        #         if sheet_name in ["Softball Conferences", "Baseball Conferences", "DNU_States", "Freelancer Data"]:
        #             continue
        #         logger.info(f"\nProcessing sheet: {sheet_name}")
        #         df = pd.read_excel(xls, sheet_name=sheet_name)
        #         sheet_tasks.append(process_sheet(sheet_name, df))

        #     results = await asyncio.gather(*sheet_tasks)

        #     for _, sheet_tokens in results:
        #         total_tokens_used += sheet_tokens

        #     logger.info(f"\nTotal tokens used across all sheets: {total_tokens_used}")
        # else:
        #     logger.error("Failed to load Excel file. Exiting.")



        # ==== FILL MISSING DATA from previous week ====
        for coach_data in coaches_collection_current.find():
            await fill_missing_data(coach_data, current_week)
            coaches_collection_current.update_one({"_id": coach_data["_id"]}, {"$set": coach_data})

    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")
    finally:
        await global_driver_pool.shutdown()  # Ensure graceful shutdown of WebDriver pool
        # Close all WebDrivers in the pool (force kill any remaining Chrome instances)
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] == 'chrome.exe' and '--headless' in proc.info['cmdline']:
                try:
                    print(f"Killing headless Chrome process with PID: {proc.info['pid']}")
                    proc.kill()
                except psutil.NoSuchProcess:
                    print(f"Process with PID {proc.info['pid']} no longer exists.")


if __name__ == "__main__":
    start_time = datetime.now()

    def signal_handler(signum, frame):
        print("\nReceived signal, shutting down gracefully...")
        loop = asyncio.get_event_loop()
        loop.create_task(global_driver_pool.shutdown())
        loop.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(main())

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Total execution time: {duration}")
    # tee.close()

# Modifications compared to coaches3.py - 1) terminate chrome instances properly upon completion. 2) make sure ALL terminal output is stored in txt file 3) fix bug in validation - phone number was not being included in input
# TODO: modify extract_relevant_html to further reduce token usage
# TODO: create more api keys

# TODO: add timeout for retries - https://claude.ai/chat/3847ce8f-2297-41aa-987f-cdfc11aaa889

# TODO: for colleges where validation failed for all coaches - add them to failed urls list

# Instead of clearing entire current coaches collection only clear data for one school at a time?