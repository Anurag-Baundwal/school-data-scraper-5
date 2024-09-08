import asyncio
import re
import shutil
import signal
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
import json
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import psutil
from config import GEMINI_API_KEYS
import random
import logging
from datetime import datetime
import os
from urllib.parse import urlparse
from fuzzywuzzy import fuzz
import json5
import ssl
import pymongo
import isoweek
import time

# ==== MAKE DIR TO STORE OUTPUT ON FILESYSTEM ====
# Create the directory if it doesn't exist
os.makedirs('rosters3', exist_ok=True)
os.makedirs('rosters3/scraping-results', exist_ok=True)

# ==== CONFIGURE LOGGING ====
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Add a file handler for DEBUG level
file_handler = logging.FileHandler('rosters3/logs.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


# ==== MONGO DB ====
# MongoDB Connection (same as before)
client = pymongo.MongoClient("mongodb://localhost:27017")
historic_db = client["historic_data_softball"]
current_db = client["current_data_softball"]

# Collections (created if they don't exist)
players_collection_current = current_db["players"]


# Add a file handler for DEBUG level
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
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

async def fetch_with_retry(url, headers, max_retries=3, timeout=20):
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


def log_raw_gemini_output(school_name, raw_output, sheet_name):
    # Log raw output
    logger.debug(f"Raw Gemini output for {school_name}:\n{raw_output}")

    # Save raw output to file
    output_dir = "rosters1/raw_gemini_outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{sheet_name}_raw_gemini_outputs.txt")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Raw Gemini output for {school_name}:\n")
        f.write(raw_output)
        f.write("\n\n")  # Add two newlines to separate outputs

    logger.info(f"Raw Gemini output for {school_name} saved to {file_path}")

class APIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_index = 0

    # def get_next_key(self):
    #     key = self.api_keys[self.current_index]
    #     self.current_index = (self.current_index + 1) % len(self.api_keys)
    #     return key
    
    def get_next_key(self): # get_random_key
        return random.choice(self.api_keys)

api_key_manager = APIKeyManager(GEMINI_API_KEYS)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, str):
            return obj.encode('utf-8').decode('unicode_escape')
        return super().default(obj)

async def load_excel_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        return xls
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        return None

def validate_player_data(players, body_html):
    valid_players = []
    html_content_lowercasecase = body_html.lower()
    
    for player in players:
        if player.get('name') and body_html:
            name_lowercase = player['name'].lower()
            
            # First, try an exact match (case-insensitive)
            if player['name'] in body_html:
                logger.debug(f"Exact match found for {player['name']}")
                valid_players.append(player)
                continue

            # If no exact match, try fuzzy matching
            match_ratio = fuzz.partial_ratio(name_lowercase, html_content_lowercasecase)
            logger.debug(f"Fuzzy match ratio for {player['name']}: {match_ratio}")

            if match_ratio >= 85:
                # Check if at least two other fields are found near the name
                name_index = html_content_lowercasecase.find(name_lowercase)
                if name_index != -1:
                    surrounding_text = html_content_lowercasecase[max(0, name_index - 2000):min(len(html_content_lowercasecase), name_index + 10000)]
                    fields_found = sum(1 for field in ['position', 'year', 'hometown', 'highSchool'] 
                                       if player.get(field) and str(player[field]).lower() in surrounding_text)
                    
                    logger.debug(f"Fields found for {player['name']}: {fields_found}")

                    if fields_found >= 2:
                        valid_players.append(player)
                    else:
                        logger.debug(f"Player {player['name']} rejected: Only {fields_found} fields found")
                else:
                    logger.debug(f"Player {player['name']} rejected: Name not found in body text")
            else:
                logger.debug(f"Player {player['name']} rejected: Fuzzy match ratio below threshold")
        else:
            logger.debug(f"Player rejected: Missing name or body text. Player data: {player}")
    
    logger.info(f"Total players: {len(players)}, Valid players: {len(valid_players)}")
    return valid_players

def save_body_html_content(school_name, body_html, sheet_name):
    output_dir = "coaches3_v2/body_html_contents"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{sheet_name}_body_html_contents.txt")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Body HTML content for {school_name}:\n")
        f.write(body_html)
        f.write("\n\n" + "="*50 + "\n\n")

    logger.info(f"Body HTML content for {school_name} saved to {file_path}")

async def gemini_based_scraping(url, school_name, nickname, sheet_name):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
    
    headers = {'User-Agent': random.choice(user_agents)}
    html_content, fetch_error = await fetch_with_retry(url, headers, max_retries=4, timeout=20) 
    if html_content is None:
        return None, False, 0, 0 
    
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.body
    if body is None:
        logger.warning(f"No body tag found in the HTML from {url}")
        return None, False, 0, 0
    body_html = str(body)
    save_body_html_content(school_name, body_html, sheet_name)
    
    try:
        current_year = datetime.now().year
        genai.configure(api_key=api_key_manager.get_next_key())
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192*16, # 200k / 128k - 16k, 32k 
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-exp-0827",
            generation_config=generation_config
        )
        
        prompt = f"""
        Analyze the HTML content of the college softball roster webpage from {url}. The expected school name is "{school_name}" and the team nickname or name should be related to "{nickname}". Focus ONLY on player information, ignoring any coach or staff data that might be present. Extract the following information for each player:
        - Name
        - Position
        - Year (Fr, So, Jr, Sr, Grad, etc)
        - Hometown
        - High School
        - Graduation Year (calculate based on the player's year and the roster year)
        Determine the roster year. Look for an explicit mention of the roster year on the page (e.g., "2024 Softball Roster"). If not found, assume it's for the upcoming season ({current_year + 1}). Also, if the roster year consists of a range then it's the first year in the range (e.g.,  "2024-2025 Softball Roster" means that the current roster year is 2024).
        For the Graduation Year calculation, use the determined roster year as the base:
        - Freshman (Fr) or First Year: Roster Year + 3
        - Sophomore (So) or Second Year: Roster Year + 2
        - Junior (Jr) or Third Year: Roster Year + 1
        - Senior (Sr) or Fourth Year: Roster Year
        - Graduate (Grad) or Fifth Year: Roster Year
        - If the year is unclear, set to null
        Format the output as a JSON string with the following structure:
        {{
            "success": true/false,
            "reason": "reason for failure" (or null if success),
            "rosterYear": YYYY,
            "players": [
                {{
                    "name": "...",
                    "position": "...",
                    "year": "...",
                    "hometown": "...",
                    "highSchool": "...",
                    "graduationYear": YYYY
                }},
                ...
            ]
        }}
        Set "success" to false if:
        1. No player data is found
        2. Any player is missing one or more of the required fields (name, position, year, hometown, highSchool)
        3. The roster year cannot be determined
        If "success" is false, provide a brief explanation in the "reason" field.
        Important: Ensure all names, including those with non-English characters, are preserved exactly as they appear in the HTML. Do not escape or modify any special characters in names, hometowns, or school names. For example, 'Montañez' should remain as 'Montañez', not 'Monta\\u00f1ez', and "O'ahu" should remain as "O'ahu", not "O\\u2018ahu".
        The response should be a valid JSON string only, without any additional formatting or markdown syntax.
        Note: If no player data is found on the page, do not create any imaginary players or list players that are not present on the page. Instead, set the 'success' field to false and provide a reason indicating that no players were found on the roster page.
        """

        token_response = model.count_tokens(prompt + body_html)
        input_tokens = token_response.total_tokens

        response = await model.generate_content_async([prompt, body_html])
        
        output_tokens = model.count_tokens(response.text).total_tokens

        # Log the raw output immediately after receiving it
        log_raw_gemini_output(school_name, response.text, sheet_name)

        # Clean the response text
        cleaned_response = response.text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]

        cleaned_response = cleaned_response.strip()

        try:
            result = json.loads(cleaned_response)
            
            # Validate the player data
            if result['success'] and 'players' in result:
                valid_players = validate_player_data(result['players'], body_html)
                result['players'] = valid_players
                result['player_count'] = len(valid_players)
              

                result['player_count_mismatch'] = False

                if len(valid_players) == 0:
                    result['success'] = False
                    result['reason'] = "No valid players found after validation"
            else:
                # Ensure players and player_count are set even if not present in the original response
                result['players'] = []
                result['player_count'] = 0
                result['player_count_mismatch'] = False
            
            # Ensure all expected fields are present
            result.setdefault('success', False)
            result.setdefault('reason', "Unknown error")
            result.setdefault('rosterYear', None)
            result.setdefault('players', [])
            result.setdefault('player_count', 0)
            result.setdefault('player_count_mismatch', False)

            return result, result['success'], input_tokens, output_tokens
        except json.JSONDecodeError as json_error:
            logger.error(f"Failed to parse JSON from Gemini response for {school_name}: {json_error}")
            logger.debug(f"Raw response: {cleaned_response}")
            return {
                'success': False,
                'reason': f"Failed to parse JSON: {str(json_error)}",
                'rosterYear': None,
                'players': [],
                'player_count': 0,
                'player_count_mismatch': False
            }, False, input_tokens, output_tokens
    except Exception as e:
        logger.error(f"Error in Gemini-based scraping for {school_name}: {str(e)}")
        return {
            'success': False,
            'reason': f"Error in scraping: {str(e)}",
            'rosterYear': None,
            'players': [],
            'player_count': 0,
            'player_count_mismatch': False
        }, False, 0, 0

async def process_school(school_data, url_column, sheet_name):
    url = school_data[url_column]
    school_name = school_data['School']
    nickname = school_data.get('Nickname', '')
    max_retries = 3
    base_delay = 5  # seconds
    reasons = []
    total_input_tokens = total_output_tokens = 0

    if pd.notna(url):
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing {school_name} (URL: {url}) - Attempt {attempt + 1}")
                result, success, input_tokens, output_tokens = await gemini_based_scraping(url, school_name, nickname, sheet_name)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_tokens = total_input_tokens + total_output_tokens
                logger.info(f"Tokens used for {school_name} {url_column}: {total_tokens}")

                if success:
                    logger.info(f"Successfully scraped data for {school_name}")
                    player_count = len(result['players'])
                    if player_count >= 35:
                        with open(f"rosters3/scraping-results/{sheet_name}_urls_for_manual_review.txt", 'a') as f:
                            f.write(f"{school_name}: {url} - {player_count} players\n")
                        logger.warning(f"Large roster detected for {school_name}: {player_count} players. Added to manual review.")
                else:
                    reason = result['reason'] if result and 'reason' in result else 'Unknown error'
                    reasons.append(f"Attempt {attempt + 1}: {reason}")
                    logger.warning(f"Scraping failed for {school_name} - Attempt {attempt + 1}: {reason}")
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        await asyncio.sleep(delay)

                # ==== SAVE TO BOTH CURRENT AND HISTORIC DBs ====
                current_week = isoweek.Week.thisweek()
                current_week_str = f"{current_week.year}-W{current_week.week:02d}"
                # historic_collection_name = f"players_{current_week_str[:4]}_{current_week_str[-2:]}"
                # Get collection names
                current_collection_name = get_collection_name(sheet_name)
                historic_collection_name = get_collection_name(sheet_name, current_week_str)
                historic_collection = historic_db[historic_collection_name]
                players_collection_current = current_db[current_collection_name]

                player_data = {
                    "school": school_name,
                    "division": sheet_name,
                    "url": url,
                    "rosterYear": result.get('rosterYear', None),
                    "players": result.get('players', []),
                    "timestamp": datetime.now(),
                    "weekNumber": int(current_week_str[-2:]),
                    "year": int(current_week_str[:4])
                }
                players_collection_current.insert_one(player_data)
                historic_collection.insert_one(player_data)

                # ==== CHANGE TRACKING ====
                track_changes(player_data, sheet_name, current_week_str)

                if success:
                    return {
                        'school': school_name,
                        'url': url,
                        'success': True,
                        'reason': None,
                        'rosterYear': result['rosterYear'],
                        'players': result['players'],
                        'player_count': result['player_count'],
                        'input_tokens': total_input_tokens,
                        'output_tokens': total_output_tokens,
                        'total_tokens': total_tokens
                    }
                else:
                    return {
                        'school': school_name,
                        'url': url,
                        'success': False,
                        'reason': '; '.join(reasons),
                        'rosterYear': None,
                        'players': [],
                        'player_count': 0,
                        'input_tokens': total_input_tokens,
                        'output_tokens': total_output_tokens,
                        'total_tokens': total_tokens
                    }
            except Exception as e:
                reason = f"Error: {str(e)}"
                reasons.append(f"Attempt {attempt + 1}: {reason}")
                logger.error(f"Error in processing {school_name}: {reason} - Attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)

        # If all retries fail, still store data in MongoDB
        current_week = isoweek.Week.thisweek()
        current_week_str = f"{current_week.year}-W{current_week.week:02d}"
        player_data = {
            "school": school_name,
            "division": sheet_name,
            "url": url,
            "rosterYear": None,
            "players": [],
            "timestamp": datetime.now(),
            "weekNumber": int(current_week_str[-2:]),
            "year": int(current_week_str[:4])
        }
        players_collection_current.insert_one(player_data)

        return {
            'school': school_name,
            'url': url,
            'success': False,
            'reason': '; '.join(reasons),
            'rosterYear': None,
            'players': [],
            'player_count': 0,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_tokens
        }
    else:
        logger.info(f"Skipping {school_name} - No URL provided")
        # Store data in MongoDB even when no URL is provided
        current_week = isoweek.Week.thisweek()
        current_week_str = f"{current_week.year}-W{current_week.week:02d}"
        player_data = {
            "school": school_name,
            "division": sheet_name,
            "url": 'N/A',
            "rosterYear": None,
            "players": [],
            "timestamp": datetime.now(),
            "weekNumber": int(current_week_str[-2:]),
            "year": int(current_week_str[:4])
        }
        players_collection_current.insert_one(player_data)

        return {
            'school': school_name,
            'url': 'N/A',
            'success': False,
            'reason': 'No URL provided',
            'rosterYear': None,
            'players': [],
            'player_count': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }

def get_collection_name(sheet_name, week_str=None):
    sheet_name_formatted = sheet_name.replace(" ", "_")
    if week_str:
        return f"players_{week_str}_{sheet_name_formatted}"
    else:
        return f"players_current_{sheet_name_formatted}"
    
def track_changes(player_data, sheet_name, current_week_str):
    school_name = player_data['school']
    current_players = set(p['name'] for p in player_data['players'])

    # Get previous week's data
    if current_week_str.endswith("W01"):
        previous_week_obj = isoweek.Week(int(current_week_str[:4]) - 1, isoweek.Week.last_week_of_year(int(current_week_str[:4]) - 1).week)
    else:
        previous_week_obj = isoweek.Week(int(current_week_str[:4]), int(current_week_str[-2:]) - 1)
    previous_week_str = f"{previous_week_obj.year}-W{previous_week_obj.week:02d}"
    # previous_week_collection_name = f"players_{previous_week_str[:4]}_{previous_week_str[-2:]}"
    previous_week_collection_name = get_collection_name(sheet_name, previous_week_str)
    previous_players = set()
    if previous_week_collection_name in historic_db.list_collection_names():
        previous_week_data = historic_db[previous_week_collection_name].find_one({"school": school_name})
        if previous_week_data:
            previous_players = set(p['name'] for p in previous_week_data['players'])

    # Determine new and removed players
    if previous_players:
        new_players = current_players - previous_players
        removed_players = previous_players - current_players
    else:
        new_players = None  # Set to None if no previous week data
        removed_players = None 

    change_report_filename = f"rosters3/change_report_{current_week_str}.txt"
    with open(change_report_filename, "a", encoding="utf-8") as f:
        f.write(f"\n==== Change Tracking for Sheet: {sheet_name} ====\n")
        f.write(f"School: {school_name}\n")
        f.write(f"  New Players: {', '.join(new_players) if new_players else 'None'}\n") 
        f.write(f"  Removed Players: {', '.join(removed_players) if removed_players else 'None'}\n")

async def process_sheet(sheet_name, df):
    all_results = []
    total_tokens_used = 0

    semaphore = asyncio.Semaphore(10)

    async def process_with_semaphore(row, url_column):
        async with semaphore:
            return await process_school(row, url_column, sheet_name)

    roster_url_column = next((col for col in df.columns if 'Roster URL' in col), None)
    if roster_url_column:
        logger.info(f"\nProcessing {roster_url_column} URLs for sheet: {sheet_name}")
        
        tasks = [process_with_semaphore(row, roster_url_column) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)

        successful_scrapes = sum(1 for r in results if r['success'])
        failed_scrapes = len(results) - successful_scrapes
        tokens_used = sum(r['total_tokens'] for r in results)
        total_tokens_used += tokens_used

        logger.info(f"\nResults for {sheet_name} - {roster_url_column}:")
        logger.info(f"Successful scrapes: {successful_scrapes}")
        logger.info(f"Failed scrapes: {failed_scrapes}")
        logger.info(f"Tokens used: {tokens_used}")

        save_results(results, f"rosters1/scraping-results/{sheet_name}_{roster_url_column}_results.json")
        save_failed_schools(results, f"rosters1/scraping-results/{sheet_name}_{roster_url_column}_failed_schools.txt")

        all_results.extend(results)
    else:
        logger.warning(f"No 'Roster URL' column found in sheet: {sheet_name}")

    logger.info(f"\nTotal tokens used for {sheet_name}: {total_tokens_used}")
    return all_results, total_tokens_used

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
    
async def main():
    global api_key_manager
    
    # Delete old output
    delete_old_output("rosters3/scraping-results")
    delete_old_output("rosters3/raw_gemini_outputs")
    delete_old_output("rosters3/body_html_contents")
    api_key_manager = APIKeyManager(GEMINI_API_KEYS)
    total_tokens_used = 0
    
    sheet_to_process = "NCAA D1"

    try:
        # ==== CLEAR CURRENT DB's PLAYERS COLLECTION ====
        players_collection_current.delete_many({})

        # ==== READ INPUT FILE AND SCRAPE EACH SHEET ====
        input_file = r"C:\Users\dell3\source\repos\school-data-scraper-4\Freelancer_Data_Mining_Project_mini.xlsx"
        xls = await load_excel_data(input_file)
        if xls is not None:
            # Skip sheets which do not have colleges in them
            for sheet_name in xls.sheet_names:
                if sheet_name in ["Softball Conferences", "Baseball Conferences", "DNU_States", "Freelancer Data"]:
                    continue
                # Skip sheets except for the one we want to process in this run
                if sheet_name != sheet_to_process:
                    logger.info(f"Skipping sheet: {sheet_name}")
                    continue
                
                # ==== CLEAR OLD DATA FROM CURRENT WEEK'S COLLECTIONS ====               
                current_week = isoweek.Week.thisweek()
                current_week_str = f"{current_week.year}-W{current_week.week:02d}"
                current_collection_name = get_collection_name(sheet_name)
                historic_collection_name = get_collection_name(sheet_name, current_week_str)
                current_db[current_collection_name].delete_many({})
                if historic_collection_name in historic_db.list_collection_names():
                    historic_db[historic_collection_name].delete_many({})

                logger.info(f"\nProcessing sheet: {sheet_name}")
                df = pd.read_excel(xls, sheet_name=sheet_name)
                _, sheet_tokens = await process_sheet(sheet_name, df)
                total_tokens_used += sheet_tokens
            logger.info(f"\nTotal tokens used across all sheets: {total_tokens_used}")
        else:
            logger.error("Failed to load Excel file. Exiting.")
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