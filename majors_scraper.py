# college majors scraper
# previously named majors2.py
import os
import pandas as pd
import pymongo
import requests
from bs4 import BeautifulSoup
import logging
import asyncio
import aiohttp
import time
import random
import base64
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import google.generativeai as genai
import json
import re
from datetime import datetime
import shutil
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
import isoweek

# ===== MONGO DB SETUP =====
client = pymongo.MongoClient("mongodb://localhost:27017", maxPoolSize=10)
historic_db = client["historic_data_softball"]
current_db = client["current_data_softball"]

# Collections (created if they don't exist)
majors_collection_current = current_db["majors"]


# ==== MAKE DIR TO STORE OUTPUT ON FILESYSTEM ====
# Create the directory if it doesn't exist
os.makedirs('majors2', exist_ok=True)
os.makedirs('majors2/scraping-results', exist_ok=True)

# ==== CONFIGURE LOGGING ====
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Add a file handler for DEBUG level
file_handler = logging.FileHandler('majors2/logs.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


def google_search(query, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res['items']

def extract_majors(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    majors = []

    # Type 1: div with style="padding:15px;"
    majors_div = soup.find('div', style="padding:15px;")
    if majors_div:
        majors = [h3.text.strip() for h3 in majors_div.find_all('h3')]

    # Type 2: table with id='majortable'
    if not majors:
        majors_table = soup.find('table', id='majortable')
        if majors_table:
            for row in majors_table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if cells:
                    majors.append(cells[0].text.strip())

    # Type 3: span elements with class="parent-line"
    if not majors:
        parent_lines = soup.find_all('span', class_="parent-line")
        for line in parent_lines:
            major = line.find('p')
            if major:
                majors.append(major.text.strip())

    # Type 4: Look for lists (ul, ol) containing potential majors
    if not majors:
        for list_elem in soup.find_all(['ul', 'ol']):
            items = list_elem.find_all('li')
            if len(items) > 5:  # Assuming a list of majors would have more than 5 items
                majors.extend([item.text.strip() for item in items])

    # Type 5: Look for header elements (h1, h2, h3) containing "major" or "program"
    if not majors:
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        for header in headers:
            if 'major' in header.text.lower() or 'program' in header.text.lower():
                next_elem = header.find_next_sibling()
                if next_elem and next_elem.name in ['ul', 'ol']:
                    majors.extend([item.text.strip() for item in next_elem.find_all('li')])

    # If still no majors found, try to find any p tags within the main content
    if not majors:
        content_div = soup.find('div', id='MajorsOffered')
        if content_div:
            majors = [p.text.strip() for p in content_div.find_all('p') if p.text.strip()]

    return list(set(majors))  # Remove duplicates


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

async def process_school(school_data, sheet_name, majors_url_column):
    school_name = school_data['School']
    max_retries = 3
    base_delay = 5
    final_result = {}
    reasons = []

    for attempt in range(max_retries):
        user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        try:
            if majors_url_column and pd.notna(school_data[majors_url_column]):
                url = school_data[majors_url_column]
                logger.info(f"Processing {school_name} (Majors URL: {url}) - Attempt {attempt + 1}")
                
                headers = {'User-Agent': random.choice(user_agents)}
                html_content, error = await fetch_with_retry(url, headers, max_retries=4, timeout=25)

                if html_content:
                    majors = extract_majors(html_content)
                    if majors:
                        logger.info(f"Successfully scraped majors for {school_name}")
                        
                        final_result = {
                            'school': school_name,
                            'url': url,
                            'success': True,
                            'reason': None,
                            'majors': majors,
                            'num_majors': len(majors)
                        }
                    else:
                        reasons.append(f"Majors URL - Attempt {attempt + 1}: No majors found on page")
                        logger.warning(f"Scraping failed for {school_name} (Majors URL) - Attempt {attempt + 1}: No majors found on page")
                else:
                    reasons.append(f"Majors URL - Attempt {attempt + 1}: {error}")
                    logger.warning(f"Scraping failed for {school_name} (Majors URL) - Attempt {attempt + 1}: {error}")
            
            if final_result.get('success'):
                break
            else:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        except Exception as e:
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
            'majors': [],
            'num_majors': 0
        }

    # ==== SAVE TO BOTH CURRENT AND HISTORIC DBs ====
    current_week = isoweek.Week.thisweek()
    current_week_str = f"{current_week.year}-W{current_week.week:02d}"
    historic_collection_name = f"majors_{current_week_str[:4]}_{current_week_str[-2:]}"
    historic_collection = historic_db[historic_collection_name]
    
    major_data = {
        "school": final_result['school'],
        "division": sheet_name, 
        "url": final_result['url'],
        "majors": final_result['majors'] if final_result['success'] else [],  
        "timestamp": datetime.now(),
        "weekNumber": int(current_week_str[-2:]),
        "year": int(current_week_str[:4])
    }
    majors_collection_current.insert_one(major_data)
    historic_collection.insert_one(major_data) 

    # ==== CHANGE TRACKING ====
    track_changes(major_data, sheet_name, current_week_str)
    return final_result

def track_changes(major_data, sheet_name, current_week_str):
    school_name = major_data['school']
    current_majors = set(major_data['majors'])

    # Get previous week's data
    if current_week_str.endswith("W01"):
        previous_week_obj = isoweek.Week(int(current_week_str[:4]) - 1, isoweek.Week.last_week_of_year(int(current_week_str[:4]) - 1).week)
    else:
        previous_week_obj = isoweek.Week(int(current_week_str[:4]), int(current_week_str[-2:]) - 1)
    previous_week_str = f"{previous_week_obj.year}-W{previous_week_obj.week:02d}"
    previous_week_collection_name = f"majors_{previous_week_str[:4]}_{previous_week_str[-2:]}"

    previous_majors = set()
    if previous_week_collection_name in historic_db.list_collection_names():
        previous_week_data = historic_db[previous_week_collection_name].find_one({"school": school_name})
        if previous_week_data:
            previous_majors = set(previous_week_data['majors'])

    # Determine new and removed majors only if previous week data exists
    if previous_majors:
        new_majors = current_majors - previous_majors
        removed_majors = previous_majors - current_majors
    else:
        new_majors = None
        removed_majors = None

    # Always mention both New Majors and Removed Majors, even if None
    change_report_filename = f"majors2/change_report_{current_week_str}.txt"  # Filename includes week number
    with open(change_report_filename, "a", encoding="utf-8") as f:
        f.write(f"\n==== Change Tracking for Sheet: {sheet_name} ====\n")
        f.write(f"School: {school_name}\n")
        f.write(f"  New Majors: {', '.join(new_majors) if new_majors else 'None'}\n") 
        f.write(f"  Removed Majors: {', '.join(removed_majors) if removed_majors else 'None'}\n")

async def process_sheet(sheet_name, df):
    all_results = []

    majors_url_column = next((col for col in df.columns if 'Undergraduate Majors URL' in col), None)

    if majors_url_column:
        logger.info(f"\nProcessing URLs for sheet: {sheet_name}")
        
        tasks = [process_school(row, sheet_name, majors_url_column) 
                 for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)

        successful_scrapes = sum(1 for r in results if r['success'])
        failed_scrapes = len(results) - successful_scrapes

        logger.info(f"\nResults for {sheet_name}:")
        logger.info(f"Successful scrapes: {successful_scrapes}")
        logger.info(f"Failed scrapes: {failed_scrapes}")

        save_results(results, sheet_name)
        save_failed_schools(results, sheet_name)

        all_results.extend(results)
    else:
        logger.warning(f"No 'Undergraduate Majors URL' column found in sheet: {sheet_name}")

    return all_results

def save_results(results, sheet_name):
    # Save to Excel and CSV
    results_data = []
    max_majors = max([len(majors) for majors in [r['majors'] for r in results]] + [0])
    columns = ['School', 'Num_Majors', 'Scraping_Method', 'URL_Used'] + [f'Major_{i+1}' for i in range(max_majors)]

    for result in results:
        sorted_majors = sorted(result['majors'])
        row_data = {
            'School': result['school'], 
            'Num_Majors': len(sorted_majors),
            'Scraping_Method': 'Provided URL' if result['success'] else 'Failed',  # Indicate if scraping was successful
            'URL_Used': result['url']
        }
        for i, major in enumerate(sorted_majors, 1):
            row_data[f'Major_{i}'] = major
        results_data.append(row_data)

    results_df = pd.DataFrame(results_data, columns=columns)
    results_df = results_df.sort_values('School')

    # Save to a single Excel file with multiple sheets
    excel_file = 'scraped_majors_data.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a' if os.path.exists(excel_file) else 'w') as writer:
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save to CSV
    results_df.to_csv(f'scraped_majors_data_{sheet_name}.csv', index=False)

def save_failed_schools(results, sheet_name):
    # Save failed URLs to Excel and text file
    failed_urls = [(r['school'], r['url'], r['reason']) for r in results if not r['success']]
    failed_df = pd.DataFrame(failed_urls, columns=['School', 'URL', 'Reason'])
    failed_df = failed_df.sort_values('School')
    
    with pd.ExcelWriter('failed_urls.xlsx', engine='openpyxl', mode='a' if os.path.exists('failed_urls.xlsx') else 'w') as writer:
        failed_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    with open("failed_urls.txt", "a") as f:
        f.write(f"\n{'=' * 50}\n")
        f.write(f"Failed URLs for sheet: {sheet_name}\n")
        f.write(f"{'=' * 50}\n\n")
        for _, row in failed_df.iterrows():
            f.write(f"{row['School']} - {row['URL']} - {row['Reason']}\n")
        f.write("\n")

def delete_old_output(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    print(f"Deleted old output in {directory}")

async def main():
    # ==== CLEAR OLD OUTPUT FILES ====
    delete_old_output("majors2/scraping-results")
    delete_old_output("majors2/raw_gemini_outputs")
    delete_old_output("majors2/body_html_contents")
    # Clear existing output files and screenshots
    output_files = ['scraped_majors_data.xlsx', 'failed_urls.xlsx', 'failed_urls.txt']
    for file in output_files:
        if os.path.exists(file):
            os.remove(file)
    
    if os.path.exists('failed_screenshots'):
        shutil.rmtree('failed_screenshots')
    os.makedirs('failed_screenshots', exist_ok=True)


    # ==== CLEAR CURRENT DB's MAJORS COLLECTION ====
    majors_collection_current.delete_many({})

    # ==== CLEAR HISTORIC DB's COLLECTION FOR CURRENT WEEK (IF EXISTS) ====
    current_week = isoweek.Week.thisweek()
    current_week_str = f"{current_week.year}-W{current_week.week:02d}"
    historic_collection_name = f"majors_{current_week_str[:4]}_{current_week_str[-2:]}"
    if historic_collection_name in historic_db.list_collection_names():
        historic_db[historic_collection_name].delete_many({})
    
    input_file = r"C:\Users\dell3\source\repos\school-data-scraper-4\Freelancer_Data_Mining_Project_mini.xlsx"   
    xls = pd.ExcelFile(input_file)
    
    for sheet_name in xls.sheet_names:
        if sheet_name in ["Softball Conferences", "Baseball Conferences", "DNU_States", "Freelancer Data"]:
            continue
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing sheet: {sheet_name}")
        logger.info(f"{'=' * 50}\n")
        
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        await process_sheet(sheet_name, df)
        
        logger.info(f"\nFinished processing sheet: {sheet_name}")
        logger.info(f"{'=' * 50}\n")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

# changes - input file, skip some sheets, add scroll to bottom function
# todo - correct monogodb integration, add change tracking, store output files in correct folder.