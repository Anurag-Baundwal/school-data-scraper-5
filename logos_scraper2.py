import asyncio
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from multiprocessing import Pool, Lock
from urllib.parse import urljoin
import time

# Set up logging (optional)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global lock for directory creation
dir_lock = Lock()


def setup_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def scrape_logos(school_data, sheet_name):
    school_name = school_data['School'].replace(" ", "_")
    url = school_data['Softball Recruitment Form']

    try:
        driver = setup_driver()
        driver.get(url)
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        img_tags = soup.find_all('img')

        with dir_lock:
            output_dir = f"logos/{sheet_name}/{school_name}"
            os.makedirs(output_dir, exist_ok=True)

        for i, img_tag in enumerate(img_tags):
            img_url = img_tag.get('src')
            if img_url:
                try:
                    img_url = urljoin(url, img_url)

                    # Image Filtering Logic
                    width = int(img_tag.get('width', 0))
                    height = int(img_tag.get('height', 0))
                    if width < 50 or height < 50:
                        continue

                    if 'logo' in img_tag.get('class', []) or 'logo' in img_tag.get('alt', '').lower():
                        pass 
                    else:
                        continue

                    parent_element = img_tag.parent
                    if parent_element.name == 'a' and 'logo' in parent_element.get('href', '').lower():
                        pass
                    else:
                        continue

                    response = requests.get(img_url, stream=True)
                    response.raise_for_status()

                    file_extension = os.path.splitext(img_url)[1] or ".jpg"
                    file_path = os.path.join(output_dir, f"image_{i}{file_extension}")

                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    logger.info(f"Saved image {i} for {school_name} from {url}")
                except Exception as e:
                    logger.error(f"Error saving image {i} for {school_name}: {str(e)}")

    except Exception as e:
        logger.error(f"Error scraping logos for {school_name}: {str(e)}")
    finally:
        if driver:
            driver.quit()


def process_sheet(sheet_name, df):
    logger.info(f"Processing sheet: {sheet_name}")

    with Pool() as pool:  
        pool.starmap(scrape_logos, [(row, sheet_name) for _, row in df.iterrows()])


async def main():
    input_file = r"C:\Users\dell3\source\repos\school-data-scraper-4\Freelancer_Data_Mining_Project_mini.xlsx"

    xls = pd.ExcelFile(input_file)
    if xls is not None:
        for sheet_name in xls.sheet_names:
            if sheet_name in ["Softball Conferences", "Baseball Conferences", "DNU_States", "Freelancer Data"]:
                continue
            
            df = pd.read_excel(xls, sheet_name=sheet_name)
            process_sheet(sheet_name, df) 

    else:
        logger.error("Failed to load Excel file. Exiting.")


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())  # Use asyncio for main function
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Total execution time: {duration:.2f} seconds")