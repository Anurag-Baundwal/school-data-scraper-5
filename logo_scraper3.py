import asyncio
import os
import re
import requests
import pandas as pd
from urllib.parse import urlparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Replace with your actual API key and Search Engine ID
GOOGLE_API_KEY = "AIzaSyDjufCWUGFzTBdtJalsTi7EeorKScgybWc"
SEARCH_ENGINE_ID = "d626f24be7e0045ed"

# Set a User-Agent to comply with website policies (especially Wikimedia)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36" # Example User-Agent

async def download_and_save_image(url, filepath):
    try:
        headers = {'User-Agent': USER_AGENT} # Add User-Agent header
        response = requests.get(url, stream=True, timeout=10, headers=headers)  # Added timeout
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Saved image: {filepath}")
        return True  # Return True on success

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return False  # Return False on failure


async def scrape_logos(school_data, sheet_name):
    school_name = school_data['School'].replace(" ", "_") #Added replace so no spaces in name
    # Sanitize the school name for filename compatibility:
    school_name = re.sub(r'[\\/*?:"<>|]', "", school_name)  # Remove invalid characters
    nickname = school_data.get('Nickname', '') #Get nickname if present
    query = f"{school_name} {nickname} athletics logo"
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&searchType=image&num=10&safe=active"  # safe search enabled

    try:
        response = requests.get(url)
        response.raise_for_status()
        search_results = response.json()

        if 'items' not in search_results:
            logger.warning(f"No images found for {school_name}")
            return

        output_dir = Path(f"logos/{sheet_name}/{school_name}")
        output_dir.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist

        top_10_dir = output_dir / "top_10_results"
        top_10_dir.mkdir(parents=True, exist_ok=True)

        downloaded_urls = set()  # Keep track of downloaded URLs to avoid duplicates

        num_results_to_download = min(10, len(search_results['items'])) # Don't try to download more than available
        # print(f"Number of results: {len(search_results['items'])}")

        for i in range(num_results_to_download):
            item = search_results['items'][i]
            image_url = item['link']

            if image_url in downloaded_urls:  # Skip if already downloaded
                continue
            downloaded_urls.add(image_url)

            file_ext = Path(urlparse(image_url).path).suffix

            if i < 5:  # Save the top 5 to the main directory
                filepath = output_dir / f"image_{i + 1}{file_ext}"
                await download_and_save_image(image_url, filepath)

            # Save all top 10 to the subdirectory
            filepath_top_10 = top_10_dir / f"image_{i + 1}{file_ext}"
            await download_and_save_image(image_url, filepath_top_10)


    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching for logos for {school_name}: {e}")

async def process_sheet(sheet_name, df):

    tasks = [scrape_logos(row, sheet_name) for _, row in df.iterrows()]
    await asyncio.gather(*tasks)




async def main():
    # input_file = r"C:\Users\dell3\source\repos\school-data-scraper-4\Freelancer_Data_Mining_Project_mini.xlsx"
    input_file = r"C:\Users\dell3\source\repos\school-data-scraper-5\Freelancer_Data_Mining_Project_Softball.xlsx"
    xls = pd.ExcelFile(input_file)

    if not xls:
        logger.error("Failed to load Excel file. Exiting.")
        return

    for sheet_name in xls.sheet_names:
        if sheet_name in ["Softball Conferences", "Baseball Conferences", "DNU_States", "Freelancer Data"]:
            continue 
        
        df = pd.read_excel(xls, sheet_name=sheet_name)
        logger.info(f"Processing sheet: {sheet_name}")
        await process_sheet(sheet_name, df)
        logger.info(f"Finished processing sheet: {sheet_name}")


if __name__ == "__main__":
    asyncio.run(main())