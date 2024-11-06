"""
College Athletics Logo Scraper

This script automates the process of finding and analyzing college athletics logos.
It performs the following steps:
1. Searches for images related to a specified college's athletics program using Google Custom Search API.
2. Downloads and saves the found images, handling various file formats including SVG.
3. Converts SVG files to PNG format for analysis using a web-based conversion service.
4. Analyzes the images using Google's Gemini AI to identify the most likely official logos.
5. Distinguishes between logos with text (e.g., college name) and those without (e.g., just the mascot).
6. Saves the identified logos separately and provides detailed information about them.

The script uses both image indices and filenames to ensure accurate identification and saving of logo files.
It handles potential errors and provides detailed logging throughout the process.

This tool is designed to help quickly gather official athletics logos for multiple colleges,
which can be useful for sports information departments, media outlets, or fan websites.

Usage:
1. Ensure all required libraries are installed (see requirements.txt).
2. Set up the necessary API keys for Google Custom Search and Gemini AI.
3. Run the script, providing the college name and nickname when prompted.

The script will output detailed logs of its process and save identified logos in a separate folder.
"""

import requests
import base64
import google.generativeai as genai
# import os
# from dotenv import load_dotenv
import json
import random
from PIL import Image
from io import BytesIO
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#------------------- API KEYS ----------------------------------------
# Python code to load and parse the environment variables:
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Parse Gemini API keys as a list
GEMINI_API_KEYS = os.getenv('GEMINI_API_KEYS').split(',')

# Load other environment variables
OXYLABS_USERNAME = os.getenv('OXYLABS_USERNAME')
OXYLABS_PASSWORD = os.getenv('OXYLABS_PASSWORD')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

#--------------------------------------------------------------------------

def get_random_gemini_api_key():
    return random.choice(GEMINI_API_KEYS)

def search_images(query, num_images=10):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'q': query,
        'searchType': 'image',
        'num': num_images,
        'fileType': 'svg,png,jpg',  # Prioritize SVG
        'imgType': 'clipart',
        'safe': 'active'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        
        logging.info("Google Custom Search API Response:")
        logging.info(json.dumps(results, indent=2))
        
        if 'items' not in results:
            logging.warning("No 'items' found in the API response.")
            if 'error' in results:
                logging.error(f"API Error: {results['error']['message']}")
            return []
        
        return [item['link'] for item in results['items']]
    except requests.exceptions.RequestException as e:
        logging.error(f"Error in Google Custom Search API request: {str(e)}")
        return []

def convert_svg_to_png(svg_url):
    conversion_url = f"https://convert.svgtopng.com/api/convert/url?url={svg_url}"
    try:
        response = requests.get(conversion_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logging.error(f"Error converting SVG to PNG: {str(e)}")
        return None
def download_and_save_image(url, index, college_name):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        os.makedirs(f"downloaded_images/{college_name}", exist_ok=True)
        
        content_type = response.headers.get('content-type', '').lower()
        if 'svg' in content_type:
            filename = f"downloaded_images/{college_name}/image_{index}.svg"
            with open(filename, 'wb') as f:
                f.write(response.content)
            logging.info(f"Saved SVG: {filename}")
            
            # Generate base64 representation of SVG
            img_data = base64.b64encode(response.content).decode('utf-8')
            return img_data, 'image/svg+xml', filename, None
        else:
            img = Image.open(BytesIO(response.content))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            filename = f"downloaded_images/{college_name}/image_{index}.png"
            img.save(filename, 'PNG')
            logging.info(f"Saved image: {filename}")
            with open(filename, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            return img_data, 'image/png', filename, None
    except Exception as e:
        logging.error(f"Error downloading/processing image from {url}: {str(e)}")
    return None, None, None, None

def analyze_images_with_gemini(image_urls, college_name):
    genai.configure(api_key=get_random_gemini_api_key())
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    images = []
    for index, url in enumerate(image_urls):
        img_data, mime_type, filename, _ = download_and_save_image(url, index, college_name)
        if img_data and mime_type:
            images.append({
                'mime_type': mime_type,
                'data': img_data,
                'filename': filename,
                'original_index': index
            })
    
    if not images:
        return "No images were successfully downloaded.", []
    
    prompt = """
    Analyze the following images (which may be in SVG or PNG format) and determine which ones are most likely to be college athletics team logos.
    Identify two types of logos:
    1. A logo with text (e.g., college name or team name)
    2. A logo without text (e.g., just the mascot or symbol)

    For each image, provide a score from 0 to 10, where 10 is most likely to be an official team logo.
    Explain your reasoning for each score.

    Return your analysis in the following JSON format:
    {
        "images": [
            {
                "index": 0,
                "filename": "image_0.svg",
                "score": 8,
                "has_text": true/false,
                "reasoning": "This image shows..."
            },
            ...
        ],
        "most_likely_logo_with_text": {"index": 0, "filename": "image_0.svg"},
        "most_likely_logo_without_text": {"index": 1, "filename": "image_1.png"},
        "confidence": "high/medium/low",
        "explanation": "Overall explanation of the choices..."
    }
    """
    
    try:
        image_contents = []
        for img in images:
            image_contents.append({
                'mime_type': img['mime_type'],
                'data': img['data']
            })
            image_contents.append(f"Filename: {img['filename']}, Index: {img['original_index']}")
        
        response = model.generate_content([prompt] + image_contents)
        return response.text, images
    except Exception as e:
        logging.error(f"Detailed Gemini API error: {str(e)}")
        return f"Error in Gemini API call: {str(e)}", images


# def download_and_save_image(url, index, college_name):
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
        
#         os.makedirs(f"downloaded_images/{college_name}", exist_ok=True)
        
#         content_type = response.headers.get('content-type', '').lower()
#         if 'svg' in content_type:
#             svg_filename = f"downloaded_images/{college_name}/image_{index}.svg"
#             with open(svg_filename, 'wb') as f:
#                 f.write(response.content)
#             logging.info(f"Saved SVG: {svg_filename}")
            
#             # Convert SVG to PNG
#             png_image = convert_svg_to_png(url)
#             if png_image:
#                 png_filename = f"downloaded_images/{college_name}/image_{index}.png"
#                 png_image.save(png_filename)
#                 logging.info(f"Converted SVG to PNG: {png_filename}")
#                 with open(png_filename, 'rb') as f:
#                     img_data = f.read()
#                 return base64.b64encode(img_data).decode('utf-8'), 'image/png', png_filename, svg_filename
#             else:
#                 return None, None, None, svg_filename
#         else:
#             img = Image.open(BytesIO(response.content))
#             if img.mode == 'RGBA':
#                 img = img.convert('RGB')
#             filename = f"downloaded_images/{college_name}/image_{index}.png"
#             img.save(filename, 'PNG')
#             logging.info(f"Saved image: {filename}")
#             with open(filename, 'rb') as f:
#                 img_data = f.read()
#             return base64.b64encode(img_data).decode('utf-8'), 'image/png', filename, None
#     except Exception as e:
#         logging.error(f"Error downloading/processing image from {url}: {str(e)}")
#     return None, None, None, None

# def analyze_images_with_gemini(image_urls, college_name):
#     genai.configure(api_key=get_random_gemini_api_key())
#     model = genai.GenerativeModel('gemini-1.5-flash')
    
#     images = []
#     svg_files = []
#     for index, url in enumerate(image_urls):
#         img_data, mime_type, filename, svg_filename = download_and_save_image(url, index, college_name)
#         if img_data and mime_type:
#             images.append({
#                 'mime_type': mime_type,
#                 'data': img_data,
#                 'filename': filename,
#                 'original_index': index,
#                 'svg_filename': svg_filename
#             })
#         elif svg_filename:
#             svg_files.append(svg_filename)
    
#     if not images:
#         return "No images were successfully downloaded.", [], svg_files
    
#     prompt = """
#     Analyze the following images and determine which ones are most likely to be college athletics team logos.
#     Identify two types of logos:
#     1. A logo with text (e.g., college name or team name)
#     2. A logo without text (e.g., just the mascot or symbol)

#     For each image, provide a score from 0 to 10, where 10 is most likely to be an official team logo.
#     Explain your reasoning for each score.

#     Return your analysis in the following JSON format:
#     {
#         "images": [
#             {
#                 "index": 0,
#                 "filename": "image_0.png",
#                 "score": 8,
#                 "has_text": true/false,
#                 "reasoning": "This image shows..."
#             },
#             ...
#         ],
#         "most_likely_logo_with_text": {"index": 0, "filename": "image_0.png"},
#         "most_likely_logo_without_text": {"index": 1, "filename": "image_1.png"},
#         "confidence": "high/medium/low",
#         "explanation": "Overall explanation of the choices..."
#     }
#     """
    
#     try:
#         image_contents = []
#         for img in images:
#             image_contents.append({
#                 'mime_type': img['mime_type'],
#                 'data': img['data']
#             })
#             image_contents.append(f"Filename: {img['filename']}, Index: {img['original_index']}")
        
#         response = model.generate_content([prompt] + image_contents)
#         return response.text, images, svg_files
#     except Exception as e:
#         logging.error(f"Detailed Gemini API error: {str(e)}")
#         return f"Error in Gemini API call: {str(e)}", images, svg_files

def save_logo(src_path, dest_folder, logo_type):
    os.makedirs(dest_folder, exist_ok=True)
    file_name = os.path.basename(src_path)
    dest_path = os.path.join(dest_folder, f"{logo_type}_{file_name}")
    shutil.copy2(src_path, dest_path)
    return dest_path

def scrape_logo(college_name, nickname):
    search_query = f"{college_name} {nickname} logo svg"
    
    logging.info(f"Searching for: {search_query}")
    
    image_urls = search_images(search_query)
    logging.info(f"Found {len(image_urls)} image URLs")
    
    analysis, images, svg_files = analyze_images_with_gemini(image_urls, college_name)
    logging.info("Raw analysis from Gemini:")
    logging.info(analysis)
    logging.info("=" * 50)
    
    try:
        # Remove code block markers if present
        json_str = analysis.strip().lstrip('```json').rstrip('```')
        result = json.loads(json_str)
        
        with_text = result['most_likely_logo_with_text']
        without_text = result['most_likely_logo_without_text']
        
        logo_folder = f"identified_logos/{college_name}"
        
        with_text_image = next((img for img in images if img['original_index'] == with_text['index']), None)
        without_text_image = next((img for img in images if img['original_index'] == without_text['index']), None)
        
        if with_text_image and without_text_image:
            with_text_path = save_logo(with_text_image['filename'], logo_folder, "with_text")
            without_text_path = save_logo(without_text_image['filename'], logo_folder, "without_text")
            
            # Save SVG files if available
            if with_text_image['svg_filename']:
                save_logo(with_text_image['svg_filename'], logo_folder, "with_text_svg")
            if without_text_image['svg_filename']:
                save_logo(without_text_image['svg_filename'], logo_folder, "without_text_svg")
            
            logging.info(f"\nLogo with text:")
            logging.info(f"URL: {image_urls[with_text['index']]}")
            logging.info(f"File name: {with_text['filename']}")
            logging.info(f"Saved at: {with_text_path}")
            
            logging.info(f"\nLogo without text:")
            logging.info(f"URL: {image_urls[without_text['index']]}")
            logging.info(f"File name: {without_text['filename']}")
            logging.info(f"Saved at: {without_text_path}")
            
            return (image_urls[with_text['index']], image_urls[without_text['index']]), analysis
        else:
            logging.error("Could not find the identified logos in the downloaded images.")
            return None, analysis
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}")
        logging.error(f"Raw response: {analysis}")
    except KeyError as e:
        logging.error(f"Key Error: {e}. The Gemini API response might be incomplete or in an unexpected format.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    
    return None, analysis

# Example usage
if __name__ == "__main__":
    college_name = input("Enter the college name: ")
    nickname = input("Enter the team nickname: ")
    logos, analysis = scrape_logo(college_name, nickname)
    if logos:
        logging.info(f"\nAnalysis:\n{analysis}")
    else:
        logging.info("Failed to find logos")