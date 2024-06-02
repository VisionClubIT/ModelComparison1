import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

# Implement a scraping logic to scrape the  website and its subpages
urls = [
    'https://u.ae/en/information-and-services',
    'https://u.ae/en/information-and-services/visa-and-emirates-id',
    'https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas',
    'https://u.ae/en/information-and-services/visa-and-emirates-id/residence-visas/golden-visa'
]

def fetch_page_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page {url}: {e}")
        return None

# Store the scraped content in a dictionary, url:content
scraped_content = {}

for url in urls:
    print(f"Scraping {url}...")
    content = fetch_page_content(url)
    if content:
        scraped_content[url] = content
        time.sleep(1) 

print("Scraping completed.")
print(len(scraped_content))#4 webpages
