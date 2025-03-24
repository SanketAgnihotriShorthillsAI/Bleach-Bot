import os
import json
import time
import logging
import random
from datetime import datetime, timezone
from typing import Optional, Dict
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BleachWikiScraper:
    """
    A web scraper for extracting structured data from the Bleach Fandom Wiki.

    This scraper uses Selenium to retrieve HTML content and BeautifulSoup to parse the data.
    
    Attributes:
        BASE_URL (str): The base URL of the Bleach Fandom Wiki.
        topics (List[str]): The list of topics to scrape.
        max_retries (int): The number of retry attempts for failed requests.
    """

    BASE_URL = "https://bleach.fandom.com/wiki/"

    def __init__(self, topics: Optional[list] = None, max_retries: int = 3) -> None:
        """
        Initializes the scraper with a list of topics and retry attempts.

        Args:
            topics (Optional[list]): List of topic names to scrape. Defaults to a single topic.
            max_retries (int): Maximum number of retries for failed page fetches.
        """
        if topics is None:
            topics = ["Ichigo_Kurosaki"]
        self.topics = [topic.replace(" ", "_") for topic in topics]  # Convert spaces to underscores
        self.max_retries = max_retries

    def setup_selenium(self) -> webdriver.Chrome:
        """
        Sets up the Selenium WebDriver with a rotating User-Agent.

        Returns:
            webdriver.Chrome: A configured Chrome WebDriver instance.
        """
        options = Options()
        ua = UserAgent()
        options.add_argument(f"user-agent={ua.random}")
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        options.add_argument("--disable-blink-features=AutomationControlled")  # Bypass bot detection
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(180)  # Increased timeout to handle slow page loads
        return driver

    def fetch_page_with_selenium(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetches the page source using Selenium and returns a BeautifulSoup object.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[BeautifulSoup]: Parsed HTML content of the page, or None if fetching fails.
        """
        attempt = 0
        while attempt < self.max_retries:
            try:
                driver = self.setup_selenium()
                logging.info(f"Fetching: {url} using Selenium (Attempt {attempt + 1})...")
                driver.get(url)

                sleep_time = random.uniform(5, 10)  # Random delay to mimic human behavior
                time.sleep(sleep_time)

                page_source = driver.page_source
                driver.quit()

                logging.info("Successfully fetched page with Selenium.")
                return BeautifulSoup(page_source, "html.parser")

            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                attempt += 1
                time.sleep(5)  # Wait before retrying

        logging.error("Failed to fetch page after multiple attempts.")
        return None

    def parse_sections(self, soup: Optional[BeautifulSoup], topic: str) -> Optional[Dict]:
        """
        Extracts structured sections from the page.

        Args:
            soup (Optional[BeautifulSoup]): Parsed HTML content of the page.
            topic (str): The topic name.

        Returns:
            Optional[Dict]: A structured dictionary of extracted data, or None if no content found.
        """
        if soup is None:
            logging.warning(f"No content fetched for {topic}! Soup is None.")
            return None

        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            logging.warning(f"No content found for {topic}")
            return None

        data = {
            "source": "Bleach Fandom Wiki",
            "url": f"{self.BASE_URL}{topic}",
            "title": topic.replace("_", " "),
            "sections": [],
            "last_updated": str(datetime.now(timezone.utc)),
        }

        current_section = None
        for element in content.find_all(["h2", "h3", "p", "ul", "ol"]):
            if element.name == "h2":
                if current_section:
                    data["sections"].append(current_section)  # Store previous section
                current_section = {"heading": element.text.strip(), "text": "", "subsections": []}
            elif element.name == "h3":
                if current_section:
                    current_section["subsections"].append({"subheading": element.text.strip(), "text": ""})
            elif element.name in ["p", "ul", "ol"]:
                if current_section:
                    if current_section["subsections"]:
                        current_section["subsections"][-1]["text"] += element.get_text(separator=" ").strip() + " "
                    else:
                        current_section["text"] += element.get_text(separator=" ").strip() + " "

        if current_section:  # Add last parsed section
            data["sections"].append(current_section)

        return data

    def save_to_json(self, data: Optional[Dict], folder: str = "bleach_wiki/raw") -> None:
        """
        Saves the extracted data to a JSON file.

        Args:
            data (Optional[Dict]): The structured data to save.
            folder (str): Directory where the JSON file will be stored.
        """
        if not data:
            return

        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{data['title'].replace(' ', '_')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Data saved to {filename} (Sections: {len(data['sections'])})")

    def run(self) -> None:
        """
        Runs the complete scraping pipeline for multiple topics.
        """
        for topic in self.topics:
            url = f"{self.BASE_URL}{topic}"
            logging.info(f"Scraping: {url}")
            soup = self.fetch_page_with_selenium(url)
            data = self.parse_sections(soup, topic)
            self.save_to_json(data)


# Run the scraper for multiple pages
if __name__ == "__main__":
    topics = ["Nobutsuna Shigyō"]
    scraper = BleachWikiScraper(topics=topics)
    scraper.run()



