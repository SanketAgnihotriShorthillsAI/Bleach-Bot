import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_URL = "https://bleach.fandom.com"
CHARACTER_PAGE = "https://bleach.fandom.com/wiki/Category:Characters"

class BleachCharacterExtractor:
    def __init__(self):
        """Initializes the scraper for Bleach Wiki character pages."""
        self.character_names = set()

    def setup_selenium(self):
        """Sets up Selenium WebDriver with a rotating User-Agent."""
        options = Options()
        ua = UserAgent()
        options.add_argument(f"user-agent={ua.random}")
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-dns-prefetch")
        options.add_argument("--disable-gpu")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(180)
        return driver

    def extract_character_names(self):
        """Scrapes the Bleach Wiki 'Characters' category page to get all character names."""
        driver = self.setup_selenium()
        wait = WebDriverWait(driver, 30)

        driver.get(CHARACTER_PAGE)

        while True:
            logging.info("Scraping character names from page...")

            # Wait until at least one character link is visible
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".category-page__members a")))
            except Exception as e:
                logging.error(f"Timeout while waiting for characters to load: {e}")
                break

            # Parse the page source with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")
            character_links = soup.select(".category-page__members a")

            if not character_links:
                logging.warning("No character names found. The page structure may have changed.")
                break

            for link in character_links:
                name = link.get_text(strip=True)
                if not name.startswith("Category:"):
                    self.character_names.add(name)

            # Try finding the "Next Page" button
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, "a.category-page__pagination-next")
                next_page_url = next_button.get_attribute("href")
                if next_page_url:
                    logging.info(f"Navigating to next page: {next_page_url}")
                    driver.get(next_page_url)
                    time.sleep(3)  # Allow time for page load
                else:
                    logging.info("Next page URL is empty. Ending scrape.")
                    break
            except Exception as e:
                logging.info("No more pages to scrape or next button not found.")
                break

        driver.quit()
        return sorted(self.character_names)

    def save_character_names(self, filename="bleach_characters.txt"):
        """Saves extracted character names to a text file."""
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)

        with open(filepath, "w", encoding="utf-8") as file:
            for name in sorted(self.character_names):
                file.write(name + "\n")

        logging.info(f"Saved {len(self.character_names)} character names to {filepath}")

    def run(self):
        """Runs the full process of extracting and saving character names."""
        logging.info("Starting Bleach Wiki Character Extraction...")
        names = self.extract_character_names()
        self.save_character_names()
        logging.info("Character extraction complete!")
        return names

# Run the extractor
if __name__ == "__main__":
    extractor = BleachCharacterExtractor()
    character_list = extractor.run()
    print("Extracted character names:")
    for name in character_list:
        print(name)
