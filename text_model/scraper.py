import cloudscraper
from bs4 import BeautifulSoup
import csv
import re
from pathlib import Path
from urllib.parse import urljoin
import requests
import json
import time

scraper = cloudscraper.create_scraper()
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "tanzania_publicinfo_dataset.csv"
LEGACY_DATA_PATH = REPO_ROOT / "data" / "tanzania_political_dataset.csv"
SOURCE_DATA_DIR = REPO_ROOT / "data" / "source_data"
DATA_SCHEMA_FIELDS = [
    "statement",
    "url",
    "text",
    "source",
    "category",
    "label",
    "dataset_name",
    "update_date",
    "source_type",
]


def scrape_world_bank_tanzania():
    """Scrape World Bank Tanzania data"""
    base_url = "https://data.worldbank.org/country/tanzania"
    records = []

    try:
        response = scraper.get(base_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract indicators and data from tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header row
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        indicator = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        year = cells[2].get_text(strip=True) if len(cells) > 2 else ""

                        if indicator and value and any(char.isdigit() for char in value):
                            statement = f"Tanzania {indicator}: {value} ({year})"
                            records.append({
                                "statement": statement,
                                "url": base_url,
                                "text": f"World Bank development indicator: {statement}",
                                "source": "World Bank Tanzania",
                                "category": "development",
                                "label": "verified",
                                "dataset_name": "WorldBank_Tanzania",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "international"
                            })

            # Extract data from indicator cards and sections
            indicator_cards = soup.find_all(['div', 'section'], class_=re.compile(r'indicator|card|data'))
            for card in indicator_cards:
                text = card.get_text(strip=True)
                # Look for patterns with numbers and Tanzania context
                if len(text) > 20 and any(char.isdigit() for char in text):
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if len(sentence) > 15 and any(char.isdigit() for char in sentence):
                            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                            if clean_sentence and 'tanzania' in clean_sentence.lower():
                                records.append({
                                    "statement": clean_sentence,
                                    "url": base_url,
                                    "text": f"World Bank Tanzania data: {clean_sentence}",
                                    "source": "World Bank Tanzania",
                                    "category": "development",
                                    "label": "verified",
                                    "dataset_name": "WorldBank_Content",
                                    "update_date": time.strftime("%Y-%m-%d"),
                                    "source_type": "international"
                                })

            # Extract data from links to specific indicators
            indicator_links = soup.find_all('a', href=re.compile(r'/indicator/'))
            for link in indicator_links[:20]:  # Limit to avoid too many requests
                try:
                    indicator_url = urljoin(base_url, link['href'])
                    indicator_response = scraper.get(indicator_url, timeout=10)
                    if indicator_response.status_code == 200:
                        indicator_soup = BeautifulSoup(indicator_response.text, "html.parser")

                        # Extract latest data values
                        data_values = indicator_soup.find_all(['span', 'div'], class_=re.compile(r'value|data'))
                        indicator_name = indicator_soup.find('h1')
                        indicator_title = indicator_name.get_text(strip=True) if indicator_name else "Indicator"

                        for value_elem in data_values:
                            value_text = value_elem.get_text(strip=True)
                            if any(char.isdigit() for char in value_text) and len(value_text) < 50:
                                statement = f"Tanzania {indicator_title}: {value_text}"
                                records.append({
                                    "statement": statement,
                                    "url": indicator_url,
                                    "text": f"World Bank indicator data: {statement}",
                                    "source": "World Bank Tanzania",
                                    "category": "development",
                                    "label": "verified",
                                    "dataset_name": "WorldBank_Indicators",
                                    "update_date": time.strftime("%Y-%m-%d"),
                                    "source_type": "international"
                                })
                except Exception as e:
                    continue

    except Exception as e:
        print(f"Error scraping World Bank: {e}")

    return records


def scrape_imf_tanzania():
    """Scrape IMF Tanzania data"""
    base_url = "https://www.imf.org/en/Countries/TZA"
    records = []

    try:
        response = scraper.get(base_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract economic data from various sections
            data_sections = soup.find_all(['div', 'section', 'article'], class_=re.compile(r'content|data|economic'))
            for section in data_sections:
                text = section.get_text(strip=True)
                # Split into sentences and look for economic data
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if (len(sentence) > 20 and any(char.isdigit() for char in sentence) and
                        any(keyword in sentence.lower() for keyword in ['gdp', 'growth', 'inflation', 'debt', 'fiscal', 'economic', 'tanzania', 'billion', 'million', 'percent'])):
                        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                        if clean_sentence:
                            records.append({
                                "statement": clean_sentence,
                                "url": base_url,
                                "text": f"IMF economic data: {clean_sentence}",
                                "source": "International Monetary Fund (IMF)",
                                "category": "economic",
                                "label": "verified",
                                "dataset_name": "IMF_Tanzania",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "international"
                            })

            # Extract data from tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        indicator = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        if indicator and value and any(char.isdigit() for char in value):
                            statement = f"Tanzania {indicator}: {value}"
                            records.append({
                                "statement": statement,
                                "url": base_url,
                                "text": f"IMF table data: {statement}",
                                "source": "International Monetary Fund (IMF)",
                                "category": "economic",
                                "label": "verified",
                                "dataset_name": "IMF_Table_Data",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "international"
                            })

            # Try to scrape latest IMF reports and publications
            report_links = soup.find_all('a', href=re.compile(r'/publications|/reports'))
            for link in report_links[:5]:  # Limit to avoid too many requests
                try:
                    report_url = urljoin(base_url, link['href'])
                    if 'imf.org' in report_url:
                        report_response = scraper.get(report_url, timeout=10)
                        if report_response.status_code == 200:
                            report_soup = BeautifulSoup(report_response.text, "html.parser")

                            # Extract key findings and data from reports
                            findings = report_soup.find_all(['p', 'div'], class_=re.compile(r'finding|summary|data'))
                            for finding in findings:
                                text = finding.get_text(strip=True)
                                if len(text) > 30 and any(char.isdigit() for char in text):
                                    sentences = re.split(r'[.!?]+', text)
                                    for sentence in sentences:
                                        if 'tanzania' in sentence.lower() and any(char.isdigit() for char in sentence):
                                            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                                            records.append({
                                                "statement": clean_sentence,
                                                "url": report_url,
                                                "text": f"IMF report data: {clean_sentence}",
                                                "source": "International Monetary Fund (IMF)",
                                                "category": "economic",
                                                "label": "verified",
                                                "dataset_name": "IMF_Reports",
                                                "update_date": time.strftime("%Y-%m-%d"),
                                                "source_type": "international"
                                            })
                except Exception as e:
                    continue

    except Exception as e:
        print(f"Error scraping IMF: {e}")

    return records


def scrape_who_tanzania():
    
    base_url = "https://www.who.int/countries/tza"
    records = []

    try:
        response = scraper.get(base_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract health statistics from various sections
            health_sections = soup.find_all(['div', 'section', 'article'], class_=re.compile(r'content|health|data'))
            for section in health_sections:
                text = section.get_text(strip=True)
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if (len(sentence) > 20 and any(char.isdigit() for char in sentence) and
                        any(keyword in sentence.lower() for keyword in ['health', 'disease', 'mortality', 'vaccination', 'treatment', 'coverage', 'tanzania', 'malaria', 'hiv', 'tuberculosis'])):
                        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                        if clean_sentence:
                            records.append({
                                "statement": clean_sentence,
                                "url": base_url,
                                "text": f"WHO health data: {clean_sentence}",
                                "source": "World Health Organization (WHO)",
                                "category": "health",
                                "label": "verified",
                                "dataset_name": "WHO_Tanzania",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "international"
                            })

            # Extract data from tables and lists
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        indicator = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        if indicator and value and any(char.isdigit() for char in value):
                            statement = f"Tanzania {indicator}: {value}"
                            records.append({
                                "statement": statement,
                                "url": base_url,
                                "text": f"WHO table data: {statement}",
                                "source": "World Health Organization (WHO)",
                                "category": "health",
                                "label": "verified",
                                "dataset_name": "WHO_Table_Data",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "international"
                            })

            # Extract data from health program pages
            program_links = soup.find_all('a', href=re.compile(r'/programmes|/initiatives|/health'))
            for link in program_links[:5]:  # Limit requests
                try:
                    program_url = urljoin(base_url, link['href'])
                    if 'who.int' in program_url:
                        program_response = scraper.get(program_url, timeout=10)
                        if program_response.status_code == 200:
                            program_soup = BeautifulSoup(program_response.text, "html.parser")

                            # Extract health program data
                            program_data = program_soup.find_all(['p', 'div'], class_=re.compile(r'content|data'))
                            for data_elem in program_data:
                                text = data_elem.get_text(strip=True)
                                if len(text) > 25 and any(char.isdigit() for char in text):
                                    sentences = re.split(r'[.!?]+', text)
                                    for sentence in sentences:
                                        if 'tanzania' in sentence.lower() and any(char.isdigit() for char in sentence):
                                            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                                            records.append({
                                                "statement": clean_sentence,
                                                "url": program_url,
                                                "text": f"WHO program data: {clean_sentence}",
                                                "source": "World Health Organization (WHO)",
                                                "category": "health",
                                                "label": "verified",
                                                "dataset_name": "WHO_Programs",
                                                "update_date": time.strftime("%Y-%m-%d"),
                                                "source_type": "international"
                                            })
                except Exception as e:
                    continue

    except Exception as e:
        print(f"Error scraping WHO: {e}")

    return records


def scrape_bot_tanzania():

    base_url = "https://www.bot.go.tz"
    records = []

    try:
        # Try multiple BoT pages
        pages_to_scrape = [
            base_url,
            f"{base_url}/Statistics",
            f"{base_url}/EconomicData"
        ]

        for page_url in pages_to_scrape:
            try:
                response = scraper.get(page_url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Extract financial and monetary data from tables
                    tables = soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows[1:]:  # Skip header
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                indicator = cells[0].get_text(strip=True)
                                value = cells[1].get_text(strip=True)
                                if indicator and value and any(char.isdigit() for char in value):
                                    statement = f"Tanzania {indicator}: {value}"
                                    records.append({
                                        "statement": statement,
                                        "url": page_url,
                                        "text": f"BoT financial data: {statement}",
                                        "source": "Bank of Tanzania (BoT)",
                                        "category": "financial",
                                        "label": "verified",
                                        "dataset_name": "BoT_Table_Data",
                                        "update_date": time.strftime("%Y-%m-%d"),
                                        "source_type": "government"
                                    })

                    # Extract data from content sections
                    content_sections = soup.find_all(['div', 'section'], class_=re.compile(r'content|data|financial'))
                    for section in content_sections:
                        text = section.get_text(strip=True)
                        sentences = re.split(r'[.!?]+', text)
                        for sentence in sentences:
                            if (len(sentence) > 15 and any(char.isdigit() for char in sentence) and
                                any(keyword in sentence.lower() for keyword in ['shilling', 'interest', 'rate', 'bank', 'monetary', 'financial', 'tanzania', 'billion', 'million', 'trillion'])):
                                clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                                if clean_sentence:
                                    records.append({
                                        "statement": clean_sentence,
                                        "url": page_url,
                                        "text": f"BoT content data: {clean_sentence}",
                                        "source": "Bank of Tanzania (BoT)",
                                        "category": "financial",
                                        "label": "verified",
                                        "dataset_name": "BoT_Content_Data",
                                        "update_date": time.strftime("%Y-%m-%d"),
                                        "source_type": "government"
                                    })

                    # Extract data from statistical reports and publications
                    report_links = soup.find_all('a', href=re.compile(r'\.pdf|\.xlsx|report|statistic'))
                    for link in report_links[:3]:  # Limit to avoid too many requests
                        try:
                            report_url = urljoin(page_url, link['href'])
                            if 'bot.go.tz' in report_url:
                                # For PDF/excel files, we can at least get the title and description
                                link_text = link.get_text(strip=True)
                                if link_text and len(link_text) > 10:
                                    records.append({
                                        "statement": f"Bank of Tanzania publishes: {link_text}",
                                        "url": report_url,
                                        "text": f"BoT publication: {link_text}",
                                        "source": "Bank of Tanzania (BoT)",
                                        "category": "financial",
                                        "label": "verified",
                                        "dataset_name": "BoT_Publications",
                                        "update_date": time.strftime("%Y-%m-%d"),
                                        "source_type": "government"
                                    })
                        except Exception as e:
                            continue

            except Exception as e:
                print(f"Error scraping BoT page {page_url}: {e}")
                continue

    except Exception as e:
        print(f"Error scraping Bank of Tanzania: {e}")

    return records


def scrape_tra_tanzania():
    """Scrape Tanzania Revenue Authority data"""
    base_url = "https://www.tra.go.tz"
    records = []

    try:
        response = scraper.get(base_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract revenue and tax data from tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        indicator = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        if indicator and value and any(char.isdigit() for char in value):
                            statement = f"Tanzania {indicator}: {value}"
                            records.append({
                                "statement": statement,
                                "url": base_url,
                                "text": f"TRA revenue data: {statement}",
                                "source": "Tanzania Revenue Authority (TRA)",
                                "category": "revenue",
                                "label": "verified",
                                "dataset_name": "TRA_Table_Data",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "government"
                            })

            # Extract data from content sections
            content_sections = soup.find_all(['div', 'section'], class_=re.compile(r'content|data|revenue'))
            for section in content_sections:
                text = section.get_text(strip=True)
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if (len(sentence) > 15 and any(char.isdigit() for char in sentence) and
                        any(keyword in sentence.lower() for keyword in ['tax', 'revenue', 'collection', 'budget', 'tanzania', 'billion', 'million', 'trillion'])):
                        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                        if clean_sentence:
                            records.append({
                                "statement": clean_sentence,
                                "url": base_url,
                                "text": f"TRA content data: {clean_sentence}",
                                "source": "Tanzania Revenue Authority (TRA)",
                                "category": "revenue",
                                "label": "verified",
                                "dataset_name": "TRA_Content_Data",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "government"
                            })

            # Extract data from news and announcements
            news_links = soup.find_all('a', href=re.compile(r'/news|/announcement|/press'))
            for link in news_links[:5]:  # Limit requests
                try:
                    news_url = urljoin(base_url, link['href'])
                    if 'tra.go.tz' in news_url:
                        news_response = scraper.get(news_url, timeout=10)
                        if news_response.status_code == 200:
                            news_soup = BeautifulSoup(news_response.text, "html.parser")

                            # Extract revenue-related news content
                            news_content = news_soup.find_all(['p', 'div'], class_=re.compile(r'content|text'))
                            for content in news_content:
                                text = content.get_text(strip=True)
                                if len(text) > 25 and any(char.isdigit() for char in text):
                                    sentences = re.split(r'[.!?]+', text)
                                    for sentence in sentences:
                                        if any(keyword in sentence.lower() for keyword in ['tax', 'revenue', 'collection', 'tanzania']):
                                            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                                            records.append({
                                                "statement": clean_sentence,
                                                "url": news_url,
                                                "text": f"TRA news data: {clean_sentence}",
                                                "source": "Tanzania Revenue Authority (TRA)",
                                                "category": "revenue",
                                                "label": "verified",
                                                "dataset_name": "TRA_News",
                                                "update_date": time.strftime("%Y-%m-%d"),
                                                "source_type": "government"
                                            })
                except Exception as e:
                    continue

    except Exception as e:
        print(f"Error scraping TRA: {e}")

    return records


def scrape_unesco_tanzania():
    """Scrape UNESCO Tanzania data"""
    base_url = "https://en.unesco.org/countries/tanzania"
    records = []

    try:
        response = scraper.get(base_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract education and culture data from tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        indicator = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        if indicator and value and any(char.isdigit() for char in value):
                            statement = f"Tanzania {indicator}: {value}"
                            records.append({
                                "statement": statement,
                                "url": base_url,
                                "text": f"UNESCO education data: {statement}",
                                "source": "UNESCO Tanzania",
                                "category": "education_culture",
                                "label": "verified",
                                "dataset_name": "UNESCO_Table_Data",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "international"
                            })

            # Extract data from content sections
            content_sections = soup.find_all(['div', 'section'], class_=re.compile(r'content|data|education|culture'))
            for section in content_sections:
                text = section.get_text(strip=True)
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if (len(sentence) > 15 and any(char.isdigit() for char in sentence) and
                        any(keyword in sentence.lower() for keyword in ['education', 'school', 'university', 'literacy', 'culture', 'heritage', 'tanzania', 'unesco'])):
                        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                        if clean_sentence:
                            records.append({
                                "statement": clean_sentence,
                                "url": base_url,
                                "text": f"UNESCO content data: {clean_sentence}",
                                "source": "UNESCO Tanzania",
                                "category": "education_culture",
                                "label": "verified",
                                "dataset_name": "UNESCO_Content_Data",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "international"
                            })

            # Extract data from project and program pages
            project_links = soup.find_all('a', href=re.compile(r'/projects|/programmes|/initiatives'))
            for link in project_links[:5]:  # Limit requests
                try:
                    project_url = urljoin(base_url, link['href'])
                    if 'unesco.org' in project_url:
                        project_response = scraper.get(project_url, timeout=10)
                        if project_response.status_code == 200:
                            project_soup = BeautifulSoup(project_response.text, "html.parser")

                            # Extract project data
                            project_content = project_soup.find_all(['p', 'div'], class_=re.compile(r'content|description'))
                            for content in project_content:
                                text = content.get_text(strip=True)
                                if len(text) > 25 and any(char.isdigit() for char in text):
                                    sentences = re.split(r'[.!?]+', text)
                                    for sentence in sentences:
                                        if 'tanzania' in sentence.lower() and any(char.isdigit() for char in sentence):
                                            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                                            records.append({
                                                "statement": clean_sentence,
                                                "url": project_url,
                                                "text": f"UNESCO project data: {clean_sentence}",
                                                "source": "UNESCO Tanzania",
                                                "category": "education_culture",
                                                "label": "verified",
                                                "dataset_name": "UNESCO_Projects",
                                                "update_date": time.strftime("%Y-%m-%d"),
                                                "source_type": "international"
                                            })
                except Exception as e:
                    continue

    except Exception as e:
        print(f"Error scraping UNESCO: {e}")

    return records
    full_url = "https://pesacheck.org" + url
    response = scraper.get(full_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_="gh-content")
        if content:
            return content.get_text(strip=True)
    return ""


def get_article_text_citizen(url):
    response = scraper.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_="article-content")
        if content:
            return content.get_text(strip=True)
    return ""


def scrape_nbs_data():
    """Scrape comprehensive National Bureau of Statistics Tanzania data"""
    base_url = "https://www.nbs.go.tz"
    records = []

    try:
        # Updated URLs based on current NBS website structure
        pages_to_scrape = [
            f"{base_url}/statistics/economic-statistics",  # GDP and economic data
            f"{base_url}/statistics/labour-statistics",    # CPI and labor data
            f"{base_url}/statistics/topic/producer-price-indices-ppi",  # PPI data
            f"{base_url}/statistics/topic/export-and-import-price-indices-xmpi",  # Trade indices
            f"{base_url}/statistics/topic/index-of-industrial-production-iip",  # Industrial production
            f"{base_url}/statistics/topic/utafiti-wa-kufuatilia-kaya-tanzania-nps-wa-mwaka-wa-6",  # National Panel Survey
            f"{base_url}/statistics",  # Main statistics page
        ]

        for page_url in pages_to_scrape:
            try:
                response = scraper.get(page_url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Extract statistical data from tables
                    tables = soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                indicator = cells[0].get_text(strip=True)
                                value = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                                if indicator and value and any(char.isdigit() for char in value):
                                    statement = f"Tanzania {indicator}: {value}"
                                    records.append({
                                        "statement": statement,
                                        "url": page_url,
                                        "text": f"Statistical data from NBS: {statement}",
                                        "source": "National Bureau of Statistics (NBS)",
                                        "category": "official_statistics",
                                        "label": "verified",
                                        "dataset_name": "NBS_Tables",
                                        "update_date": time.strftime("%Y-%m-%d"),
                                        "source_type": "government"
                                    })

                    # Extract data from divs and spans containing statistics
                    stat_elements = soup.find_all(['div', 'span', 'p'], class_=re.compile(r'(stat|data|value|number)', re.I))
                    for elem in stat_elements:
                        text = elem.get_text(strip=True)
                        if len(text) > 5 and any(char.isdigit() for char in text):
                            # Look for parent elements that might contain indicator names
                            parent = elem.find_parent()
                            if parent:
                                parent_text = parent.get_text(strip=True)
                                if len(parent_text) > len(text) and parent_text != text:
                                    statement = parent_text
                                else:
                                    statement = text
                            else:
                                statement = text

                            records.append({
                                "statement": statement,
                                "url": page_url,
                                "text": f"NBS statistics: {statement}",
                                "source": "National Bureau of Statistics (NBS)",
                                "category": "official_statistics",
                                "label": "verified",
                                "dataset_name": "NBS_Stats",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "government"
                            })

                    # Extract data from paragraphs containing numbers
                    paragraphs = soup.find_all('p')
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        # Look for sentences with numbers
                        sentences = re.split(r'[.!?]+', text)
                        for sentence in sentences:
                            if any(char.isdigit() for char in sentence) and len(sentence) > 20:
                                # Clean up the sentence
                                clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                                if clean_sentence:
                                    records.append({
                                        "statement": clean_sentence,
                                        "url": page_url,
                                        "text": f"NBS information: {clean_sentence}",
                                        "source": "National Bureau of Statistics (NBS)",
                                        "category": "official_statistics",
                                        "label": "verified",
                                        "dataset_name": "NBS_Content",
                                        "update_date": time.strftime("%Y-%m-%d"),
                                        "source_type": "government"
                                    })

                    # Look for data links and datasets
                    data_links = soup.find_all('a', href=re.compile(r'\.(csv|xlsx|pdf|xls)$', re.I))
                    for link in data_links[:20]:  # Increased limit
                        href = link.get('href')
                        if not href.startswith('http'):
                            href = urljoin(base_url, href)

                        title = link.get_text(strip=True) or "NBS Dataset"
                        if title and len(title) > 3:
                            records.append({
                                "statement": f"NBS Dataset: {title}",
                                "url": href,
                                "text": f"Statistical dataset from National Bureau of Statistics Tanzania: {title}",
                                "source": "National Bureau of Statistics (NBS)",
                                "category": "official_statistics",
                                "label": "verified",
                                "dataset_name": "NBS_Datasets",
                                "update_date": time.strftime("%Y-%m-%d"),
                                "source_type": "government"
                            })

            except Exception as e:
                print(f"Error scraping NBS page {page_url}: {e}")
                continue

    except Exception as e:
        print(f"Error scraping NBS: {e}")

    return records


def scrape_ministry_health():
    """Scrape comprehensive Ministry of Health data"""
    base_url = "https://hmisportal.moh.go.tz"
    records = []

    try:
        # Try multiple health ministry pages
        pages_to_scrape = [
            base_url,
            f"{base_url}/dashboard",
            f"{base_url}/reports"
        ]

        for page_url in pages_to_scrape:
            try:
                response = scraper.get(page_url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Extract health statistics from various elements
                    elements_with_data = soup.find_all(['div', 'p', 'span', 'td', 'li'])
                    for elem in elements_with_data:
                        text = elem.get_text(strip=True)
                        if len(text) > 10 and any(char.isdigit() for char in text):
                            # Look for health-related keywords
                            health_keywords = ['health', 'disease', 'hospital', 'clinic', 'vaccination',
                                             'malaria', 'HIV', 'TB', 'cancer', 'diabetes', 'mortality',
                                             'birth', 'death', 'patient', 'doctor', 'nurse', 'medicine']

                            if any(keyword.lower() in text.lower() for keyword in health_keywords):
                                records.append({
                                    "statement": f"Health Statistics: {text}",
                                    "url": page_url,
                                    "text": f"Health data from Ministry of Health Tanzania: {text}",
                                    "source": "Ministry of Health Tanzania",
                                    "category": "health",
                                    "label": "verified",
                                    "dataset_name": "MoH_HMIS",
                                    "update_date": time.strftime("%Y-%m-%d"),
                                    "source_type": "government"
                                })

                    # Extract data from tables
                    tables = soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                indicator = cells[0].get_text(strip=True)
                                value = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                                if indicator and value and any(char.isdigit() for char in value):
                                    statement = f"Tanzania Health - {indicator}: {value}"
                                    records.append({
                                        "statement": statement,
                                        "url": page_url,
                                        "text": f"Health statistics from MoH: {statement}",
                                        "source": "Ministry of Health Tanzania",
                                        "category": "health",
                                        "label": "verified",
                                        "dataset_name": "MoH_Tables",
                                        "update_date": time.strftime("%Y-%m-%d"),
                                        "source_type": "government"
                                    })

            except Exception as e:
                print(f"Error scraping MoH page {page_url}: {e}")
                continue

    except Exception as e:
        print(f"Error scraping Ministry of Health: {e}")

    return records


def scrape_ministry_education():
    """Scrape comprehensive Ministry of Education data"""
    base_url = "https://esmis.moe.go.tz"
    records = []

    try:
        # Try multiple education ministry pages
        pages_to_scrape = [
            base_url,
            f"{base_url}/dashboard",
            f"{base_url}/statistics"
        ]

        for page_url in pages_to_scrape:
            try:
                response = scraper.get(page_url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Extract education statistics from various elements
                    elements_with_data = soup.find_all(['div', 'p', 'span', 'td', 'li'])
                    for elem in elements_with_data:
                        text = elem.get_text(strip=True)
                        if len(text) > 10 and any(char.isdigit() for char in text):
                            # Look for education-related keywords
                            education_keywords = ['education', 'school', 'student', 'teacher', 'enrollment',
                                                'literacy', 'primary', 'secondary', 'university', 'college',
                                                'exam', 'certificate', 'diploma', 'degree', 'curriculum']

                            if any(keyword.lower() in text.lower() for keyword in education_keywords):
                                records.append({
                                    "statement": f"Education Statistics: {text}",
                                    "url": page_url,
                                    "text": f"Education data from Ministry of Education Tanzania: {text}",
                                    "source": "Ministry of Education Tanzania",
                                    "category": "education",
                                    "label": "verified",
                                    "dataset_name": "MoE_ESMIS",
                                    "update_date": time.strftime("%Y-%m-%d"),
                                    "source_type": "government"
                                })

                    # Extract data from tables
                    tables = soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                indicator = cells[0].get_text(strip=True)
                                value = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                                if indicator and value and any(char.isdigit() for char in value):
                                    statement = f"Tanzania Education - {indicator}: {value}"
                                    records.append({
                                        "statement": statement,
                                        "url": page_url,
                                        "text": f"Education statistics from MoE: {statement}",
                                        "source": "Ministry of Education Tanzania",
                                        "category": "education",
                                        "label": "verified",
                                        "dataset_name": "MoE_Tables",
                                        "update_date": time.strftime("%Y-%m-%d"),
                                        "source_type": "government"
                                    })

            except Exception as e:
                print(f"Error scraping MoE page {page_url}: {e}")
                continue

    except Exception as e:
        print(f"Error scraping Ministry of Education: {e}")

    return records


def scrape_hurumap():
    """Scrape comprehensive HURUmap Tanzania data"""
    base_url = "https://tanzania.hurumap.org"
    records = []

    try:
        # Try multiple HURUmap pages
        pages_to_scrape = [
            base_url,
            f"{base_url}/about",
            f"{base_url}/data"
        ]

        for page_url in pages_to_scrape:
            try:
                response = scraper.get(page_url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Extract demographic data from various elements
                    elements_with_data = soup.find_all(['div', 'p', 'span', 'td', 'li'])
                    for elem in elements_with_data:
                        text = elem.get_text(strip=True)
                        if len(text) > 10 and any(char.isdigit() for char in text):
                            # Look for demographic-related keywords
                            demographic_keywords = ['population', 'census', 'demographic', 'ethnic', 'language',
                                                  'urban', 'rural', 'age', 'gender', 'household', 'migration',
                                                  'fertility', 'mortality', 'birth', 'death', 'poverty']

                            if any(keyword.lower() in text.lower() for keyword in demographic_keywords):
                                records.append({
                                    "statement": f"Demographic Data: {text}",
                                    "url": page_url,
                                    "text": f"Demographic data from HURUmap Tanzania: {text}",
                                    "source": "HURUmap Tanzania",
                                    "category": "demographics",
                                    "label": "verified",
                                    "dataset_name": "HURUmap_Data",
                                    "update_date": time.strftime("%Y-%m-%d"),
                                    "source_type": "research"
                                })

                    # Extract data from tables and charts
                    tables = soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                indicator = cells[0].get_text(strip=True)
                                value = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                                if indicator and value and any(char.isdigit() for char in value):
                                    statement = f"Tanzania Demographics - {indicator}: {value}"
                                    records.append({
                                        "statement": statement,
                                        "url": page_url,
                                        "text": f"Census data from HURUmap: {statement}",
                                        "source": "HURUmap Tanzania",
                                        "category": "demographics",
                                        "label": "verified",
                                        "dataset_name": "HURUmap_Tables",
                                        "update_date": time.strftime("%Y-%m-%d"),
                                        "source_type": "research"
                                    })

            except Exception as e:
                print(f"Error scraping HURUmap page {page_url}: {e}")
                continue

    except Exception as e:
        print(f"Error scraping HURUmap: {e}")

    return records


def get_article_text_citizen(url):
    response = scraper.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_="article-content")
        if content:
            return content.get_text(strip=True)
    return ""


def scrape_pesacheck(pages=5):
    base_url = "https://pesacheck.org/tag/tanzania/page/{}/"
    records = []
    for i in range(1, pages + 1):
        url = base_url.format(i)
        response = scraper.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.find_all("article")
            print(f"✅ Page {i} done — {len(articles)} records found")
            for art in articles:
                img = art.find("img")
                link_elem = art.find("a")
                if img and link_elem:
                    title = img.get("alt", "")
                    link = link_elem.get("href", "")
                    if title and link:
                        text = get_article_text_pesacheck(link)
                        records.append({
                            "statement": title,
                            "url": "https://pesacheck.org" + link,
                            "text": text,
                            "source": "PesaCheck",
                            "category": "social",
                            "label": "verified"
                        })
        else:
            print(f"❌ Page {i} failed — status {response.status_code}")
    return records


def scrape_thecitizen(pages=1):
    base_url = "https://www.thecitizen.co.tz/tanzania/news"
    records = []
    seen = set()
    matcher = re.compile(r"^/tanzania/news/(national|international|africa|entertainment)/.+-\d+$")

    for i in range(1, pages + 1):
        url = base_url if i == 1 else f"{base_url}?page={i}"
        response = scraper.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            anchors = soup.find_all("a", href=matcher)
            print(f"✅ The Citizen page {i} done — {len(anchors)} articles found")
            for anchor in anchors:
                href = anchor.get("href", "")
                full_url = urljoin("https://www.thecitizen.co.tz", href)
                if full_url in seen:
                    continue
                seen.add(full_url)
                raw_title = anchor.get_text(" ", strip=True)
                title = re.sub(r"^(National|International|Africa|Entertainment)", "", raw_title, flags=re.I).strip()
                title = re.sub(r"(?:\d{1,2}\s+hours?\s+ago|Yesterday|[A-Za-z]{3}\s+\d{1,2}(?:,\s*\d{4})?)\s*-\s*\d+\s*min\s*read$", "", title).strip()
                text = get_article_text_citizen(full_url)
                records.append({
                    "statement": title,
                    "url": full_url,
                    "text": text,
                    "source": "The Citizen",
                    "category": "public",
                    "label": "news"
                })
        else:
            print(f"❌ The Citizen page {i} failed — status {response.status_code}")
    return records


def save_records(records, path=DATA_PATH):
    if not records:
        print("No records to save.")
        return

    # Use all available fields from the records
    if records:
        fieldnames = list(records[0].keys())
    else:
        fieldnames = ["statement", "url", "text", "source", "category", "label"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Dataset saved to {path} with {len(records)} records.")


def normalize_source_record(row, source_name="unknown"):
    return {
        "statement": row.get("statement") or row.get("text") or row.get("title") or "",
        "url": row.get("url", ""),
        "text": row.get("text", ""),
        "source": row.get("source", source_name),
        "category": row.get("category", "social"),
        "label": row.get("label", "unverified"),
        "dataset_name": row.get("dataset_name", ""),
        "update_date": row.get("update_date", ""),
        "source_type": row.get("source_type", ""),
    }


def load_source_csv(csv_path):
    records = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = normalize_source_record(row, source_name=csv_path.stem)
            if record["statement"]:
                records.append(record)
    return records


def load_source_directory(source_dir=SOURCE_DATA_DIR):
    records = []
    if not source_dir.exists() or not source_dir.is_dir():
        return records
    for csv_path in sorted(source_dir.glob("*.csv")):
        records.extend(load_source_csv(csv_path))
    return records


def create_dataset_from_sources(source_dir=SOURCE_DATA_DIR, output_path=DATA_PATH):
    source_records = load_source_directory(source_dir)
    if source_records:
        save_records(source_records, output_path)
    return source_records


def load_local_authoritative_dataset(path=DATA_PATH):
    if not path.exists() and LEGACY_DATA_PATH.exists():
        path = LEGACY_DATA_PATH

    records = []
    if not path.exists():
        return load_source_directory(SOURCE_DATA_DIR)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "statement": row.get("statement") or row.get("title") or "",
                "url": row.get("url", ""),
                "text": row.get("text", ""),
                "source": row.get("source", ""),
                "category": row.get("category", "social"),
                "label": row.get("label", "verified")
            })
    return records


def scrape_mwananchi():
    """Scrape Mwananchi newspaper for Tanzanian news and information"""
    base_url = "https://www.mwananchi.co.tz"
    records = []

    try:
        response = scraper.get(base_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract news articles from list pages
            articles = soup.find_all(['article', 'div'], class_=re.compile(r'post|article|news|item'))
            article_links = soup.find_all('a', href=re.compile(r'/20\d{2}/|/article/'))

            for link in article_links[:15]:  # Limit to recent articles
                try:
                    article_url = urljoin(base_url, link['href'])
                    if 'mwananchi' in article_url:
                        article_response = scraper.get(article_url, timeout=10)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.text, "html.parser")

                            # Extract article title and content
                            title_elem = article_soup.find(['h1', 'h2'])
                            content_elem = article_soup.find(['div', 'article'], class_=re.compile(r'content|entry|body'))

                            if title_elem and content_elem:
                                title = title_elem.get_text(strip=True)
                                content = content_elem.get_text(strip=True)

                                # Extract sentences with numbers and Tanzania context
                                sentences = re.split(r'[.!?]+', content)
                                for sentence in sentences:
                                    if (len(sentence) > 30 and any(char.isdigit() for char in sentence) and
                                        any(keyword in sentence.lower() for keyword in ['tanzania', 'dar', 'moshi', 'kilimanjaro', 'arusha', 'economy', 'health', 'education', 'population', 'government'])):
                                        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                                        if clean_sentence and len(clean_sentence) < 300:
                                            records.append({
                                                "statement": clean_sentence,
                                                "url": article_url,
                                                "text": f"News from Mwananchi: {clean_sentence}",
                                                "source": "Mwananchi Newspaper",
                                                "category": "news",
                                                "label": "verified",
                                                "dataset_name": "Mwananchi_News",
                                                "update_date": time.strftime("%Y-%m-%d"),
                                                "source_type": "news"
                                            })
                except Exception as e:
                    continue

    except Exception as e:
        print(f"Error scraping Mwananchi: {e}")

    return records


def scrape_daily_news():
    """Scrape Daily News Tanzania for news and information"""
    base_url = "https://www.dailynews.co.tz"
    records = []

    try:
        response = scraper.get(base_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract news articles
            article_links = soup.find_all('a', href=re.compile(r'/20\d{2}/|/news/|/article/'))

            for link in article_links[:15]:  # Limit to recent articles
                try:
                    article_url = urljoin(base_url, link['href'])
                    if 'dailynews' in article_url:
                        article_response = scraper.get(article_url, timeout=10)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.text, "html.parser")

                            # Extract article content
                            title_elem = article_soup.find(['h1', 'h2'])
                            content_elem = article_soup.find(['div', 'article', 'main'], class_=re.compile(r'content|entry|body|article'))

                            if title_elem and content_elem:
                                title = title_elem.get_text(strip=True)
                                content = content_elem.get_text(strip=True)

                                # Extract sentences with information
                                sentences = re.split(r'[.!?]+', content)
                                for sentence in sentences:
                                    if (len(sentence) > 30 and any(char.isdigit() for char in sentence) and
                                        any(keyword in sentence.lower() for keyword in ['tanzania', 'dar', 'country', 'economy', 'business', 'health', 'development', 'people', 'government'])):
                                        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                                        if clean_sentence and len(clean_sentence) < 300:
                                            records.append({
                                                "statement": clean_sentence,
                                                "url": article_url,
                                                "text": f"News from Daily News: {clean_sentence}",
                                                "source": "Daily News Tanzania",
                                                "category": "news",
                                                "label": "verified",
                                                "dataset_name": "DailyNews_Articles",
                                                "update_date": time.strftime("%Y-%m-%d"),
                                                "source_type": "news"
                                            })
                except Exception as e:
                    continue

    except Exception as e:
        print(f"Error scraping Daily News: {e}")

    return records


def scrape_citizen():
    """Scrape The Citizen newspaper for Tanzanian news and information"""
    base_url = "https://www.thecitizen.co.tz"
    records = []

    try:
        response = scraper.get(base_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract news articles
            article_links = soup.find_all('a', href=re.compile(r'/20\d{2}/|/citizen-news/|/article/'))

            for link in article_links[:15]:  # Limit to recent articles
                try:
                    article_url = urljoin(base_url, link['href'])
                    if 'citizen' in article_url:
                        article_response = scraper.get(article_url, timeout=10)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.text, "html.parser")

                            # Extract article content
                            title_elem = article_soup.find(['h1', 'h2'])
                            content_elem = article_soup.find(['div', 'article', 'main'], class_=re.compile(r'content|entry|body|post-content'))

                            if title_elem and content_elem:
                                title = title_elem.get_text(strip=True)
                                content = content_elem.get_text(strip=True)

                                # Extract sentences with information
                                sentences = re.split(r'[.!?]+', content)
                                for sentence in sentences:
                                    if (len(sentence) > 30 and any(char.isdigit() for char in sentence) and
                                        any(keyword in sentence.lower() for keyword in ['tanzania', 'dar es salaam', 'economy', 'business', 'society', 'politics', 'development', 'nation', 'people'])):
                                        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                                        if clean_sentence and len(clean_sentence) < 300:
                                            records.append({
                                                "statement": clean_sentence,
                                                "url": article_url,
                                                "text": f"News from The Citizen: {clean_sentence}",
                                                "source": "The Citizen Newspaper",
                                                "category": "news",
                                                "label": "verified",
                                                "dataset_name": "Citizen_Articles",
                                                "update_date": time.strftime("%Y-%m-%d"),
                                                "source_type": "news"
                                            })
                except Exception as e:
                    continue

    except Exception as e:
        print(f"Error scraping The Citizen: {e}")

    return records


def build_dataset():
    print("🔍 Collecting comprehensive data from Tanzanian and international sources...")

    # Official Tanzanian government sources
    nbs_data = scrape_nbs_data()
    print(f"✅ NBS: {len(nbs_data)} records")

    health_data = scrape_ministry_health()
    print(f"✅ Ministry of Health: {len(health_data)} records")

    education_data = scrape_ministry_education()
    print(f"✅ Ministry of Education: {len(education_data)} records")

    hurumap_data = scrape_hurumap()
    print(f"✅ HURUmap: {len(hurumap_data)} records")

    # Additional Tanzanian government sources
    bot_data = scrape_bot_tanzania()
    print(f"✅ Bank of Tanzania: {len(bot_data)} records")

    tra_data = scrape_tra_tanzania()
    print(f"✅ Tanzania Revenue Authority: {len(tra_data)} records")

    # International sources
    world_bank_data = scrape_world_bank_tanzania()
    print(f"✅ World Bank: {len(world_bank_data)} records")

    imf_data = scrape_imf_tanzania()
    print(f"✅ IMF: {len(imf_data)} records")

    who_data = scrape_who_tanzania()
    print(f"✅ WHO: {len(who_data)} records")

    unesco_data = scrape_unesco_tanzania()
    print(f"✅ UNESCO: {len(unesco_data)} records")

    # News sources
    mwananchi_data = scrape_mwananchi()
    print(f"✅ Mwananchi: {len(mwananchi_data)} records")

    daily_news_data = scrape_daily_news()
    print(f"✅ Daily News: {len(daily_news_data)} records")

    citizen_data = scrape_citizen()
    print(f"✅ The Citizen: {len(citizen_data)} records")

    # Local existing data
    local = load_local_authoritative_dataset()
    print(f"✅ Local dataset: {len(local)} records")

    # Combine all sources
    all_records = (nbs_data + health_data + education_data + hurumap_data +
                   bot_data + tra_data + world_bank_data + imf_data +
                   who_data + unesco_data + mwananchi_data + daily_news_data +
                   citizen_data + local)

    # Normalize all records to have consistent fields
    normalized_records = []
    for record in all_records:
        normalized_record = {
            "statement": record.get("statement", ""),
            "url": record.get("url", ""),
            "text": record.get("text", ""),
            "source": record.get("source", ""),
            "category": record.get("category", ""),
            "label": record.get("label", ""),
            "dataset_name": record.get("dataset_name", ""),
            "update_date": record.get("update_date", ""),
            "source_type": record.get("source_type", "")
        }
        normalized_records.append(normalized_record)

    # Deduplicate based on statement
    unique_statements = {rec["statement"]: rec for rec in normalized_records}
    final_records = list(unique_statements.values())

    print(f"📊 Total unique records: {len(final_records)}")
    print("🎯 Sources integrated: NBS, MoH, MoE, HURUmap, BoT, TRA, World Bank, IMF, WHO, UNESCO, Mwananchi, Daily News, The Citizen")
    return final_records


if __name__ == "__main__":
    print("🧹 Building social/public information dataset...")
    all_records = build_dataset()
    save_records(all_records)
