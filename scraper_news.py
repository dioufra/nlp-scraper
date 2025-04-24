# ┌────────────────────────────────────┐
# │           Imports & Setup          │
# └────────────────────────────────────┘

import json
import time
import random
import requests
import os

from uuid import uuid4
from typing import List, Dict, Optional
from urllib.parse import quote, urljoin
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, String, Column, Date, Text


# Constants
DAYS_LIMIT = 90
MAX_WORKERS = min(20, (os.cpu_count() or 1) * 5)
MIN_ARTICLES = 10
QUANTITY = 50

BASE_DOMAIN = "https://www.aljazeera.com"
DATABASE_URL = "sqlite:///data/articles.db"
OUTPUT_PATH = "data/articles.json"

# ┌────────────────────────────────────┐
# │       SQLAlchemy Setup             │
# └────────────────────────────────────┘

engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)


class Article(Base):
    __tablename__ = "article"
    uuid = Column(String, primary_key=True, index=True)
    url = Column(String, unique=True, nullable=False)
    date = Column(Date)
    headline = Column(Text)
    body = Column(Text)


Base.metadata.create_all(engine)

# ┌────────────────────────────────────┐
# │       Web Scraper Functions        │
# └────────────────────────────────────┘


def get_links(
    target_count: int = 500, min_delay: float = 1, max_delay: float = 3
) -> List[str]:
    """Scrape Al Jazeera article URLs via GraphQL endpoint."""

    graphql_url = "https://www.aljazeera.com/graphql?wp-site=aje&operationName=ArchipelagoAjeSectionPostsQuery"

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "original-domain": "www.aljazeera.com",
        "user-agent": "Mozilla/5.0",
        "wp-site": "aje",
    }

    urls = set()
    offset = 0

    while len(urls) < target_count:
        try:
            time.sleep(random.uniform(min_delay, max_delay))

            variables = {
                "category": "news",
                "categoryType": "categories",
                "postTypes": [
                    "blog",
                    "episode",
                    "opinion",
                    "post",
                    "video",
                    "external-article",
                    "gallery",
                    "podcast",
                    "longform",
                    "liveblog",
                ],
                "quantity": QUANTITY,
                "offset": offset,
            }

            encoded_vars = quote(json.dumps(variables))
            url = f"{graphql_url}&variables={encoded_vars}&extensions=%7B%7D"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            articles = response.json().get("data", {}).get("articles", [])
            if not articles:
                print("No more articles found.")
                break

            for article in articles:
                link = article.get("link")
                if link:
                    full_url = urljoin(BASE_DOMAIN, link.lstrip("/"))
                    full_url = full_url.replace("http://", "https://", 1)
                    urls.add(full_url)
                    print(f"Collected {len(urls)}/{target_count}: {full_url}")

            offset += QUANTITY

        except Exception as e:
            print(f"Request error: {e}, retrying...")
            time.sleep(max_delay * 2)

    return list(urls)[:target_count]


def parse_article(url: str) -> Optional[Dict]:
    """Scrape article content from the given URL."""
    try:
        print(f"Scraping {url}")
        time.sleep(0.5 + random.random())

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        date_tag = soup.select_one(
            '.article-dates .date-simple span[aria-hidden="true"]'
        )
        if not date_tag:
            return None

        headline = soup.find("h1").get_text(strip=True)
        paragraphs = soup.find_all("p")
        body = " ".join(p.get_text(strip=True) for p in paragraphs)

        return {
            "uuid": str(uuid4()),
            "url": url,
            "date": datetime.strptime(date_tag.text.strip(), "%d %b %Y").date(),
            "headline": headline,
            "body": body,
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


# ┌────────────────────────────────────┐
# │         Scraping Pipeline          │
# └────────────────────────────────────┘


def scrape_articles_threaded(urls: List[str]) -> List[Dict]:
    """Scrape multiple articles concurrently."""

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(parse_article, urls))
    return [r for r in results if r is not None]


def insert_articles(articles: List[Dict]):
    """Insert new articles into the database."""
    session = Session()
    try:
        session.bulk_insert_mappings(Article, articles)
        session.commit()
        print("saved in ./data/articles.db")
    finally:
        session.close()


# ┌────────────────────────────────────┐
# │                Main                │
# └────────────────────────────────────┘


def main():
    session = Session()
    try:
        existing_urls = {row[0] for row in session.query(Article.url).all()}
        candidate_urls = get_links()

        print(f"Found {len(candidate_urls)} URLs. Scraping content...")
        scraped = scrape_articles_threaded(candidate_urls)

        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=DAYS_LIMIT)).date()
        new_articles = [
            article
            for article in scraped
            if article["url"] not in existing_urls and article["date"] >= cutoff_date
        ]

        print(f"Inserting {len(new_articles)} new articles...")
        insert_articles(new_articles)

    finally:
        session.close()


if __name__ == "__main__":
    main()
