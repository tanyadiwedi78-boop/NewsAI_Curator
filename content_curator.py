import os
import json
import time
import smtplib
import feedparser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Literal
from pathlib import Path
from datetime import date

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from groq import Groq

load_dotenv()


# ============================================================
# 1. SCHEMA — LLM follows structure
# ============================================================

class Article(BaseModel):
    title: str
    source: str
    url: str
    summary: str = Field(..., description="1-2 line business-relevant summary")
    category: Literal["Research", "Product Launch", "Funding", "Policy", "Other"]
    relevance_score: int = Field(..., ge=1, le=10)
    tags: list[str] = Field(default_factory=list)


class FilterResult(BaseModel):
    keep: bool
    reason: str


class Digest(BaseModel):
    generated_at: str
    articles: list[Article]


# ============================================================
# 2. INGEST — RSS feeds se raw, messy, unstructured text
# ============================================================

RSS_FEEDS = {
    "TechCrunch AI": "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "Wired AI": "https://www.wired.com/feed/tag/ai/latest/rss",
    "MIT Technology Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
}

LOOKBACK_HOURS = 24 * 7  # past week


@dataclass
class RawArticle:
    title: str
    source: str
    url: str
    published: datetime
    raw_text: str


def _parse_date(entry) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        if getattr(entry, key, None):
            return datetime.fromtimestamp(time.mktime(getattr(entry, key)), tz=timezone.utc)
    return datetime.now(timezone.utc)


def fetch_rss_articles() -> list[RawArticle]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    articles: list[RawArticle] = []

    for source_name, feed_url in RSS_FEEDS.items():
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            published = _parse_date(entry)
            if published < cutoff:
                continue
            raw_text = getattr(entry, "summary", "") or getattr(entry, "title", "")
            articles.append(RawArticle(
                title=getattr(entry, "title", "Untitled"),
                source=source_name,
                url=getattr(entry, "link", ""),
                published=published,
                raw_text=raw_text,
            ))
    return articles


# ============================================================
# 3. LLM CLIENT — Groq API call wrapper
# ============================================================

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Check your .env file.")
        _client = Groq(api_key=api_key)
    return _client


def call_llm(prompt: str, model: str = "openai/gpt-oss-120b") -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You return ONLY valid JSON. No markdown fences, no preamble."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# 4. EXTRACT — raw text -> LLM -> validated Article (+ retry on failure)
# ============================================================

MAX_RETRIES = 2


extract_stats = {"first_try": 0, "retried_success": 0, "failed": 0}


def _build_extract_prompt(article: RawArticle, error_context: str = "") -> str:
    schema_json = json.dumps(Article.model_json_schema(), indent=2)
    correction = (
        f"\nYour previous response was INVALID. Error: {error_context}\n"
        "Fix it and return ONLY valid JSON matching the schema.\n"
        if error_context else ""
    )
    return f"""


{schema_json}

Article title: {article.title}
Source: {article.source}
Article text: {article.raw_text}
{correction}
"""


def extract_article(raw: RawArticle) -> Article | None:
    error_context = ""
    for attempt in range(MAX_RETRIES + 1):
        prompt = _build_extract_prompt(raw, error_context)
        response_text = call_llm(prompt)
        try:
            data = json.loads(response_text)
            data.setdefault("url", raw.url)
            data.setdefault("source", raw.source)
            article = Article(**data)
            if attempt == 0:
                extract_stats["first_try"] += 1
            else:
                extract_stats["retried_success"] += 1
            return article
        except (json.JSONDecodeError, ValidationError) as e:
            error_context = str(e)
            continue

    extract_stats["failed"] += 1
    print(f"[extract] Gave up on '{raw.title[:50]}...' after {MAX_RETRIES} retries.")
    return None


def extract_articles(raw_articles: list[RawArticle]) -> list[Article]:
    results = []
    for raw in raw_articles:
        article = extract_article(raw)
        if article:
            results.append(article)

    total = sum(extract_stats.values())
    if total:
        print(
            f"[extract] Reliability: {extract_stats['first_try']}/{total} first-try, "
            f"{extract_stats['retried_success']}/{total} retried, "
            f"{extract_stats['failed']}/{total} failed."
        )
    return results


# ============================================================
# 5. FILTER — dedup/quality check, 2nd structured LLM call
# ============================================================

def _build_filter_prompt(article: Article, seen_titles: list[str]) -> str:
    schema_json = json.dumps(FilterResult.model_json_schema(), indent=2)
    return f"""
Decide whether to KEEP this article in an AI news digest.
Reject it if it's a near-duplicate of any title already seen, or not
substantively about AI.

Return ONLY valid JSON matching this schema:
{schema_json}

Titles already kept: {seen_titles}

Candidate article:
Title: {article.title}
Summary: {article.summary}
Category: {article.category}
"""


def _decide(article: Article, seen_titles: list[str]) -> FilterResult:
    for _ in range(MAX_RETRIES + 1):
        prompt = _build_filter_prompt(article, seen_titles)
        response_text = call_llm(prompt)
        try:
            data = json.loads(response_text)
            return FilterResult(**data)
        except (json.JSONDecodeError, ValidationError):
            continue
    return FilterResult(keep=True, reason="filter step failed, defaulted to keep")


def filter_articles(articles: list[Article]) -> list[Article]:
    kept: list[Article] = []
    seen_titles: list[str] = []
    for article in sorted(articles, key=lambda a: a.relevance_score, reverse=True):
        decision = _decide(article, seen_titles)
        if decision.keep:
            kept.append(article)
            seen_titles.append(article.title)
        else:
            print(f"[filter] Dropped '{article.title[:50]}...' — {decision.reason}")
    return kept


# ============================================================
# 6. EXPORT — JSON file
# ============================================================

DIGEST_FILE = Path("output/news_digest.json")

def save_json(articles: list):
    DIGEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    
    if DIGEST_FILE.exists():
        with open(DIGEST_FILE, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    today = str(date.today())  # "2026-09-02"
    all_data[today] = [a.model_dump() for a in articles]  

    with open(DIGEST_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"[export] Updated {DIGEST_FILE} with {len(articles)} articles for {today}")



# ============================================================
# 7. EMAIL — HTML formatting + send (no LLM call needed here)
# ============================================================

def build_html(articles: list[Article]) -> str:
    rows = ""
    for a in articles:
        rows += f"""
        <div style="margin-bottom:16px; padding:12px; border-left:4px solid #4a4a4a;">
            <div style="font-weight:bold;">{a.title}</div>
            <div style="font-size:12px; color:#666;">{a.source} · {a.category} · relevance {a.relevance_score}/10</div>
            <div style="margin-top:4px;">{a.summary}</div>
            <div style="margin-top:4px;"><a href="{a.url}">Read more</a></div>
        </div>
        """
    return f"<html><body><h2>AI News Digest</h2><p>{len(articles)} articles curated for you.</p>{rows}</body></html>"


def send_email(articles: list[Article]) -> None:
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    recipients = [r.strip() for r in os.environ.get("EMAIL_RECIPIENT", "").split(";") if r.strip()]

    required = {
        "EMAIL_SENDER": sender,
        "EMAIL_PASSWORD": password,
        "SMTP_SERVER": smtp_server,
        "EMAIL_RECIPIENT": recipients,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "[email] Missing SMTP config in .env: " + ", ".join(missing) + ". "
            "Check the spelling of these variable names (e.g. EMAIL_RECIPIENT, not EMAIL_RECEPIENT)."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"AI News Digest — {len(articles)} articles"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(build_html(articles), "html"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())
    print(f"[email] Sent digest to {len(recipients)} recipient(s).")


# ============================================================
# 8. MAIN 
# ============================================================

def main():
    print("=== Step 1: Ingest ===")
    raw_articles = fetch_rss_articles()
    print(f"Fetched {len(raw_articles)} raw articles.")
    if not raw_articles:
        print("No articles found in lookback window. Exiting.")
        return

    print("\n=== Step 2: Extract (structured, with retry) ===")
    articles = extract_articles(raw_articles)
    print(f"Successfully structured {len(articles)}/{len(raw_articles)} articles.")

    print("\n=== Step 3: Filter (dedup / quality, structured) ===")
    kept_articles = filter_articles(articles)
    print(f"Kept {len(kept_articles)}/{len(articles)} after filtering.")

    print("\n=== Step 4: Export ===")
    save_json(kept_articles)


    print("\n=== Step 5: Email ===")
    send_email(kept_articles)

    print("\n=== Done ===")
    print(f"Extraction reliability stats: {extract_stats}")


if __name__ == "__main__":
    main()