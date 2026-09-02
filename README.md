# AI News RSS Aggregator

An automated pipeline that collects, filters, and summarizes AI-related news and research from various sources, delivering a curated digest via email.

## Overview

This tool helps you stay informed about the latest developments in AI by:
1. Pulls recent articles from AI news RSS feeds (TechCrunch AI, Wired AI, MIT Technology Review AI).
2. Filtering for only the most recent articles (past week)
3. Removing duplicates and low-quality articles
4. Extracting and summarizing content
6. Delivering a digest via email with a CSV attachment

## Features

- **Multi-source collection**: Aggregates content from RSS feeds and blogs
- **Intelligent filtering**: Uses LLM-based filtering to eliminate duplicates and low-quality content
- **Concise summaries**: Generates business-relevant, single-line summaries of each article
- **Email delivery**: Sends a formatted HTML digest with all articles
- **CSV download**: Includes a CSV attachment with all article data for reference or analysis

## Requirements

- Python 3.8+
- Required libraries:
  - feedparser
  - openai (or similar client)
  - Groq
  - pydantic

## Configuration

Set the following environment variables before running:

```
GROQ_API_KEY=your_fireworks_api_key
SMTP_SERVER=your_smtp_server
SMTP_PORT=your_smtp_port
EMAIL_SENDER=your_sender_email
EMAIL_PASSWORD=your_email_password
EMAIL_RECIPIENT=recipient1@example.com; recipient2@example.com
```

## Usage

Simply run the script:

```
content_curator.py
```

The script will:
1. Fetch articles from the configured RSS feeds 
2. Process and filter them
3. Send an email digest to the configured recipients

## Customization

### Adding RSS Sources

Modify the `RSS_FEEDS` dictionary to add or remove sources:

```python
RSS_FEEDS = {  
    "Wired AI": "https://www.wired.com/feed/tag/ai/latest/rss",
    "TechCrunch AI": "https://techcrunch.com/tag/artificial-intelligence/feed/",
    # Add your sources here
}
```

## How It Works

1. **Article Collection**:
   - `fetch_rss_articles()`: Collects articles from RSS feeds (TechCrunch AI, Wired AI, MIT Tech Review AI) published in the last 7 days

2. **Structured Extraction**:
   - `extract_article()`: Sends each raw article to the LLM and converts it into a structured summary (title, category, relevance score, tags) validated against a strict schema — automatically retries if the LLM's output doesn't match the schema

3. **Quality Control**:
   - `filter_articles()`: A second LLM call removes duplicates and low-relevance articles, keeping only the most useful ones

4. **Delivery**:
   - `save_json()`: Saves the day's articles into `output/news_digest.json`, organized by date
   - `build_html()`: Formats the kept articles into an HTML digest
   - `send_email()`: Delivers the digest to recipients via SMTP

## Troubleshooting

- **No articles being returned**: Check that your RSS feeds and blog URLs are valid and contain recent content
- **Email delivery issues**: Verify your SMTP settings and email credentials

## Automation

A GitHub Actions workflow runs the pipeline every day at 9:00 AM IST and commits the updated digest automatically.

