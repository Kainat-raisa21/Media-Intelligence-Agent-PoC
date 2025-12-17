
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json
import csv
import pandas as pd
from urllib.parse import urljoin
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage



#loading groq api key
load_dotenv()
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")





######## ________    BBC Screapper __________   ########
def fetch_bbc_headlines() -> pd.DataFrame:
    """Scrapes news data from BBC website"""


    base_url = 'https://www.bbc.com'
    url = 'https://www.bbc.com/news'
        
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from {url}: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all article containers
    articles = soup.find_all('a', attrs={'data-testid': 'internal-link'})

    news_list = []

    for idx, article in enumerate(articles, start=1):
        headline_tag = article.find('h2', attrs={'data-testid': 'card-headline'})
        summary_tag = article.find('p', attrs={'data-testid': 'card-description'})

        if not headline_tag:
            continue

        headline_text = headline_tag.get_text(strip=True)
        summary_text = summary_tag.get_text(strip=True) if summary_tag else "No summary available"

        # Get article link and build absolute URL
        article_url = article.get('href')
        if not article_url.startswith('http'):
            article_url = urljoin(base_url, article_url)

        # Fetch full article text
        full_text = ""
        try:
            article_response = requests.get(article_url)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.text, 'html.parser')

            # Extract paragraphs from article content
            content_div = article_soup.find('div', class_='article__content-container')
            if content_div:
                paragraphs = content_div.find_all('p')
                full_text = ' '.join(p.get_text(strip=True) for p in paragraphs)
            else:
                # Try fallback: find any article text container
                paragraphs = article_soup.find_all('p')
                full_text = ' '.join(p.get_text(strip=True) for p in paragraphs[:10])  # fallback limit

        except requests.exceptions.RequestException as e:
            print(f"Error fetching full text for {article_url}: {e}")

        news_list.append({
                "headline": headline_text,
                "summary": summary_text,
                "full_text": full_text or "No article text found",
                "url": article_url
            })

    # Convert to DataFrame
    df = pd.DataFrame(news_list, columns=["headline", "summary", "full_text", "url"])

    return df


    


#############  -----------------  CNN Scraper ______________  #################
def fetch_cnn_headlines() -> pd.DataFrame:
    """ Scrapes news from CNN website"""
    
    BASE_URL = "https://edition.cnn.com"
    SECTION_URL = BASE_URL + "/world"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    }

    def get_world_articles():
        """Fetch article titles and links from the CNN World page."""
        try:
            resp = requests.get(SECTION_URL, headers=headers, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching CNN World page: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []

        for li in soup.find_all("li", class_=lambda c: c and "card" in c.split()):
            a_tag = li.find("a", href=True, class_=lambda c: c and "container__link" in c.split())
            if not a_tag:
                continue

            link = urljoin(BASE_URL, a_tag["href"])
            title_tag = li.find("span", class_="container__headline-text")
            title = title_tag.get_text(strip=True) if title_tag else a_tag.get_text(strip=True)

            if title and link:
                articles.append({"title": title, "link": link})


        return articles

    def get_article_text(article_url: str) -> str:
        """Extract the full text content from a CNN article."""
        try:
            resp = requests.get(article_url, headers=headers, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching article {article_url}: {e}")
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # Primary selector for CNN articles
        content_div = soup.find("div", class_="article__content")
        if not content_div:
            # Fallback: get all <p> tags
            paragraphs = soup.find_all("p")
        else:
            paragraphs = content_div.find_all("p")

        full_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
        return full_text.strip()

    # Fetch articles
    articles = get_world_articles()
    news_data = []

    for art in articles:
        full_text = get_article_text(art["link"])
        news_data.append({
            "title": art["title"],
            "link": art["link"],
            "full_text": full_text if full_text else "No article text found"
        })

    # Convert to DataFrame
    df = pd.DataFrame(news_data, columns=["title", "link", "full_text"])
    return df





llm_with_tools = llm.bind_tools([fetch_bbc_headlines, fetch_cnn_headlines])


sys_msg = SystemMessage(
    content=(
        "You are an expert media intelligence analyst specialized in evaluating and comparing news coverage across multiple sources. "
        "Your task is to analyze news articles related to a specific topic requested by the user, identify factual overlaps, differences, and detect potential bias or inconsistencies in tone, framing, or reporting. "
        "Generate a concise, structured analytical report that includes: "
        "1. Summary of the topic based on the collected articles. "
        "2. Comparison of how different sources covered the same issue. "
        "3. Bias and consistency analysis (identify emotional tone, political leaning, or framing bias). "
        "4. Key factual takeaways agreed upon across outlets. "
        "5. Contradictions or missing perspectives, if any. "
        "6. Final analytical summary written in a neutral, professional tone. "
        "When using insights or quotes from scraped articles, include the source and article link in your response."
    )
)



def media_intelligence_assistant(state: MessagesState):
    response = llm_with_tools.invoke([sys_msg] + state["messages"])

    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("media_intelligence_assistant", media_intelligence_assistant)
builder.add_node("tools", ToolNode([fetch_bbc_headlines, fetch_cnn_headlines]))

builder.add_edge(START, "media_intelligence_assistant")  # Fix: Use string instead of function reference

builder.add_conditional_edges("media_intelligence_assistant", tools_condition)
builder.add_edge("tools", "media_intelligence_assistant")


graph = builder.compile()

query = input("Hey, how can i help you?")

messages = [HumanMessage(content=query)]
messages = graph.invoke({"messages": messages})


for m in messages["messages"]:
    m.pretty_print()
