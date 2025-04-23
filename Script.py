# Imports
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
import re
import  streamlit as st
from datetime import datetime


load_dotenv()  # Load variables from .env file

print(os.getcwd())  # Print the current working directory to confirm it's correct
# Now you can access them like this:
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

print("OpenAI Key:", openai_key)
print("Tavily Key:", tavily_key)

# Sample tool definition to get you started, you can explore further and more~
tool = TavilySearchResults(
    tavily_api_key = tavily_key,
    max_results=2,
    topic = "news",
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    include_domains=["reuters.com", "theguardian.com", "wsj.com", "ft.com", "insurtechdigital.com", 
     "beinsure.com", "insuranceerm.com", "reinasia.com", "bloomberg.com", "businessinsurance.com", 
     "riskandinsurance.com", "carriermanagement.com", "artemis.bm"], # include only these domains
    exclude_domains=["insurance-news-today.xyz", "bestinsurancenews.info", "financialmarketupdate.blogspot.com", 
     "worldfinancialalerts.com", "insurancebreakingnews.biz", "insurtechhype.online", 
     "newsinsurancehub.com", "globalinsuranceinsights.site", "climaterisktrends.info", 
     "quickinsurancenews.top", "reinsuranceupdate.live", "cryptoinsuranceinvest.net"]
    # ,
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)


tool_researchpaper = TavilySearchResults(
    tavily_api_key = tavily_key,
    max_results=1,
    topic = "research",
    include_raw_content= True,
    search_depth="advanced",
    include_answer=True,
    include_images=True,
    include_domains=[
    "arxiv.org",
    "jstor.org",
    "ieee.org",
    "springer.com",
    "nature.com",
    "sciencedirect.com",
    "researchgate.net",
    "mdpi.com",
    "tandfonline.com",
    "acm.org",
    "plos.org",
    "cambridge.org",
    "sciencemag.org",
    "ncbi.nlm.nih.gov",  # PubMed
    "biorxiv.org",
    "ssrn.com",
    "paperswithcode.com",
    "openaccess.thecvf.com",
    "doi.org",
    "frontiersin.org",
    ".edu"  # general academic domain
] # include only these domains
    # ,
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

response = tool.invoke({"query": 'Climate risk insurance trends'})

def generate_search_query(article_text, model="gpt-3.5-turbo"):
    """
    Uses LangChain's ChatOpenAI to turn a news article into an academic research query.
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key = openai_apikey)
    if not article_text:
        return ""

    try:
        response = model.invoke([
            SystemMessage(content="You generate precise academic research queries based on news articles."),
            HumanMessage(content=f"Generate a precise academic search query to find research papers supporting this article:\n\n{article_text}")
        ])
        return response.content.strip()

    except Exception as e:
        print("Error generating search query:", e)
        return ""

def extract_date(response):
    try:
        # First check if the response contains a direct published date
        published_date = response[0].get("published_date", None)
        if published_date:
            return published_date  # Return if already present
        
        # Otherwise, extract from raw_content using regex
        raw_content = response.get("raw_content", None)
        if not raw_content:
            raw_content = response.get("content", "")
        
        # Regular expression to match published date/time with keywords
        date_pattern = r'(?:Published (?:Date|Time):\s*)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)|' \
                       r'(?:Published (?:Date|Time):\s*)([A-Za-z]+\s\d{1,2},\s\d{4})'

        # Find all matches
        matches = re.findall(date_pattern, raw_content)

        # Extract only the date component from matched results
        extracted_dates = [match[0] or match[1] for match in matches if match[0] or match[1]]

        if extracted_dates:
            # Extract only the YYYY-MM-DD portion
            date_pattern = r'\d{4}-\d{2}-\d{2}'
            match = re.search(date_pattern, extracted_dates[0])
            if match:
                return match.group()
        
        # If no date found, return None
        return None

    except Exception as e:
        print(f"Error: {e}")  # Log any unexpected errors
        return None

  def extract_tags_from_article(content, model="gpt-3.5-turbo"):
    if not content:
        return []
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key = openai_apikey)
    system_prompt = "You are an assistant that extracts relevant tags from news articles."
    user_prompt = (
        f"Extract 3 to 5 relevant tags for the following news article:\n\n"
        f"{content}\n\n"
        f"Only return a comma-separated list of tags."
    )

    try:
        # Send message to the LLM
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        tag_string = response.content
        tags = [tag.strip() for tag in tag_string.split(",") if tag.strip()]
        return tags

    except Exception as e:
        print("Error extracting tags:", e)
        return []

def structure_the_response(response):
    # Placeholder for processing response data
    structured_resp = []
    for res in response:
        structured_resp.append({'title' : res['title'],
         'source' : res['url'].split('/')[2].split('.')[0],
         'summary' : res['content'],
         'date' : extract_date(res),
         'tags' : extract_tags_from_article(res['content'], model="gpt-3.5-turbo")})
    return structured_resp


structured_response = structure_the_response(response)


def find_research_references_correlating_with_each_news_snnipets(structured_response):
    """
    Your code goes here.
    This function is set as a placeholder to get guide to the results expectations. You can always have own approach to set the functions.
    

    Expection with this block is to have a comparison of the news with the ground truth so that the legitimacy of the report could be
    maintanined.
    """
    research_list = []
    for resp in structured_response:
        search_query = generate_search_query(resp)
        research_response = tool_researchpaper.invoke({"query": search_query})
        research_list.append({'title' : research_response[0].get('title'),
         'source' : research_response[0].get('url').split('/')[2].split('.')[0],
         'summary' : research_response[0].get('content'),
         'date' : extract_date(research_response[0].get('content')),
         'tags' : extract_tags_from_article(research_response[0].get('content'), model="gpt-3.5-turbo")})
    
    enriched_responses = structured_response
    references_dict = research_list

    return enriched_responses, references_dict


enriched_responses, references_dict = find_research_references_correlating_with_each_news_snnipets(structured_response)

news_articles, research_papers = enriched_responses, references_dict 

def parse_date_safe(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None  # fallback for missing or invalid dates

def define_ui_and_visual_elements(enriched_responses, references_dict):
    """
    Enriches news articles by linking them with related research papers based on shared tags.
    Handles missing or malformed dates gracefully.
    """

    # Build tag-to-reference map from research papers
    tag_to_references = {}
    for paper in references_dict:
        for tag in paper.get("tags", []):
            if tag not in tag_to_references:
                tag_to_references[tag] = []
            tag_to_references[tag].append({
                "title": paper["title"],
                "date": paper.get("date"),
                "date_obj": parse_date_safe(paper.get("date")),
            })

    # Build enriched report from news articles
    enriched_report = []
    for article in enriched_responses:
        structured = {
            "title": article.get("title"),
            "source": article.get("source"),
            "summary": article.get("summary"),
            "date": article.get("date"),
            "date_obj": parse_date_safe(article.get("date")),
            "tags": article.get("tags", []),
            "references": []
        }

        # Match references by tags
        for tag in structured["tags"]:
            if tag in tag_to_references:
                structured["references"].extend(tag_to_references[tag])

        # Sort references by date (if available)
        structured["references"].sort(
            key=lambda r: r["date_obj"] or datetime.min,
            reverse=True
        )

        enriched_report.append(structured)

    # Sort enriched articles by date (if available)
    enriched_report.sort(
        key=lambda a: a["date_obj"] or datetime.min,
        reverse=True
    )

    # Clean up before return (optional: remove internal date_obj fields)
    for entry in enriched_report:
        entry.pop("date_obj", None)
        for ref in entry["references"]:
            ref.pop("date_obj", None)

    return enriched_report


### Functional Design
def filter_by_tag(data, selected_tag):
    """Filter a list of items by checking if the selected tag is in their tags list."""
    return [item for item in data if selected_tag in item["tags"]]

def get_all_tags(datasets):
    """Extract a sorted list of unique tags from multiple datasets."""
    tags = set()
    for data in datasets:
        for item in data:
            tags.update(item["tags"])
    return sorted(tags)


### Layout design 
st.title("Climate Risk Insurance Dashboard")

# Sidebar for tag selection
st.sidebar.header("Filter by Tag")
all_tags = get_all_tags([news_articles, research_papers])
selected_tag = st.sidebar.selectbox("Select a tag", all_tags)

# Display filtered News Articles
st.header(f"News Articles Tagged: {selected_tag}")
filtered_news = filter_by_tag(news_articles, selected_tag)
if filtered_news:
    for article in filtered_news:
        st.subheader(article["title"])
        st.caption(f"{article['source']} | {article['date']}")
        st.write(article["summary"])
        st.markdown("---")
else:
    st.write("No news articles match this tag.")

### Interactions output Design
st.header(f"Research Papers Tagged: {selected_tag}")
filtered_research = filter_by_tag(research_papers, selected_tag)
if filtered_research:
    for paper in filtered_research:
        st.subheader(paper["title"])
        st.caption(f"{paper['authors']} | {paper['date']}")
        st.write(paper["abstract"])
        st.markdown("---")
else:
    st.write("No research papers match this tag.")

st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard integrates news and academic research on climate risk insurance. "
    "Select a tag to see related items from both news and arXiv research."
)
