import os
import pandas as pd
from gnews import GNews
from datetime import datetime
from langid import classify
from joblib import Parallel, delayed
import streamlit as st
import xlsxwriter
from io import BytesIO

# Function to check if the article is in English
def is_english(text):
    lang, _ = classify(text)
    return lang == 'en'

# Function to get full article using newspaper3k and perform NLP
def get_full_article_with_nlp(url):
    article = GNews().get_full_article(url)

    # Perform NLP on the article
    try:
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        print(f"Error performing NLP on the article: {e}")

    return {
        'title': article.title,
        'text': article.text,
        'authors': article.authors,
        'summary': article.summary,
        'keywords': article.keywords,
    }

def process_article(article):
    source = article.get('source', 'N/A')
    if source == 'N/A':
        # Check if the source is still not available and try to extract it from the URL
        url = article.get('url', '')
        if url:
            source_from_url = url.split('/')[2]  # Extract domain from the URL
            source = source_from_url if source_from_url else 'N/A'

    if is_english(article.get('title', '')):
        title = article.get('title', 'N/A')

        # Extract information after " ... - " or " - " in the title
        if ' ... - ' in title:
            title_parts = title.split(' ... - ')
        elif ' - ' in title:
            title_parts = title.split(' - ')
        else:
            title_parts = [title]

        article_source = title_parts[1].strip() if len(title_parts) > 1 else 'N/A'

        result = {
            'Title': article.get('title', 'N/A'),
            'Source': source,
            'Article_Source': article_source,
            'URL': article.get('url', 'N/A'),
        }

        try:
            # Get the full article using newspaper3k and perform NLP
            full_article = get_full_article_with_nlp(article.get('url'))
            result.update(full_article)
        except Exception as e:
            print(f"Error processing article: {e}")

        return result
    else:
        return None

# Streamlit UI
st.title("News Scraper and Analyzer")
st.sidebar.header("Search Parameters")

# User input
query = st.sidebar.text_input("Enter Keywords (separated by space)", 'datacentre cloud')
start_date = st.sidebar.date_input("Enter Start Date", datetime(2023, 11, 1))
end_date = st.sidebar.date_input("Enter End Date", datetime(2023, 12, 30))

# Set the language filter
GNews.language = 'en'

# Initialize GNews object with query parameters
google_news = GNews(country='US', max_results=100)

# Search button
if st.sidebar.button("Search"):
    with st.spinner("Please wait while scraping the web..."):
        # Set the start and end dates based on UI input
        google_news.start_date = start_date
        google_news.end_date = end_date

        # Get news results
        news_results = google_news.get_news(query)

        # Parallel processing using joblib
        results_list = Parallel(n_jobs=-1)(delayed(process_article)(article) for article in news_results)

        # Filter out None values (articles not in English)
        results_list = [result for result in results_list if result is not None]
        # Process articles sequentially without joblib
        '''results_list = []
        for article in news_results:
            result = process_article(article)
            if result is not None:
                results_list.append(result)'''


        # Create a DataFrame from the list of results
        df = pd.DataFrame(results_list)

        # Filter out null values (articles not in English or with null title/text)
        #df = df.dropna(subset=['Title', 'text'])

        # Display results
        st.subheader("Results")
        st.dataframe(df)

        # Download button
        # Save the filtered DataFrame to an Excel file
        excel_file_path_filtered = f"{query.split()[0]}_{query.split()[1]}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_filtered.xlsx"
        df.to_excel(excel_file_path_filtered, index=False, engine='openpyxl')

        # Create an in-memory Excel file for the filtered DataFrame
        output_filtered = BytesIO()
        df.to_excel(output_filtered, index=False, engine='openpyxl')

        # Download button for the filtered DataFrame
        st.sidebar.download_button(
            label="Download Filtered Results",
            data=output_filtered.getvalue(),
            file_name=f"{query.split()[0]}_{query.split()[1]}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="filtered_excel",
            help="Click to download the filtered results"
        )
