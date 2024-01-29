import os
import pandas as pd
import re
import textwrap
from gnews import GNews
from datetime import datetime, timedelta
from langid import classify
from joblib import Parallel, delayed
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches
from io import BytesIO
import nltk
nltk.download('punkt')
# Function to check if the article is in English
def is_english(text):
    lang, _ = classify(text)
    return lang == 'en'

# Function to get full article using GNews and perform NLP
def get_full_article_with_nlp(url):
    article = GNews().get_full_article(url)
    try:
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        print(f"Error performing NLP on the article: {e}")
        return None
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
        url = article.get('url', '')
        source = url.split('/')[2] if url else 'N/A'
    if not is_english(article.get('title', '')):
        return None
    title_parts = re.split(' ... - | - ', article.get('title', 'N/A'))
    article_source = title_parts[1].strip() if len(title_parts) > 1 else 'N/A'
    result = {'Title1': article.get('title', 'N/A'), 'Source': source, 'Article_Source': article_source, 'URL': article.get('url', 'N/A')}
    full_article = get_full_article_with_nlp(article.get('url'))
    if full_article:
        result.update(full_article)
    return result

def setup_streamlit_ui():
    st.title("News Scraper and Analyzer")
    st.sidebar.header("Search Parameters")
    query = st.sidebar.text_input("Enter Keywords (separated by space)", 'aws cloud datacenter')
    end_date_default = datetime.now().date()
    start_date_default = end_date_default - timedelta(days=7)
    start_date = st.sidebar.date_input("Enter Start Date", start_date_default)
    end_date = st.sidebar.date_input("Enter End Date", end_date_default)
    return query, start_date, end_date

def fetch_and_process_news(query, start_date, end_date):
    GNews.language = 'en'
    google_news = GNews(country='US', max_results=100)
    google_news.start_date = start_date
    google_news.end_date = end_date
    news_results = google_news.get_news(query)
    results_list = Parallel(n_jobs=-1)(delayed(process_article)(article) for article in news_results)
    results_list = [result for result in results_list if result is not None]
    df = pd.DataFrame(results_list)
    df = df.dropna(subset=['Title1', 'text'])
    df = df[['Article_Source', 'title', 'text', 'summary', 'keywords', 'URL']]
    return df

def cluster_and_generate_wordclouds(df):
    df['processed_summary'] = df['summary'].apply(lambda x: re.sub('[^A-Za-z]+', ' ', str(x)).lower())
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_summary'])
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters, n_init=10)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    texts_per_cluster = [[df.iloc[i]['summary'] for i in range(len(clusters)) if clusters[i] == j] for j in range(num_clusters)]
    for i, texts in enumerate(texts_per_cluster):
        if not texts:
            texts_per_cluster[i] = ["No articles assigned to this cluster."]
    summaries = [texts_per_cluster[j][0] for j in range(num_clusters)]
    doc = Document()
    doc.add_heading('Cluster Summaries and Word Clouds', 0)
    for i, summary in enumerate(summaries):
        with st.container():
            expander = st.expander(f"Cluster {i+1} Summary and Word Cloud", expanded=True)
            with expander:
                formatted_summary = "\n".join(textwrap.wrap(summary, width=150))
                st.write(formatted_summary)
                doc.add_heading(f'Cluster {i+1} Summary:', level=1)
                doc.add_paragraph(formatted_summary)
                wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, min_font_size=10).generate(str(summary))
                plt.figure(figsize=(8, 8), facecolor=None)
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(plt)
                wordcloud_image_stream = BytesIO()
                plt.savefig(wordcloud_image_stream, format='png')
                plt.close()
                wordcloud_image_stream.seek(0)
                doc.add_picture(wordcloud_image_stream, width=Inches(6))
                doc.add_paragraph()
    output_doc = BytesIO()
    doc.save(output_doc)
    output_doc.seek(0)
    return output_doc

def main():
    query, start_date, end_date = setup_streamlit_ui()
    if st.sidebar.button("Search"):
        with st.spinner("Please wait while scraping the web..."):
            df = fetch_and_process_news(query, start_date, end_date)
            st.subheader("Results")
            st.dataframe(df)
            output_doc = cluster_and_generate_wordclouds(df)
            st.sidebar.download_button(
                label="Download Cluster Summary",
                data=output_doc.getvalue(),
                file_name=f"{query.split()[0]}_{query.split()[1]}_cluster_summary.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="cluster_summary_docx",
                help="Click to download the cluster summary document",
            )

if __name__ == "__main__":
    main()
