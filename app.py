from flask import Flask, request, render_template
import pandas as pd
import gzip
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import openai
import os
import re
import json

# Initialize Flask app
app = Flask(__name__)

# Load data
data = []
with gzip.open('AMAZON_FASHION.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))

metadata = []
with gzip.open('meta_AMAZON_FASHION.json.gz') as f:
    for l in f:
        metadata.append(json.loads(l.strip()))

df = pd.DataFrame.from_dict(data)
df = df[df['reviewText'].notna()]
df_meta = pd.DataFrame.from_dict(metadata)

# Function to display LDA topics
def display_topics(model, feature_names, no_top_words):
    topic_summaries = []
    for topic_idx, topic in enumerate(model.components_):
        topic_summaries.append(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topic_summaries

def analyze_product_reviews(mode, reviews_data):

    # Define the prompt based on the mode
    if mode == 'positive':
        prompt = "I have collected a set of positive customer reviews for a product. These reviews reflect what customers enjoy and appreciate about the product. Please analyze the content of these reviews to identify and summarize the key positive aspects and features that customers highlight. Focus on understanding what makes customers happy with this product, any specific features or qualities they frequently praise, and any recurring patterns of satisfaction you observe.\n\n" + str(reviews_data)  
    elif mode == 'negative':
        prompt = "I have performed an LDA (Latent Dirichlet Allocation) analysis on a collection of product reviews and obtained the following topics. Each topic is represented by a list of its most significant words. Please analyze these topics to identify and summarize the main issues or problems customers have with the product\n\n" + str(reviews_data)
    else:
        return "Invalid mode selected."

    # Call the OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return str(e)


# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your open AI code "

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/product_info', methods=['POST'])
def product_info():
    asin = request.form['asin']
    product_df = df[df['asin'] == asin]

    if product_df.empty:
        return render_template('index.html', error="No product found with this ASIN.")

    ratings_count = product_df['overall'].value_counts().sort_index().to_dict()

    product_meta = df_meta[df_meta['asin'] == asin].iloc[0]
    product_title = product_meta.get('title', 'No Title')
    product_description = product_meta.get('description', 'No Description')

    return render_template('product_info.html', title=product_title, description=product_description, asin=asin, ratings_count=ratings_count)


@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    asin = request.form['asin']
    review_type = request.form['review_type']

    if review_type == 'positive':
        filtered_df = df[(df['asin'] == asin) & (df['overall'] >= 4)]['reviewText']
    else:  # Negative
        filtered_df = df[(df['asin'] == asin) & (df['overall'] <= 3)]['reviewText']

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(2, 2)) #Term Frequency-Inverse Document Frequency 
    
    data_vectorized = vectorizer.fit_transform(filtered_df)

    lda = LDA(n_components=15, random_state=0)
    lda.fit(data_vectorized)

    tf_feature_names = vectorizer.get_feature_names_out()
    top_words_per_topic = display_topics(lda, tf_feature_names, 15)

    chatgpt_response = analyze_product_reviews(review_type, top_words_per_topic)

    return render_template('results.html', lda_topics=top_words_per_topic, chatgpt_response=chatgpt_response)


if __name__ == '__main__':
    app.run(debug=True)
