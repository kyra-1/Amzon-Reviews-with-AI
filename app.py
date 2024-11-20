from flask import Flask, request, render_template
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import gzip
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt
import openai
import os
import io
import base64
from collections import Counter

# Flask app initialization
app = Flask(__name__, static_folder='static')

# Load data
data = []
with gzip.open('AMAZON_FASHION.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

metadata = []
with gzip.open('meta_AMAZON_FASHION.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        metadata.append(json.loads(line.strip()))

df = pd.DataFrame.from_dict(data)
df = df[df['reviewText'].notna()]  # Drop rows without reviews
df_meta = pd.DataFrame.from_dict(metadata)

# Function to generate plots as base64 strings
def generate_plot(plot_func, *args, **kwargs):
    buf = io.BytesIO()
    plot_func(*args, **kwargs)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    return plot_data

# Function to visualize word frequencies
def plot_word_frequencies(positivedata, negdata):
    positive_words = ' '.join(positivedata).split()
    negative_words = ' '.join(negdata).split()

    # Calculate word counts
    positive_counts = Counter(positive_words).most_common(10)
    negative_counts = Counter(negative_words).most_common(10)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Positive data
    axes[0].barh(
        [word for word, _ in positive_counts],
        [count for _, count in positive_counts],
        color='green'
    )
    axes[0].set_title("Top Positive Words")
    axes[0].invert_yaxis()

    # Negative data
    axes[1].barh(
        [word for word, _ in negative_counts],
        [count for _, count in negative_counts],
        color='red'
    )
    axes[1].set_title("Top Negative Words")
    axes[1].invert_yaxis()

    # Save as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Display LDA topics
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topics

# OpenAI API interaction
def analyze_product_reviews(mode, reviews_data):
    prompt = f"Analyze {'positive' if mode == 'positive' else 'negative'} reviews: {reviews_data}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Flask routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/product_info', methods=['POST'])
def product_info():
    asin = request.form['asin']
    product_df = df[df['asin'] == asin]

    if product_df.empty:
        return render_template('index.html', error="No product found with this ASIN.")

    # Ratings distribution plot
    ratings_count = product_df['overall'].value_counts().sort_index()
    def plot_ratings():
        ratings_count.plot(kind='bar', color='skyblue')
        plt.title('Ratings Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')

    ratings_plot = generate_plot(plot_ratings)

    # Product metadata
    product_meta = df_meta[df_meta['asin'] == asin]
    if product_meta.empty:
        product_title, product_description = "No Title", "No Description"
    else:
        product_meta = product_meta.iloc[0]
        product_title = product_meta.get('title', 'No Title')
        product_description = product_meta.get('description', 'No Description')

    return render_template(
        'product_info.html',
        title=product_title,
        description=product_description,
        asin=asin,
        ratings_plot=ratings_plot,
        ratings_count = ratings_count.to_dict() if not ratings_count.empty else None
    )

@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    asin = request.form['asin']
    review_type = request.form['review_type']

    filtered_reviews = df[(df['asin'] == asin) & (df['overall'] >= 4 if review_type == 'positive' else df['overall'] <= 3)]['reviewText']

    # TF-IDF and LDA
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(2, 2))
    data_vectorized = vectorizer.fit_transform(filtered_reviews)

    lda = LDA(n_components=5, random_state=0)
    lda.fit(data_vectorized)

    tf_feature_names = vectorizer.get_feature_names_out()
    topics = display_topics(lda, tf_feature_names, 10)

    # LDA plot
    def plot_lda():
        plt.figure(figsize=(8, 4))
        for idx, topic in enumerate(lda.components_):
            plt.plot(topic, label=f"Topic {idx+1}")
        plt.title("LDA Topic Distributions")
        plt.xlabel("Terms")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid(alpha=0.5)

    lda_plot = generate_plot(plot_lda)

    # Positive/negative word frequencies
    positivedata = df[(df['asin'] == asin) & (df['overall'] >= 4)]['reviewText'].tolist()
    negdata = df[(df['asin'] == asin) & (df['overall'] <= 3)]['reviewText'].tolist()
    word_frequencies_plot = plot_word_frequencies(positivedata, negdata)

    # ChatGPT analysis
    chatgpt_response = analyze_product_reviews(review_type, topics)

    return render_template(
        'results.html',
        lda_topics=topics,
        lda_plot=lda_plot,
        word_frequencies_plot=word_frequencies_plot,
        chatgpt_response=chatgpt_response
    )

if __name__ == '__main__':
    app.run(debug=True)
