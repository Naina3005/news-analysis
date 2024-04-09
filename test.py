import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load Excel file into a DataFrame
df = pd.read_excel('Assignment.xlsx')  

# Extract text data from the DataFrame
articles = df['Article'].tolist()

# Clean up articles
def preprocess(article):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(article.lower())
    cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    cleaned_article = ' '.join(cleaned_tokens)
    return cleaned_article

cleaned_articles = [preprocess(article) for article in articles]

# Check the mood of articles
def analyze_mood(cleaned_article):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(cleaned_article)
    if sentiment_scores['compound'] >= 0.05:
        mood = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        mood = 'Negative'
    else:
        mood = 'Neutral'
    return mood

moods = [analyze_mood(article) for article in cleaned_articles]

# Find connections using Latent Dirichlet Allocation (LDA)
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(cleaned_articles)
num_topics = 3  # Adjust as needed
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(tfidf_matrix)
feature_names = vectorizer.get_feature_names_out()
topics = []
for idx, topic in enumerate(lda_model.components_):
    topic_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    topics.append(topic_words)

# Print results
for i, article in enumerate(articles):
    print(f"Article {i+1}:")
    print(f"Mood: {moods[i]}")
    print(f"Cleaned Article: {cleaned_articles[i]}")

    print()

print("Topics:")
for idx, topic_words in enumerate(topics):
    print(f"Topic {idx+1}: {' '.join(topic_words)}")   

output_text = df.head().to_string()

# Write the output to a text file
with open('output.txt', 'w') as f:
    f.write(output_text)

print("Output written to 'output.txt' file.")
