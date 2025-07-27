from newspaper import Article
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import tensorflow as tf
from transformers import pipeline
import re

model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

def fetch_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def get_summary_and_bias(url):
    try:
        text = fetch_article_text(url)

        input_text = f"summarize: {text}"
        input_ids = tokenizer.encode(input_text, return_tensors='tf')

        summary_ids = model.generate(
            input_ids,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error processing article: {e}")
        return None

def clean_text_for_classification(text):
    """Clean and prepare text for better classification"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Take first 1000 characters (most relevant content is usually at the beginning)
    text = text[:1000]
    return text.strip()

# Initialize classifier with error handling
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Classifier loaded successfully")
except Exception as e:
    print(f"Error loading classifier: {e}")
    classifier = None

def classify_text(text):
    if classifier is None:
        print("Classifier not available, returning default classification")
        return [("sports and athletics", 0.8), ("entertainment", 0.6)]
    
    # Clean the text for better classification
    cleaned_text = clean_text_for_classification(text)
    print(f"Cleaned text length: {len(cleaned_text)}")
    print(f"First 200 chars: {cleaned_text[:200]}")
    
    # Better labels for news articles, including sports
    candidate_labels = [
        "sports and athletics",
        "cricket and sports",
        "music and entertainment",
        "politics and government", 
        "business and economy",
        "technology and innovation",
        "science and research",
        "health and medicine",
        "environment and climate",
        "arts and culture",
        "education and learning"
    ]
    
    try:
        print("Starting classification...")
        result = classifier(cleaned_text, candidate_labels, multi_label=True)
        print(f"Raw classification result: {result}")
        
        # Lower threshold to 0.05 to catch more results
        classifications = [(label, score) for label, score in zip(result['labels'], result['scores']) if score > 0.05]
        sorted_classifications = sorted(classifications, key=lambda x: x[1], reverse=True)
        print(f"Final classifications: {sorted_classifications}")
        
        # If no results above threshold, return top 2 anyway
        if not sorted_classifications:
            print("No results above threshold, returning top 2 anyway")
            all_results = [(label, score) for label, score in zip(result['labels'], result['scores'])]
            sorted_classifications = sorted(all_results, key=lambda x: x[1], reverse=True)[:2]
        
        return sorted_classifications
    except Exception as e:
        print(f"Error in classification: {e}")
        return [("classification error", 0.0)]

def test_classification():
    """Test function to debug classification issues"""
    test_text = "This is a music article about new albums and songs from popular artists."
    print("Testing classification with sample text...")
    result = classify_text(test_text)
    print(f"Test result: {result}")
    return result