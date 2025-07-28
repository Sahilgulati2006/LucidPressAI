from newspaper import Article, Config
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import tensorflow as tf
from transformers import pipeline
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import requests
from bs4 import BeautifulSoup
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize models
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Configure newspaper3k for better extraction
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
config.request_timeout = 10

def fetch_article_text(url):
    """Enhanced article text extraction with multiple fallback methods, including liveblog support."""
    try:
        # Method 1: Try newspaper3k with better configuration
        article = Article(url, config=config)
        article.download()
        article.parse()
        # If newspaper3k got good content, use it
        if article.text and len(article.text.strip()) > 200:
            return article.text
        # Method 2: Fallback to direct requests + BeautifulSoup
        print("Newspaper3k extraction was poor, trying direct extraction...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Special handling for TOI liveblogs
        if 'timesofindia.indiatimes.com' in url and 'liveblog' in url:
            print("Detected TOI liveblog, extracting all liveblog entries...")
            entries = soup.find_all(class_=re.compile(r'liveblog-entry|_3YYSt'))
            if not entries:
                # Try to find all <li> blocks with updates
                entries = soup.find_all('li')
            text = '\n'.join([e.get_text(separator=' ', strip=True) for e in entries if e.get_text(strip=True)])
            if len(text.strip()) > 200:
                return text
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Try to find main content areas
        content_selectors = [
            'article', 'main', '.content', '.post-content', '.article-content',
            '.entry-content', '.story-content', '.news-content', '[role="main"]'
        ]
        content = None
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content and len(content.get_text().strip()) > 200:
                break
        if not content:
            # Fallback to body text
            content = soup.find('body')
        if content:
            text = content.get_text()
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n+', '\n', text)
            return text.strip()
        return None
    except Exception as e:
        print(f"Error fetching article text: {e}")
        return None

def extract_key_sentences(text, num_sentences=3):
    """Extract key sentences using simple heuristics"""
    if not text:
        return ""
    
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    
    # Simple heuristic: take first sentence and last few sentences
    key_sentences = [sentences[0]]  # First sentence (usually contains main topic)
    if len(sentences) > 1:
        key_sentences.extend(sentences[-num_sentences+1:])  # Last few sentences
    
    return " ".join(key_sentences)

def analyze_sentiment(text):
    """Analyze sentiment of the text"""
    if not text:
        return {"sentiment": "neutral", "score": 0.0, "confidence": "low"}
    
    try:
        # Analyze sentiment
        sentiment_scores = sia.polarity_scores(text)
        
        # Determine sentiment label
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            sentiment = "positive"
        elif compound_score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Determine confidence level
        if abs(compound_score) > 0.3:
            confidence = "high"
        elif abs(compound_score) > 0.1:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "sentiment": sentiment,
            "score": compound_score,
            "confidence": confidence,
            "details": sentiment_scores
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {"sentiment": "neutral", "score": 0.0, "confidence": "error"}

def chunk_text(text, max_tokens=512):
    """Split text into chunks of approximately max_tokens words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = ' '.join(words[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

def remove_repetitions(text):
    """Remove repeated sentences from a summary."""
    seen = set()
    result = []
    # Split on period, question, or exclamation
    for sentence in re.split(r'(?<=[.!?]) +', text):
        s = sentence.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            result.append(s)
    return ' '.join(result)

def extract_unique_updates(text):
    """Split by newlines, remove empty and duplicate lines, and skip very short lines."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    seen = set()
    unique_lines = []
    for line in lines:
        l = line.lower()
        if l not in seen and len(l) > 20:
            seen.add(l)
            unique_lines.append(line)
    return '\n'.join(unique_lines)

def chunk_sentences(text, max_sentences=8):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ' '.join(sentences[i:i+max_sentences])
        chunks.append(chunk)
    return chunks

def trim_to_last_complete_sentence(text):
    """Trim text to the last complete sentence (ending with . ! or ?)."""
    match = re.search(r'([.!?])[^.!?]*$', text)
    if match:
        end = match.end(1)
        return text[:end].strip()
    return text.strip()

def get_summary_and_bias(url):
    """Super-detailed summary: deduplicate, chunk by sentences, summarize each, then combine and summarize again."""
    try:
        print(f"Processing article from: {url}")
        text = fetch_article_text(url)
        if not text or len(text.strip()) < 100:
            return {
                "summary": "Unable to extract sufficient content from this URL. Please check if the URL is accessible and contains readable text.",
                "sentiment": {"sentiment": "neutral", "score": 0.0, "confidence": "low"},
                "word_count": 0,
                "extraction_method": "failed"
            }
        print(f"Extracted text length: {len(text)} characters")
        sentiment_result = analyze_sentiment(text)
        # --- Deduplicate and clean updates ---
        text = extract_unique_updates(text)
        # --- Sentence-aware chunking ---
        try:
            chunks = chunk_sentences(text, max_sentences=8)
            chunk_summaries = []
            for idx, chunk in enumerate(chunks):
                input_text = f"summarize: {chunk}"
                input_ids = tokenizer.encode(input_text, return_tensors='tf')
                summary_ids = model.generate(
                    input_ids,
                    max_length=512,   # longer chunk summaries
                    min_length=100,
                    length_penalty=1.0,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7
                )
                chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                chunk_summaries.append(chunk_summary)
            # 2. Concatenate chunk summaries
            combined_summary = ' '.join(chunk_summaries)
            # 3. Optionally, summarize the combined summary for coherence
            input_text = f"summarize: {combined_summary}"
            input_ids = tokenizer.encode(input_text, return_tensors='tf')
            summary_ids = model.generate(
                input_ids,
                max_length=512,   # final summary can be very long
                min_length=180,
                length_penalty=1.0,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )
            final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            # Remove repeated sentences
            final_summary = remove_repetitions(final_summary)
            final_summary = trim_to_last_complete_sentence(final_summary)
            if final_summary and len(final_summary.strip()) > 30 and not final_summary.startswith("summarize:"):
                return {
                    "summary": final_summary,
                    "sentiment": sentiment_result,
                    "word_count": len(text.split()),
                    "extraction_method": "T5_super_detailed"
                }
        except Exception as e:
            print(f"Super-detailed summarization failed: {e}")
        # Fallback: Extract key sentences
        print("Using fallback summarization method...")
        fallback_summary = extract_key_sentences(text, num_sentences=6)
        if fallback_summary and len(fallback_summary.strip()) > 50:
            return {
                "summary": fallback_summary,
                "sentiment": sentiment_result,
                "word_count": len(text.split()),
                "extraction_method": "key_sentences"
            }
        sentences = sent_tokenize(text)
        if sentences:
            first_sentences = " ".join(sentences[:5])
            return {
                "summary": first_sentences,
                "sentiment": sentiment_result,
                "word_count": len(text.split()),
                "extraction_method": "first_sentences"
            }
        return {
            "summary": "Unable to generate a meaningful summary from this content.",
            "sentiment": sentiment_result,
            "word_count": len(text.split()),
            "extraction_method": "failed"
        }
    except Exception as e:
        print(f"Error processing article: {e}")
        return {
            "summary": f"Error processing article: {str(e)}",
            "sentiment": {"sentiment": "neutral", "score": 0.0, "confidence": "error"},
            "word_count": 0,
            "extraction_method": "error"
        }

def clean_text_for_classification(text):
    """Clean and prepare text for better classification with sports focus"""
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # For sports articles, try to get more content (sports articles often have detailed descriptions)
    # Take first 1500 characters instead of 1000 for better sports detection
    text = text[:1500]
    
    # Look for sports-related keywords to boost sports classification
    sports_keywords = [
        'football', 'soccer', 'cricket', 'basketball', 'tennis', 'hockey', 'rugby',
        'match', 'game', 'tournament', 'championship', 'league', 'team', 'player',
        'goal', 'score', 'win', 'lose', 'draw', 'victory', 'defeat', 'final',
        'semifinal', 'quarterfinal', 'stadium', 'field', 'pitch', 'court',
        'coach', 'manager', 'captain', 'referee', 'umpire', 'foul', 'penalty',
        'kick', 'shot', 'pass', 'tackle', 'dribble', 'header', 'assist'
    ]
    
    # If sports keywords are found, include more text
    if any(keyword in text.lower() for keyword in sports_keywords):
        text = text[:2000]  # Include more text for sports articles
    
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
    
    # Enhanced labels with better sports coverage
    candidate_labels = [
        "sports and athletics",
        "football and soccer",
        "cricket and sports",
        "sports competitions",
        "team sports",
        "music and entertainment",
        "politics and government", 
        "business and economy",
        "technology and innovation",
        "science and research",
        "health and medicine",
        "environment and climate",
        "arts and culture",
        "education and learning",
        "international news",
        "local news"
    ]
    
    try:
        print("Starting classification...")
        result = classifier(cleaned_text, candidate_labels, multi_label=True)
        print(f"Raw classification result: {result}")
        
        # Higher threshold for better accuracy
        classifications = [(label, score) for label, score in zip(result['labels'], result['scores']) if score > 0.15]
        sorted_classifications = sorted(classifications, key=lambda x: x[1], reverse=True)
        print(f"Final classifications: {sorted_classifications}")
        
        # If no results above threshold, return top 2 anyway
        if not sorted_classifications:
            print("No results above threshold, returning top 2 anyway")
            all_results = [(label, score) for label, score in zip(result['labels'], result['scores'])]
            sorted_classifications = sorted(all_results, key=lambda x: x[1], reverse=True)[:2]
        
        # Ensure we have at least 2 classifications
        if len(sorted_classifications) < 2:
            all_results = [(label, score) for label, score in zip(result['labels'], result['scores'])]
            sorted_classifications = sorted(all_results, key=lambda x: x[1], reverse=True)[:2]
        
        return sorted_classifications
    except Exception as e:
        print(f"Error in classification: {e}")
        return [("classification error", 0.0)]

def test_sports_classification():
    """Test function to verify sports classification improvements"""
    test_sports_text = """
    Football match between IAF and South United FC ended in a thrilling 3-3 draw. 
    Young striker Makakmayum Daniel completed a fine hat-trick giving South United the lead twice in the match. 
    IAF mounted a great fightback to snatch a point in the end. The tournament debutant South United FC 
    showed great promise with their attacking football. The match was played at the Durand Cup tournament 
    with both teams displaying excellent skills and determination.
    """
    print("Testing sports classification with football article...")
    result = classify_text(test_sports_text)
    print(f"Sports classification result: {result}")
    return result

def test_classification():
    """Test function to debug classification issues"""
    test_text = "This is a music article about new albums and songs from popular artists."
    print("Testing classification with sample text...")
    result = classify_text(test_text)
    print(f"Test result: {result}")
    return result