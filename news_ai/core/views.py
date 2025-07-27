from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from .utils import get_summary_and_bias, classify_text, fetch_article_text

def index(request):
    if request.method == 'POST':
        print("POST request received!")
        print("POST data:", request.POST)
        
        # ✅ Correct field name
        article_url = request.POST.get('article_url')
        print("Received URL:", article_url)
        
        if article_url:
            try:
                # Get summary
                summary = get_summary_and_bias(article_url)
                if summary is None:
                    summary = "Error: Could not generate summary"
                    messages.error(request, "Summary generation failed")
                else:
                    messages.success(request, "Analysis completed successfully")
                
                # Get article text and classify it
                print("Attempting to fetch article text...")
                article_text = fetch_article_text(article_url)
                print(f"Article text length: {len(article_text) if article_text else 0}")
                
                if article_text and len(article_text.strip()) > 50:
                    print("Attempting classification...")
                    classification = classify_text(article_text)
                    print("Classification result:", classification)
                else:
                    classification = [("Error: Could not extract article text", 0.0)]
                    messages.warning(request, "Article text extraction failed")
                    
                # Store results in session and redirect
                request.session['analysis_results'] = {
                    'summary': summary,
                    'classification': classification,
                    'url': article_url
                }
                
                return redirect('index')
                    
            except Exception as e:
                print(f"Error processing article: {e}")
                messages.error(request, f"Error: {str(e)}")
                return redirect('index')
        else:
            print("No URL found in POST data")
            messages.error(request, "No URL provided")
            return redirect('index')

    # GET request - display results from session if available
    analysis_results = request.session.get('analysis_results', {})
    
    return render(request, 'core/index.html', {
        'summary': analysis_results.get('summary', ''),
        'classification': analysis_results.get('classification', []),
        'last_url': analysis_results.get('url', ''),
        'debug_info': []  # No debug info for GET requests
    })
