from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.views.generic import CreateView
from django.urls import reverse_lazy
from django.contrib.auth.mixins import LoginRequiredMixin
from .utils import get_summary_and_bias, classify_text, fetch_article_text
from .forms import UserRegistrationForm, UserLoginForm, UserProfileForm
from .models import User, UserActivity

@login_required
def index(request):
    if request.method == 'POST':
        print("POST request received!")
        print("POST data:", request.POST)
        
        # ✅ Correct field name
        article_url = request.POST.get('article_url')
        print("Received URL:", article_url)
        
        if article_url:
            try:
                # Get enhanced summary with sentiment analysis
                analysis_result = get_summary_and_bias(article_url)
                
                if analysis_result and analysis_result.get('summary'):
                    summary = analysis_result['summary']
                    sentiment = analysis_result.get('sentiment', {})
                    word_count = analysis_result.get('word_count', 0)
                    extraction_method = analysis_result.get('extraction_method', 'unknown')
                    
                    messages.success(request, f"Analysis completed successfully using {extraction_method} method.")
                else:
                    summary = "Error: Could not generate summary"
                    sentiment = {"sentiment": "neutral", "score": 0.0, "confidence": "low"}
                    word_count = 0
                    extraction_method = "failed"
                    messages.error(request, "Summary generation failed")
                
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
                
                # Track user activity with enhanced details
                UserActivity.objects.create(
                    user=request.user,
                    activity_type='article_analysis',
                    details={
                        'url': article_url,
                        'summary_length': len(summary) if summary else 0,
                        'classification_count': len(classification) if classification else 0,
                        'word_count': word_count,
                        'extraction_method': extraction_method,
                        'sentiment': sentiment.get('sentiment', 'neutral'),
                        'sentiment_score': sentiment.get('score', 0.0)
                    }
                )
                
                # Increment user's article count
                request.user.increment_articles_analyzed()
                    
                # Store results in session and redirect
                request.session['analysis_results'] = {
                    'summary': summary,
                    'classification': classification,
                    'url': article_url,
                    'sentiment': sentiment,
                    'word_count': word_count,
                    'extraction_method': extraction_method
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
        'sentiment': analysis_results.get('sentiment', {}),
        'word_count': analysis_results.get('word_count', 0),
        'extraction_method': analysis_results.get('extraction_method', ''),
        'debug_info': [],  # No debug info for GET requests
        'user': request.user
    })

def register_view(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            
            # Log registration activity
            UserActivity.objects.create(
                user=user,
                activity_type='registration',
                details={'registration_method': 'web_form'}
            )
            
            messages.success(request, f'Welcome {user.first_name}! Your account has been created successfully.')
            return redirect('index')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'core/register.html', {'form': form})

def login_view(request):
    """User login view"""
    if request.method == 'POST':
        form = UserLoginForm(request, data=request.POST)
        if form.is_valid():
            email = form.cleaned_data.get('email')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user)
                
                # Log login activity
                UserActivity.objects.create(
                    user=user,
                    activity_type='login',
                    details={'login_method': 'web_form'}
                )
                
                messages.success(request, f'Welcome back, {user.first_name}!')
                return redirect('index')
    else:
        form = UserLoginForm()
    
    return render(request, 'core/login.html', {'form': form})

def logout_view(request):
    """User logout view"""
    if request.user.is_authenticated:
        # Log logout activity
        UserActivity.objects.create(
            user=request.user,
            activity_type='logout',
            details={'logout_method': 'web_form'}
        )
    
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('index')

@login_required
def profile_view(request):
    """User profile view"""
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated successfully.')
            return redirect('profile')
    else:
        form = UserProfileForm(instance=request.user)
    
    # Get user activity history
    activities = UserActivity.objects.filter(user=request.user)[:10]
    
    return render(request, 'core/profile.html', {
        'form': form,
        'activities': activities,
        'user': request.user
    })

@login_required
def dashboard_view(request):
    """User dashboard with analytics"""
    user = request.user
    
    # Get user statistics
    total_activities = UserActivity.objects.filter(user=user).count()
    recent_activities = UserActivity.objects.filter(user=user)[:5]
    
    context = {
        'user': user,
        'total_activities': total_activities,
        'recent_activities': recent_activities,
        'articles_analyzed': user.articles_analyzed,
        'last_analysis': user.last_analysis_date,
    }
    
    return render(request, 'core/dashboard.html', context)
