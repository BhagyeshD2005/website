import os
import re
import tempfile
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from google import genai
from dotenv import load_dotenv
from groq import Groq
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Data class for search results."""
    title: str
    link: str
    snippet: str
    display_link: str


@dataclass
class APIResponse:
    """Standard API response format."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None


# Database Models
db = SQLAlchemy()
login_manager = LoginManager()


class User(UserMixin, db.Model):
    """User model for authentication."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Usage tracking
    chat_requests = db.Column(db.Integer, default=0)
    document_analyses = db.Column(db.Integer, default=0)
    search_requests = db.Column(db.Integer, default=0)
    
    def set_password(self, password):
        """Set password hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash."""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp."""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def increment_usage(self, usage_type):
        """Increment usage counter."""
        if usage_type == 'chat':
            self.chat_requests += 1
        elif usage_type == 'document':
            self.document_analyses += 1
        elif usage_type == 'search':
            self.search_requests += 1
        db.session.commit()


class UsageLog(db.Model):
    """Usage log model for tracking user activity."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(50), nullable=False)  # 'chat', 'document', 'search'
    details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
    
    user = db.relationship('User', backref=db.backref('usage_logs', lazy=True))


@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login."""
    return User.query.get(int(user_id))


def admin_required(f):
    """Decorator to require admin access."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def log_user_activity(action, details=None):
    """Log user activity."""
    if current_user.is_authenticated:
        log_entry = UsageLog(
            user_id=current_user.id,
            action=action,
            details=details,
            ip_address=request.remote_addr
        )
        db.session.add(log_entry)
        db.session.commit()
        current_user.increment_usage(action)


class ConfigManager:
    """Manages application configuration and API keys."""
    
    def __init__(self):
        # Use environment variable or generate a random secret key
        self.secret_key = os.getenv('SECRET_KEY') or os.urandom(24).hex()
        
        # Use PostgreSQL for production, SQLite for development
        self.database_url = os.getenv('DATABASE_URL')
        if self.database_url and self.database_url.startswith('postgres://'):
            # Fix for newer SQLAlchemy versions
            self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)
        elif not self.database_url:
            # Fallback to SQLite for local development
            self.database_url = 'sqlite:///legal_assistant.db'
        
        # AI API Keys
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.google_ai_api_key = os.getenv('GOOGLE_AI_API_KEY')
        
        # Search API Keys
        self.google_search_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        
        # Port for deployment
        self.port = int(os.getenv('PORT', 5000))
        
    def validate_keys(self) -> Dict[str, List[str]]:
        """Validate API keys and return categorized missing keys."""
        missing_keys = {
            'ai_services': [],
            'search_services': [],
            'general': []
        }
        
        # AI Services
        if not self.groq_api_key:
            missing_keys['ai_services'].append('GROQ_API_KEY')
        if not self.google_ai_api_key:
            missing_keys['ai_services'].append('GOOGLE_AI_API_KEY')
            
        # Search Services
        if not self.google_search_api_key:
            missing_keys['search_services'].append('GOOGLE_SEARCH_API_KEY')
        if not self.google_cse_id:
            missing_keys['search_services'].append('GOOGLE_CSE_ID')
            
        return missing_keys


class AIClientManager:
    """Manages AI client initialization and operations."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.groq_client = None
        self.google_client = None
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize AI clients with error handling."""
        # Initialize Groq client
        if self.config.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.config.groq_api_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("GROQ_API_KEY not found - chat functionality will be limited")
        
        # Initialize Google Generative AI client
        if self.config.google_ai_api_key:
            try:
                self.google_client = genai.Client(api_key=self.config.google_ai_api_key)
                logger.info("Google AI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google AI client: {e}")
        else:
            logger.warning("GOOGLE_AI_API_KEY not found - document analysis will be unavailable")
    
    def is_groq_available(self) -> bool:
        """Check if Groq client is available."""
        return self.groq_client is not None
    
    def is_google_available(self) -> bool:
        """Check if Google AI client is available."""
        return self.google_client is not None


class LegalContextManager:
    """Manages legal context templates for different domains."""
    
    CONTEXTS = {
        "general": (
            "You are a legal assistant. Provide structured output in the following format:\n\n"
            "Summary:\n\nRelevant Laws:\n\nLegal Advice:\n\n"
            "Risks or Penalties:\n\nNotable Judgments:"
        ),
        "case_law": (
            "You are a legal researcher. Provide structured output like this:\n\n"
            "Summary:\n\nNotable Judgments:\n\nInterpretation:\n\nImplications:"
        ),
        "contract": (
            "You are a contract law expert. Format the answer as:\n\n"
            "Summary:\n\nEssential Clauses:\n\nRelevant Laws:\n\nPrecautions:"
        ),
        "compliance": (
            "You are a compliance expert. Format as:\n\n"
            "Overview:\n\nApplicable Laws:\n\nCompliance Steps:\n\n"
            "Penalties for Non-Compliance:"
        ),
        "startup": (
            "You help startups with legal issues. Provide output as:\n\n"
            "Overview:\n\nLegal Requirements:\n\nIP & Compliance Tips:\n\n"
            "Common Mistakes:"
        ),
        "consumer": (
            "You are a consumer protection expert. Structure your output as:\n\n"
            "Problem Overview:\n\nRelevant Law:\n\nLegal Remedies:\n\n"
            "Important Judgments:"
        ),
        "criminal": (
            "You are a criminal lawyer. Provide structured legal analysis:\n\n"
            "Offense Summary:\n\nIPC/CrPC Sections:\n\nRelevant Judgments:\n\n"
            "Legal Opinion:"
        )
    }
    
    @classmethod
    def get_context(cls, context_key: str) -> str:
        """Get legal context template by key."""
        return cls.CONTEXTS.get(context_key, cls.CONTEXTS["general"])
    
    @classmethod
    def get_available_contexts(cls) -> List[str]:
        """Get list of available context keys."""
        return list(cls.CONTEXTS.keys())


class ResponseFormatter:
    """Handles formatting of AI responses for display."""
    
    SECTION_HEADERS = [
        "Summary:", "Relevant Laws:", "Legal Advice:", "Risks or Penalties:", 
        "Notable Judgments:", "Interpretation:", "Implications:", "Essential Clauses:", 
        "Precautions:", "Overview:", "Applicable Laws:", "Compliance Steps:", 
        "Penalties for Non-Compliance:", "Legal Requirements:", "IP & Compliance Tips:", 
        "Common Mistakes:", "Problem Overview:", "Relevant Law:", "Legal Remedies:", 
        "Important Judgments:", "Offense Summary:", "IPC/CrPC Sections:", "Legal Opinion:"
    ]
    
    @classmethod
    def format_response(cls, text: str) -> List[Tuple[str, str]]:
        """Format AI response into structured sections."""
        # Convert markdown bold to HTML
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = text.replace('*', '')
        
        # Add newlines before headers for reliable splitting
        for header in cls.SECTION_HEADERS:
            text = text.replace(header, f"\n{header}")
        
        # Parse sections
        formatted_blocks = []
        current_title = ""
        current_body = []
        
        for line in text.splitlines():
            line = line.strip()
            if any(line.startswith(h) for h in cls.SECTION_HEADERS):
                if current_title:
                    formatted_blocks.append((current_title, "<br>".join(current_body)))
                
                parts = line.split(":", 1)
                current_title = parts[0]
                current_body = [parts[1].strip()] if len(parts) > 1 else []
            elif line:
                current_body.append(line)
        
        if current_title:
            formatted_blocks.append((current_title, "<br>".join(current_body)))
        
        return formatted_blocks
    
    @classmethod
    def clean_section_response(cls, text: str) -> str:
        """Clean response text by removing headers and formatting."""
        # Remove headers and structured formatting
        text = re.sub(r'^[A-Z][a-z\s]*:', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*\*[A-Z][a-z\s]*:\*\*', '', text, flags=re.MULTILINE)
        
        # Format for HTML display
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = text.replace('\n', '<br>')
        
        return text.strip()


class WebSearchService:
    """Handles web search operations using Google Custom Search."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def is_available(self) -> bool:
        """Check if search service is properly configured."""
        return bool(self.config.google_search_api_key and self.config.google_cse_id)
    
    def search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Perform Google Custom Search."""
        if not self.is_available():
            return {
                'success': False, 
                'error': 'Search service not configured. Please set GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID.'
            }
            
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.config.google_search_api_key,
                'cx': self.config.google_cse_id,
                'q': query,
                'num': min(num_results, 10)
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                result = SearchResult(
                    title=item.get('title', ''),
                    link=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    display_link=item.get('displayLink', '')
                )
                results.append(result.__dict__)
            
            logger.info(f"Google Search completed: {len(results)} results for query '{query}'")
            
            return {
                'success': True,
                'results': results,
                'total_results': data.get('searchInformation', {}).get('totalResults', 0)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Search API error: {e}")
            return {'success': False, 'error': f'Search API error: {str(e)}'}
        except Exception as e:
            logger.error(f"Unexpected error in Google search: {e}")
            return {'success': False, 'error': f'Search error: {str(e)}'}


class DocumentAnalysisService:
    """Handles document analysis using Google AI."""
    
    def __init__(self, ai_manager: AIClientManager):
        self.ai_manager = ai_manager
    
    def analyze_pdf(self, file_path: str, filename: str) -> APIResponse:
        """Analyze a PDF document."""
        if not self.ai_manager.is_google_available():
            return APIResponse(
                success=False, 
                error='Google AI client not initialized. Please set GOOGLE_AI_API_KEY.'
            )
        
        try:
            # Upload file to Google AI
            uploaded_file = self.ai_manager.google_client.files.upload(file=file_path)
            logger.info(f"File uploaded to Google AI: {filename}")
            
            # Generate analysis
            response = self.ai_manager.google_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=["analyze", uploaded_file]
            )
            
            logger.info(f"Document analysis completed for: {filename}")
            
            return APIResponse(
                success=True,
                data={
                    'analysis': response.text,
                    'filename': filename
                }
            )
            
        except Exception as e:
            logger.error(f"Document analysis error for {filename}: {e}")
            return APIResponse(success=False, error=f'Analysis error: {str(e)}')


class ChatService:
    """Handles chat interactions with AI models."""
    
    def __init__(self, ai_manager: AIClientManager):
        self.ai_manager = ai_manager
    
    def get_legal_response(self, query: str, context_key: str) -> APIResponse:
        """Get legal response using Groq."""
        if not self.ai_manager.is_groq_available():
            return APIResponse(
                success=False, 
                error='Groq client not initialized. Please set GROQ_API_KEY.'
            )
        
        try:
            context_instruction = LegalContextManager.get_context(context_key)
            prompt = f"{context_instruction}\n\nQuery: {query}"
            
            chat_completion = self.ai_manager.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                stream=False,
            )
            
            response = chat_completion.choices[0].message.content
            formatted_output = ResponseFormatter.format_response(response)
            
            logger.info(f"Legal chat response generated for context: {context_key}")
            
            return APIResponse(
                success=True,
                data={
                    'formatted': formatted_output,
                    'original_query': query,
                    'function': context_key
                }
            )
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return APIResponse(success=False, error=f'Chat error: {str(e)}')
    
    def get_section_response(self, section_data: Dict[str, str]) -> APIResponse:
        """Get focused response for a specific section."""
        if not self.ai_manager.is_groq_available():
            return APIResponse(
                success=False, 
                error='Groq client not initialized. Please set GROQ_API_KEY.'
            )
        
        try:
            prompt = f"""
You are a legal expert providing focused answers about specific legal topics.

Context: The user originally asked about "{section_data.get('original_query')}" 
in the {section_data.get('function_key')} legal domain.

Section Topic: {section_data.get('section_title')}
Section Content: {section_data.get('section_content')}

User's Question: {section_data.get('user_question')}

Instructions:
- Provide a direct, focused answer to the user's question
- Base your response only on the section content and your legal knowledge
- Do not include any headers, structured formatting, or section titles
- Write in a conversational, explanatory tone
- Focus on practical, actionable information
- Keep the response concise but comprehensive

Answer:
"""
            
            chat_completion = self.ai_manager.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                stream=False,
            )
            
            ai_response = chat_completion.choices[0].message.content
            formatted_response = ResponseFormatter.clean_section_response(ai_response)
            
            return APIResponse(success=True, data={'response': formatted_response})
            
        except Exception as e:
            logger.error(f"Section query error: {e}")
            return APIResponse(success=False, error=f'Section query error: {str(e)}')


class ResearchService:
    """Combines web search with AI analysis for comprehensive research."""
    
    def __init__(self, ai_manager: AIClientManager, search_service: WebSearchService):
        self.ai_manager = ai_manager
        self.search_service = search_service
    
    def analyze_search_results(self, search_results: List[Dict], question: str) -> APIResponse:
        """Analyze search results using AI."""
        if not self.ai_manager.is_groq_available():
            return APIResponse(
                success=False, 
                error='Groq API not configured. Please set GROQ_API_KEY.'
            )
        
        try:
            # Prepare context from search results
            context = f"Research Question: {question}\n\nSearch Results:\n"
            
            for i, result in enumerate(search_results[:5], 1):
                context += f"{i}. Title: {result['title']}\n"
                context += f"   URL: {result['link']}\n"
                context += f"   Summary: {result['snippet']}\n\n"
            
            prompt = f"""Based on the following search results, provide a comprehensive analysis.

{context}

Please provide:
1. A summary of key findings
2. Different perspectives or viewpoints found
3. Credible sources and their reliability
4. Gaps in information or areas needing further research
5. A conclusion with actionable insights

Format your response in a clear, structured manner."""
            
            chat_completion = self.ai_manager.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=2000,
            )
            
            analysis = chat_completion.choices[0].message.content
            
            logger.info(f"Search results analysis completed for: {question}")
            
            return APIResponse(
                success=True,
                data={
                    'analysis': analysis,
                    'model_used': 'llama-3.1-8b-instant'
                }
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return APIResponse(success=False, error=f'Analysis error: {str(e)}')
    
    def deep_research(self, query: str, num_results: int = 10) -> APIResponse:
        """Perform comprehensive research combining search and analysis."""
        # Step 1: Search
        search_result = self.search_service.search(query, num_results)
        
        if not search_result['success']:
            return APIResponse(success=False, error=search_result['error'])
        
        # Step 2: Analyze
        analysis_result = self.analyze_search_results(search_result['results'], query)
        
        return APIResponse(
            success=True,
            data={
                'search_results': search_result['results'],
                'total_results': search_result['total_results'],
                'analysis': analysis_result.data.get('analysis') if analysis_result.success else 'Analysis failed',
                'analysis_success': analysis_result.success,
                'model_used': analysis_result.data.get('model_used') if analysis_result.success else 'N/A'
            }
        )


def create_app() -> Flask:
    """Application factory function."""
    app = Flask(__name__)
    
    # Initialize configuration and services
    config = ConfigManager()
    app.secret_key = config.secret_key
    app.config['SQLALCHEMY_DATABASE_URI'] = config.database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access this page.'
    
    # Check for missing API keys
    missing_keys = config.validate_keys()
    
    # Log missing keys by category
    for category, keys in missing_keys.items():
        if keys:
            logger.warning(f"Missing {category} API keys: {', '.join(keys)}")
    
    # Initialize services
    ai_manager = AIClientManager(config)
    search_service = WebSearchService(config)
    doc_service = DocumentAnalysisService(ai_manager)
    chat_service = ChatService(ai_manager)
    research_service = ResearchService(ai_manager, search_service)
    
    # Create database tables and default admin user
    with app.app_context():
        try:
            db.create_all()
            
            # Create default admin user if none exists
            if not User.query.filter_by(is_admin=True).first():
                admin_user = User(
                    username='admin',
                    email='admin@legalassistant.com',
                    is_admin=True,
                    is_active=True
                )
                admin_user.set_password('admin123')  # Change this in production!
                db.session.add(admin_user)
                db.session.commit()
                logger.info("Default admin user created - username: admin, password: admin123")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    # Authentication Routes
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """User login."""
        if current_user.is_authenticated:
            return redirect(url_for('home'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                flash('Please enter both username and password.', 'error')
                return render_template('login.html')
            
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password) and user.is_active:
                login_user(user)
                user.update_last_login()
                flash(f'Welcome back, {user.username}!', 'success')
                
                # Redirect admin to admin panel
                if user.is_admin:
                    return redirect(url_for('admin_dashboard'))
                
                # Redirect regular users to home
                next_page = request.args.get('next')
                return redirect(next_page if next_page else url_for('home'))
            else:
                flash('Invalid username or password, or account is disabled.', 'error')
        
        return render_template('login.html')
    
    @app.route('/logout')
    @login_required
    def logout():
        """User logout."""
        logout_user()
        flash('You have been logged out successfully.', 'info')
        return redirect(url_for('login'))
    
    # Admin Routes
    @app.route('/admin')
    @login_required
    @admin_required
    def admin_dashboard():
        """Admin dashboard."""
        users = User.query.all()
        total_users = len(users)
        active_users = len([u for u in users if u.is_active])
        
        # Get recent activity
        recent_logs = UsageLog.query.order_by(UsageLog.timestamp.desc()).limit(10).all()
        
        # Usage statistics
        total_chats = sum(u.chat_requests for u in users)
        total_documents = sum(u.document_analyses for u in users)
        total_searches = sum(u.search_requests for u in users)
        
        return render_template('admin_dashboard.html',
                             users=users,
                             total_users=total_users,
                             active_users=active_users,
                             recent_logs=recent_logs,
                             total_chats=total_chats,
                             total_documents=total_documents,
                             total_searches=total_searches)
    
    @app.route('/admin/create_user', methods=['GET', 'POST'])
    @login_required
    @admin_required
    def create_user():
        """Create new user."""
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            is_admin = request.form.get('is_admin') == 'on'
            
            # Validation
            if not all([username, email, password]):
                flash('All fields are required.', 'error')
