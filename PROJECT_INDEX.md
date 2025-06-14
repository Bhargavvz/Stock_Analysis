# Stock Price Prediction Model - Project Index

## 📁 Project Structure Overview

```
a_stock_price_prediction_model/
├── 📄 manage.py                              # Django management script
├── 📄 requirements.txt                       # Python dependencies
├── 📄 README.md                             # Project documentation
├── 📄 .gitignore                            # Git ignore rules
├── 📄 PROJECT_INDEX.md                      # This index file
├── 📊 Datasets.csv                          # Training dataset (14,069 records)
├── 📊 Results.csv                           # Model results and predictions
├── 🗄️ a_stock_price_prediction_model.sql    # Database schema and data
│
├── 🎯 a_stock_price_prediction_model/       # Django project settings
│   ├── __init__.py
│   ├── asgi.py                              # ASGI configuration
│   ├── settings.py                          # Django settings
│   ├── urls.py                              # URL routing
│   ├── wsgi.py                              # WSGI configuration
│   └── __pycache__/                         # Python bytecode cache
│
├── 👥 Remote_User/                          # User management app
│   ├── __init__.py
│   ├── admin.py                             # Django admin configuration
│   ├── apps.py                              # App configuration
│   ├── forms.py                             # Django forms
│   ├── models.py                            # Database models
│   ├── tests.py                             # Unit tests
│   ├── views.py                             # Business logic and ML algorithms
│   ├── migrations/                          # Database migrations
│   │   ├── __init__.py
│   │   ├── 0001_initial.py
│   │   ├── 0002_clientposts_model.py
│   │   ├── 0003_clientposts_model_usefulcounts.py
│   │   ├── 0004_auto_20190429_1027.py
│   │   ├── 0005_clientposts_model_dislikes.py
│   │   ├── 0006_review_model.py
│   │   ├── 0007_clientposts_model_names.py
│   │   └── __pycache__/
│   └── __pycache__/                         # Python bytecode cache
│
├── 🔧 Service_Provider/                     # Service provider app
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── views.py
│   ├── migrations/
│   └── __pycache__/
│
├── 🎨 Template/                             # Frontend templates
│   ├── htmls/                               # HTML templates
│   │   ├── custom.css                       # Custom styles
│   │   ├── images/                          # Template images
│   │   ├── media/                           # Media files
│   │   ├── RUser/                           # Remote user templates
│   │   └── SProvider/                       # Service provider templates
│   └── images/                              # Static images
│       ├── Banner.jpg
│       ├── bg.jpg
│       └── ...
│
└── 📊 static/                               # Static files
    └── images/
        └── image.jpg
```

## 📊 Database Tables

### Core Tables (from SQL schema):
- `auth_user` - Django user authentication
- `auth_group` - User groups and permissions
- `auth_permission` - Permission system
- `django_content_type` - Content type framework
- `django_session` - Session management

### Application-Specific Tables:
- `remote_user_clientregister_model` - User registration data
- `remote_user_predict_investor_sentiment` - Sentiment predictions
- `remote_user_detection_accuracy` - Model accuracy metrics
- `remote_user_detection_ratio` - Prediction ratios

## 🤖 Machine Learning Models

### Implemented Algorithms:
1. **Deep Neural Network (DNN)** - 91.82% accuracy
2. **Support Vector Machine (SVM)** - 89.69% accuracy
3. **Logistic Regression** - 92.31% accuracy
4. **Decision Tree Classifier** - 91.00% accuracy
5. **K-Neighbors Classifier** - 93.29% accuracy
6. **Gradient Boosting Classifier** - 92.14% accuracy

### Data Processing:
- **Text Preprocessing**: NLTK-based text cleaning and tokenization
- **Feature Extraction**: TF-IDF vectorization
- **Deep Learning**: TensorFlow/Keras LSTM models
- **Ensemble Methods**: Voting classifier for improved accuracy

## 📈 Dataset Information

### Training Data (Datasets.csv):
- **Size**: 14,069 records
- **Features**:
  - `Investor_Age`: Age of the investor
  - `Investor_Gender`: Gender classification
  - `PDate`: Prediction timestamp
  - `Stock_Text`: Social media text about stocks
  - `Stock_Name`: Stock ticker symbol (e.g., TSLA)
  - `Company_Name`: Full company name
  - `Label`: Sentiment classification (0/1)

### Stock Coverage:
- Primary focus on Tesla (TSLA) stock sentiment
- Social media text analysis for sentiment prediction
- Timestamp-based prediction tracking

## 🔧 Technology Stack

### Backend:
- **Django 3.2.7** - Web framework
- **Python 3.7+** - Programming language
- **SQLite/MySQL** - Database systems

### Machine Learning:
- **scikit-learn 1.0** - ML algorithms
- **TensorFlow 2.8.0** - Deep learning
- **NLTK 3.6.5** - Natural language processing
- **pandas 1.3.3** - Data manipulation
- **NumPy 1.21.2** - Numerical computing

### Frontend:
- **HTML/CSS** - Web interface
- **Bootstrap** - Responsive design
- **JavaScript** - Interactive elements

## 🎯 Key Features

### User Management:
- User registration and authentication
- Profile management
- Session handling

### Prediction System:
- Real-time sentiment analysis
- Multiple ML model integration
- Accuracy tracking and reporting
- Historical prediction storage

### Analytics:
- Model performance comparison
- Trend analysis (Uptrends: 75%, Downtrends: 25%)
- Visualization of results

## 🚀 Getting Started

### Prerequisites:
```bash
# Install Python 3.7+
# Install pip package manager
```

### Installation:
```bash
# Clone the repository
git clone [repository-url]

# Navigate to project directory
cd a_stock_price_prediction_model

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start development server
python manage.py runserver
```

### Database Setup:
- Import `a_stock_price_prediction_model.sql` for complete schema
- Ensure proper MySQL/SQLite configuration in `settings.py`

## 📝 File Descriptions

### Core Django Files:
- `manage.py`: Django's command-line utility
- `settings.py`: Configuration settings
- `urls.py`: URL routing patterns
- `wsgi.py`: WSGI server configuration

### Models (`Remote_User/models.py`):
- `ClientRegister_Model`: User registration
- `predict_investor_sentiment`: Prediction storage
- `detection_accuracy`: Model accuracy tracking
- `detection_ratio`: Prediction ratio metrics

### Views (`Remote_User/views.py`):
- User authentication and registration
- ML model loading and prediction
- Data preprocessing and analysis
- Results visualization

### Templates:
- User interface components
- Form handling
- Results display
- Responsive design elements

## 🔍 Development Notes

### Model Performance:
- Best performing: K-Neighbors Classifier (93.29%)
- Ensemble approach for robust predictions
- Real-time prediction capabilities

### Data Pipeline:
1. Data ingestion from CSV
2. Text preprocessing and cleaning
3. Feature extraction (TF-IDF)
4. Model training and validation
5. Prediction and storage
6. Results visualization

### Future Enhancements:
- Additional stock symbols
- Real-time data feeds
- Advanced visualization
- Mobile responsiveness
- API development
