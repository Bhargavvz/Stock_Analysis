# Stock Price Prediction Model with Enhanced Accuracy

This project uses advanced machine learning techniques to predict stock price trends based on investor sentiment analysis.

## Improvements Made

1. **Enhanced Text Preprocessing**
   - Added lemmatization to reduce words to their base forms
   - Implemented stopword removal to eliminate noise
   - Applied regex patterns to clean text (URLs, symbols, etc.)

2. **Advanced Feature Engineering**
   - Added VADER sentiment analysis features (positive, negative, neutral, compound)
   - Replaced simple bag-of-words with TF-IDF vectorization
   - Included n-grams (1-3) to capture phrases

3. **LSTM Neural Network Implementation**
   - Added a deep learning model for sequence modeling
   - Captures temporal patterns in investor sentiment
   - Employs dropout and regularization to prevent overfitting

4. **Improved Machine Learning Models**
   - Optimized hyperparameters for all models
   - Added class balancing to handle imbalanced data
   - Implemented Random Forest classifier
   - Improved MLP configuration

5. **Weighted Ensemble Model**
   - Created a weighted voting classifier
   - Weights models based on their accuracy
   - Dynamic selection between LSTM and ensemble based on confidence

## How to Use

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the models (admin access required):
   - Log in as service provider (Admin/Admin)
   - Navigate to "Train Model" section
   - The system will train all models and save them for future use

3. Make predictions:
   - Log in as a user
   - Navigate to "Predict Sentiment" section
   - Enter stock-related text to get trend prediction (Uptrends/Downtrends)

## Model Performance

The improved model achieves higher accuracy through:
- Better text processing
- Advanced feature engineering
- Deep learning with LSTM
- Weighted ensemble approach

This leads to more reliable stock trend predictions based on investor sentiment analysis. 