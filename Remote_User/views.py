from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,predict_investor_sentiment,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Investor_Sentiment_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            Investor_Age= request.POST.get('Investor_Age')
            Investor_Gender= request.POST.get('Investor_Gender')
            PDate= request.POST.get('PDate')
            Stock_Text= request.POST.get('Stock_Text')
            Stock_Name= request.POST.get('Stock_Name')
            Company_Name= request.POST.get('Company_Name')

        # Load trained models and preprocessing tools
        import pickle
        import os
        
        # Initialize prediction components
        try:
            # Try to load the advanced models
            with open('classifier_models.pkl', 'rb') as f:
                models_data = pickle.load(f)
                
            tfidf = models_data['tfidf']
            classifier = models_data['classifier']
            lemmatizer = models_data['lemmatizer']
            stop_words = models_data['stop_words']
            
            # Check if LSTM model exists
            use_lstm = os.path.exists('lstm_model.pkl')
            if use_lstm:
                with open('lstm_model.pkl', 'rb') as f:
                    lstm_data = pickle.load(f)
                tokenizer = lstm_data['tokenizer']
                max_seq_length = lstm_data['max_seq_length']
                lstm_model = lstm_data['model']
            
            # Text preprocessing function (same as in training)
            import re
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            
            # Ensure NLTK data is downloaded
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon')
            
            sid = SentimentIntensityAnalyzer()
            
            def preprocess_text(text):
                # Convert to lowercase
                text = str(text).lower()
                # Remove URLs
                text = re.sub(r'http\S+|www\S+|https\S+', '', text)
                # Remove user @ references and '#' from tweet
                text = re.sub(r'\@\w+|\#', '', text)
                # Remove punctuations and numbers
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\d+', '', text)
                # Remove stopwords and lemmatize
                text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
                return text
                
            # Preprocess the input text
            processed_text = preprocess_text(Stock_Text)
            
            # Extract sentiment features
            sentiment_neg = sid.polarity_scores(str(Stock_Text))['neg']
            sentiment_neu = sid.polarity_scores(str(Stock_Text))['neu']
            sentiment_pos = sid.polarity_scores(str(Stock_Text))['pos']
            sentiment_compound = sid.polarity_scores(str(Stock_Text))['compound']
            
            # Vectorize text using TF-IDF
            X_tfidf = tfidf.transform([processed_text])
            
            # Add sentiment features
            import numpy as np
            import scipy.sparse as sp
            sentiment_features = np.array([[sentiment_neg, sentiment_neu, sentiment_pos, sentiment_compound]])
            X = sp.hstack((X_tfidf, sentiment_features))
            
            # Make prediction using ensemble
            predict_text = classifier.predict(X)
            ensemble_confidence = max(classifier.predict_proba(X)[0])
            
            # If LSTM model is available, use it for improved accuracy
            if use_lstm:
                # Tokenize and pad the input text
                sequence = tokenizer.texts_to_sequences([processed_text])
                padded_sequence = pad_sequences(sequence, maxlen=max_seq_length)
                
                # Make prediction with LSTM
                lstm_pred = lstm_model.predict(padded_sequence)[0][0]
                lstm_prediction = 1 if lstm_pred > 0.5 else 0
                lstm_confidence = max(lstm_pred, 1 - lstm_pred)
                
                # If LSTM confidence is higher, use LSTM prediction
                if lstm_confidence > ensemble_confidence:
                    predict_text = np.array([lstm_prediction])
            
            prediction = int(predict_text[0])
            
            if (prediction == 0):
                val = 'Uptrends'
            elif (prediction == 1):
                val = 'Downtrends'
            
        except FileNotFoundError:
            # Fallback to original model if advanced models aren't available
            df = pd.read_csv('Datasets.csv', encoding='latin-1')

            def apply_response(Label):
                if (Label == 0):
                    return 0  # Uptrends
                elif (Label == 1):
                    return 1  # Downtrends

            df['Results'] = df['Label'].apply(apply_response)

            cv = CountVectorizer()
            X = df['Stock_Text'].apply(str)
            y = df['Results']

            X = cv.fit_transform(X)

            models = []
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
            X_train.shape, X_test.shape, y_train.shape

            # SVM Model
            print("SVM")
            from sklearn import svm

            lin_clf = svm.LinearSVC()
            lin_clf.fit(X_train, y_train)
            predict_svm = lin_clf.predict(X_test)
            svm_acc = accuracy_score(y_test, predict_svm) * 100
            models.append(('svm', lin_clf))

            print("Logistic Regression")
            from sklearn.linear_model import LogisticRegression
            reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
            models.append(('logistic', reg))

            print("Decision Tree Classifier")
            dtc = DecisionTreeClassifier()
            dtc.fit(X_train, y_train)
            models.append(('DecisionTreeClassifier', dtc))

            classifier = VotingClassifier(models)
            classifier.fit(X_train, y_train)

            vector1 = cv.transform([Stock_Text]).toarray()
            predict_text = classifier.predict(vector1)

            pred = str(predict_text).replace("[", "")
            pred1 = pred.replace("]", "")

            prediction = int(pred1)

            if (prediction == 0):
                val = 'Uptrends'
            elif (prediction == 1):
                val = 'Downtrends'

        print(val)
        
        predict_investor_sentiment.objects.create(
        Investor_Age=Investor_Age,
        Investor_Gender=Investor_Gender,
        PDate=PDate,
        Stock_Text=Stock_Text,
        Stock_Name=Stock_Name,
        Company_Name=Company_Name,
        Prediction=val)

        return render(request, 'RUser/Predict_Investor_Sentiment_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Investor_Sentiment_Type.html')



