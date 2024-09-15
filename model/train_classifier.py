# Import necessary libraries
import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  


# Initialize stop words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tfidf = TfidfTransformer()


def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name='DisasterResponse', con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X, Y


def tokenize(text):
    """
    Tokenizes input text by normalizing case, removing punctuation, 
    lemmatizing words, and filtering out stopwords.
    
    Args:
        text (str): The text to be tokenized.
    
    Returns:
        list: A list of processed tokens.
    """
    
    
    # Normalize text: lowercasing and removing punctuation
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens if token not in stop_words]
    
    return tokens

def extract_pos_features(X):
    """
    Extracts part-of-speech (POS) features from a list of text data.

    For each text, the function counts the number of nouns, verbs, and adjectives
    and returns them as a numpy array.

    Parameters:
    X (list of str): Input text data.

    Returns:
    np.ndarray: Array of shape (n_samples, 3) where each row contains
                the counts of nouns, verbs, and adjectives for each text.
    """

    def pos_features_nltk(text):
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        # Count the occurrences of each POS tag
        pos_counts = nltk.FreqDist(tag for (word, tag) in pos_tags)

        # Extract specific features (e.g., number of nouns, verbs, adjectives)
        num_nouns = (
            pos_counts["NN"]
            + pos_counts["NNS"]
            + pos_counts["NNP"]
            + pos_counts["NNPS"]
        )
        num_verbs = (
            pos_counts["VB"]
            + pos_counts["VBD"]
            + pos_counts["VBG"]
            + pos_counts["VBN"]
            + pos_counts["VBP"]
            + pos_counts["VBZ"]
        )
        num_adjectives = pos_counts["JJ"] + pos_counts["JJR"] + pos_counts["JJS"]

        return np.array([num_nouns, num_verbs, num_adjectives])

    # Apply the POS feature extraction to the entire dataset
    return np.array([pos_features_nltk(text) for text in X])


def extract_keyword_features(X):
    """
    Extracts keyword-based features from a list of text data.
    
    For each text, the function counts the occurrences of predefined 
    keywords (stored in the 'keywords' dictionary) across various 
    categories and returns the counts as a numpy array.

    Parameters:
    X (list of str): Input list of text data.

    Returns:
    np.ndarray: Array of shape (n_samples, n_features) where each row contains
                the counts of keywords for each category in the corresponding text.
    """
    
    keywords = {
    'related': ['related', 'connection', 'relevant'],
    'request': ['request', 'need', 'require', 'ask'],
    'offer': ['offer', 'provide', 'supply', 'give'],
    'aid_related': ['aid', 'help', 'assist', 'support', 'relief'],
    'medical_help': ['medical', 'doctor', 'nurse', 'hospital', 'medicine'],
    'medical_products': ['medicines', 'drugs', 'supplies', 'equipment'],
    'search_and_rescue': ['rescue', 'search', 'save', 'find'],
    'security': ['security', 'safe', 'protect', 'guard', 'sos'],
    'military': ['military', 'army', 'soldiers', 'troops'],
    'water': ['water', 'drink', 'hydrate', 'thirst'],
    'food': ['food', 'hunger', 'eat', 'nutrition'],
    'shelter': ['shelter', 'house', 'home', 'accommodation'],
    'clothing': ['clothes', 'clothing', 'wear', 'apparel'],
    'money': ['money', 'funds', 'cash', 'payment'],
    'missing_people': ['missing', 'lost', 'disappear', 'find'],
    'refugees': ['refugees', 'displaced', 'asylum', 'immigrant'],
    'death': ['death', 'dead', 'fatal', 'deceased'],
    'other_aid': ['other aid', 'additional help', 'extra support'],
    'infrastructure_related': ['infrastructure', 'roads', 'bridges', 'buildings'],
    'transport': ['transport', 'vehicle', 'car', 'truck'],
    'buildings': ['building', 'construction', 'structure'],
    'electricity': ['electricity', 'power', 'energy', 'light'],
    'tools': ['tools', 'equipment', 'gear'],
    'hospitals': ['hospital', 'clinic', 'health center'],
    'shops': ['shop', 'store', 'market'],
    'aid_centers': ['aid center', 'help center', 'support center'],
    'other_infrastructure': ['other infrastructure', 'facilities', 'utilities'],
    'weather_related': ['weather', 'climate', 'storm', 'rain'],
    'floods': ['flood', 'flooding', 'water overflow'],
    'storm': ['storm', 'hurricane', 'typhoon', 'cyclone'],
    'fire': ['fire', 'burn', 'flame'],
    'earthquake': ['earthquake', 'tremor', 'seismic'],
    'cold': ['cold', 'freeze', 'chill'],
    'other_weather': ['other weather', 'weather condition', 'climate issue'],
    'direct_report': ['direct report', 'first-hand', 'on the ground']
}
    
    
    def keyword_features(text):
        text_lower = text.lower()
        features = []
        for category, words in keywords.items():
            # Count the presence of any keyword in the text
            count = sum(text_lower.count(word) for word in words)
            features.append(count)
        return np.array(features)

    # Apply the keyword feature extraction to the entire dataset
    return np.array([keyword_features(text) for text in X])



def build_model():
    """
    Builds a machine learning pipeline that processes text data, 
    extracts features using a combination of text vectorization, 
    POS tagging, and keyword detection, and applies a MultiOutputClassifier 
    with XGBoost for classification.

    Returns:
    Pipeline: A scikit-learn pipeline for multi-label text classification.
    """
    # Define the pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),
            ('pos_features', FunctionTransformer(extract_pos_features, validate=False)),
            ('keyword_features', FunctionTransformer(extract_keyword_features, validate=False))
        ])),
        ('classifier', MultiOutputClassifier(
            XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=7,
                learning_rate=0.2,
                n_estimators=50,
                reg_alpha=0.01,
                random_state=42,
                scale_pos_weight=4,
                n_jobs=-1
            )
        ))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test):
    """
    Evaluates a multi-label classification model by generating classification reports 
    for each label (column) in the test set.

    Parameters:
    model : scikit-learn estimator
        The trained multi-output classification model.
    X_test : pandas DataFrame
        The test feature set.
    Y_test : pandas DataFrame
        The true labels for the test set, where each column corresponds to a different label.
    
    Returns:
    report_df : pandas DataFrame
        A DataFrame containing precision, recall, and F1-score for each label, 
        for both the 0 and 1 classes.
    """
    
    # Initialize an empty DataFrame to store the report
    report_df = pd.DataFrame()
    
    # Get the predictions from the model
    predictions = model.predict(X_test)
    

    # Iterate over each label (column) in Y_test
    for i, col in enumerate(Y_test.columns):
        # Generate the classification report for each label
        report = classification_report(Y_test.iloc[:, i], predictions[:, i], output_dict=True)
        print(report)
        
        # Create a new row with relevant metrics
        new_row = pd.DataFrame([{
            'Label': col, 
            '0[precision]': report['0']['precision'],
            '0[recall]': report['0']['recall'],
            '0[f1-score]': report['0']['f1-score'], 
            '1[precision]': report['1']['precision'],
            '1[recall]': report['1']['recall'],
            '1[f1-score]': report['1']['f1-score']
        }])
        
        # Concatenate the new row to the report dataframe
        report_df = pd.concat([report_df, new_row], ignore_index=True)
    
    return report_df


def save_model(model, model_filepath):
    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()