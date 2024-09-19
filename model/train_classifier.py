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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# Import custom feature extraction functions
sys.path.append('../shared')
from feature_extraction import extract_pos_features, extract_keyword_features

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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

def build_pipeline():
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
    
    # Print out the evaluation report
    print(report_df)
    
    return report_df

def save_model(model, model_filepath):
    """
    Saves the trained model as a pickle file.
    
    Args:
        model: Trained model
        model_filepath: Path where the pickle file will be saved
    """
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)

def main():
    """
    Main function to train and evaluate the machine learning model.
    The function takes the database file path and model file path as inputs.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:3]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building pipeline...')
        pipeline = build_pipeline()
        
        print('Training model...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(pipeline, X_test, Y_test)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(pipeline, model_filepath)
    
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
