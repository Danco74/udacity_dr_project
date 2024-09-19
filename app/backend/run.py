import sys
import os
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, inspect, text
import sqlalchemy

# Import custom feature extraction functions
sys.path.append('../../shared')
import feature_extraction


app = Flask(__name__)

# Enable CORS on all routes
CORS(app)


def tokenize(text):
    """Tokenize and lemmatize the input text."""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def create_db_summary(engine):
    """Create a summary of the database by extracting its version, tables, and dialect information."""
    
    print(engine.dialect.name)
    
    summary = {
        "database_url": str(engine.url),
        "driver": engine.dialect.driver,
        "dialect": engine.dialect.name,
        "version": None,
        "tables": None,
        "sqlalchemy_version": sqlalchemy.__version__,
    }

    # Get database version (SQLite example)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT sqlite_version();"))
        summary["version"] = result.fetchone()[0]

    # Get table names and column info
    inspector = inspect(engine)
    summary["tables"] = inspector.get_table_names()

    print(summary)

    return summary


# Define global variables to store database summary, model, and DataFrame
db_summary = None
model = None
df = None


@app.route("/")
@app.route("/dist")
def get_dist():
    """Return the sum of all tag occurrences in the DataFrame, excluding certain columns."""
    
    global df

    # Drop specific columns and calculate the sum for each remaining column
    results = df.drop(columns=["id", "message", "original", "genre"]).sum().to_dict()

    # Return calculated results in JSON format
    return jsonify(results)


@app.route("/tags", methods=["GET"])
def get_tags():
    """Return the list of column names (tags) in the DataFrame."""
    
    global df
    results = df.columns.to_list()
    
    # Return column names in JSON format
    return jsonify(results)


@app.route("/comatrix", methods=["GET"])
def get_co_matrix():
    """Generate and return a co-occurrence matrix for the tag columns in the DataFrame."""
    
    global df

    # Drop non-relevant columns for co-occurrence matrix calculation
    mat_df = df.drop(columns=["id", "message", "original", "genre"])
    co_occurrence_matrix = pd.DataFrame(0, index=mat_df.columns, columns=mat_df.columns)
    
    # Iterate through each column and compute the co-occurrence with other columns
    for col in mat_df.columns:
        for other_col in mat_df.columns:
            # Count the number of times both the current column and the other column have value 1
            co_occurrence = ((mat_df[col] == 1) & (mat_df[other_col] == 1)).sum()
            co_occurrence_matrix.loc[col, other_col] = co_occurrence

    # Convert the co-occurrence matrix to a dictionary
    co_occurrence_dict = co_occurrence_matrix.to_dict()

    # Return the dictionary as a JSON response
    return jsonify(co_occurrence_dict)


@app.route("/db_summary", methods=["GET"])
def get_db_summary():
    """Return the database summary information in JSON format."""
    
    global db_summary
    return jsonify(db_summary)


@app.route("/go", methods=["POST"])
def go():
    """Handle user queries, classify them using the trained model, and return the results."""
    
    # Get the raw data from the request body (assuming it's plain text)
    query = request.get_data(as_text=True)
    
    # Check if the request body is empty
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Use the model to predict classification for the query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    # Convert np.int64 to int for JSON serialization
    cleaned_data = {key: int(value) for key, value in classification_results.items()}
    
    return jsonify(cleaned_data)


def main():
    """Main entry point for the Flask app. Loads the database, model, and starts the server."""
    
    global model, df, db_summary

    # Check if the right number of arguments is provided
    if len(sys.argv) != 4:
        print("Error: Invalid number of arguments.")
        print("Usage: python app.py <database_path> <table_name> <model_path>")
        sys.exit(1)

    # Extract arguments from the command line
    database_path, table_name, model_path = sys.argv[1:4]

    # Validate that the database path exists
    if not os.path.exists(database_path):
        print(f"Error: Database file '{database_path}' not found.")
        sys.exit(1)

    # Validate that the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    # Try to load the database
    try:
        engine = create_engine(f"sqlite:///{database_path}")
        df = pd.read_sql_table(table_name, engine)
    except Exception as e:
        print(f"Error: Failed to load the database or table '{table_name}'. Reason: {e}")
        sys.exit(1)

    # Try to load the model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error: Failed to load the model from '{model_path}'. Reason: {e}")
        sys.exit(1)
        
    # Try to create the database summary
    try:
        db_summary = create_db_summary(engine)
    except Exception as e:
        print(f"Error: Failed to create summary for database. Reason: {e}")
        sys.exit(1)

    # Run the Flask app
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
