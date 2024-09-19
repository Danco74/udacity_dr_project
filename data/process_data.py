import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
    messages_filepath (str): Filepath for the messages CSV file.
    categories_filepath (str): Filepath for the categories CSV file.

    Returns:
    pd.DataFrame: Merged DataFrame containing both messages and categories.
    """
    # Read the datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on 'id'
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean and transform the categories data in the DataFrame.

    Args:
    df (pd.DataFrame): Merged DataFrame containing messages and categories.

    Returns:
    pd.DataFrame: Cleaned DataFrame with separate category columns.
    """
    # Extract category names from the first row
    category_names = df.loc[0, 'categories'].split(';')
    category_names = [re.sub(r'-\d+', '', name) for name in category_names]
    
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = category_names
    
    # Convert category values to numeric (0 or 1)
    categories = categories.apply(lambda col: col.str[-1]).apply(pd.to_numeric)
    
    # Identify rows where all category values are either 0 or 1
    valid_rows = categories.apply(lambda col: col.map(lambda x: x in [0, 1])).all(axis=1)
    
    # Filter the DataFrame to keep only valid rows
    categories = categories[valid_rows]
    df = df[valid_rows]
    
    # Drop the original 'categories' column from df
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new 'categories' DataFrame
    df = pd.concat([df, categories], axis=1)
    
    # Drop categories with 0 positives
    df = df.drop('child_alone',axis=1)
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned DataFrame into an SQLite database.

    Args:
    df (pd.DataFrame): Cleaned DataFrame to save.
    database_filename (str): Filepath for the SQLite database.

    Returns:
    None
    """
    # Create a SQLite engine
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save the DataFrame to the specified SQLite database and table
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()