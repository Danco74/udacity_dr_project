import numpy as np
import nltk
from nltk import word_tokenize, pos_tag

# Make sure the necessary NLTK corpora are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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
        try:
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
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            return np.array([0, 0, 0])  # Return zeros on failure

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
