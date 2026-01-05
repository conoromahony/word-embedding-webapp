"""
Flask application for word embedding visualization.
Allows users to explore GloVe and Word2Vec embeddings.
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from gensim.models import KeyedVectors
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration for embedding file paths
GLOVE_PATH = os.environ.get('GLOVE_PATH', 'embeddings/glove.6B.50d.txt')
WORD2VEC_PATH = os.environ.get('WORD2VEC_PATH', 'embeddings/GoogleNews-vectors-negative300.bin')

# Global variables to store loaded embeddings
embeddings = {
    'glove': None,
    'word2vec': None
}


def load_glove_embeddings(file_path, max_words=100000):
    """
    Load GloVe embeddings from a text file.
    
    Args:
        file_path: Path to the GloVe embedding file
        max_words: Maximum number of words to load (to manage memory)
    
    Returns:
        Dictionary with word as key and embedding vector as value
    """
    embeddings_dict = {}
    word_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if word_count >= max_words:
                    break
                try:
                    values = line.split()
                    if len(values) < 2:
                        continue  # Skip malformed lines
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    embeddings_dict[word] = vector
                    word_count += 1
                except (ValueError, IndexError) as e:
                    # Skip lines with invalid data
                    logging.warning(f"Skipping malformed line in GloVe file: {e}")
                    continue
        
        logging.info(f"Loaded {len(embeddings_dict)} GloVe embeddings")
        return embeddings_dict
    except FileNotFoundError:
        logging.error(f"GloVe file not found: {file_path}")
        return None


def load_word2vec_embeddings(file_path):
    """
    Load Word2Vec embeddings using gensim.
    
    Args:
        file_path: Path to the Word2Vec binary file
    
    Returns:
        KeyedVectors object
    """
    try:
        model = KeyedVectors.load_word2vec_format(file_path, binary=True)
        logging.info(f"Loaded Word2Vec embeddings: {len(model.index_to_key)} words")
        return model
    except FileNotFoundError:
        logging.error(f"Word2Vec file not found: {file_path}")
        return None


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def find_most_similar_glove(word, embeddings_dict, top_n=12):
    """
    Find most similar words using GloVe embeddings.
    
    Note: This function has O(n) complexity as it computes similarity with all words.
    For production use with large vocabularies, consider optimizing with:
    - Pre-computed similarity matrices
    - Approximate nearest neighbor algorithms (e.g., FAISS, Annoy)
    - Caching frequently queried words
    
    Args:
        word: Target word
        embeddings_dict: Dictionary of word embeddings
        top_n: Number of similar words to return
    
    Returns:
        List of tuples (word, similarity_score)
    """
    if word not in embeddings_dict:
        return None
    
    word_vector = embeddings_dict[word]
    similarities = []
    
    for other_word, other_vector in embeddings_dict.items():
        if other_word != word:
            similarity = cosine_similarity(word_vector, other_vector)
            similarities.append((other_word, float(similarity)))
    
    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    """
    API endpoint to get word embedding and similar words.
    
    Expects JSON with:
        - word: The word to look up
        - model: Either 'glove' or 'word2vec'
    
    Returns JSON with:
        - word: The queried word
        - embedding: The embedding vector
        - similar_words: List of similar words with scores
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON in request body'}), 400
    
    word_input = data.get('word', '').strip()
    model_type = data.get('model', 'glove')
    
    if not word_input:
        return jsonify({'error': 'Please provide a word'}), 400
    
    # Handle case sensitivity based on model type
    # GloVe embeddings are typically lowercase
    # Word2Vec may preserve case for proper nouns
    if model_type == 'glove':
        word = word_input.lower()
    else:
        word = word_input
    
    # Load embeddings if not already loaded
    if model_type == 'glove' and embeddings['glove'] is None:
        if os.path.exists(GLOVE_PATH):
            embeddings['glove'] = load_glove_embeddings(GLOVE_PATH)
        else:
            return jsonify({
                'error': f'GloVe embeddings not found. Please download and place them at {GLOVE_PATH}'
            }), 404
    
    if model_type == 'word2vec' and embeddings['word2vec'] is None:
        if os.path.exists(WORD2VEC_PATH):
            embeddings['word2vec'] = load_word2vec_embeddings(WORD2VEC_PATH)
        else:
            return jsonify({
                'error': f'Word2Vec embeddings not found. Please download and place them at {WORD2VEC_PATH}'
            }), 404
    
    # Process based on model type
    if model_type == 'glove':
        if embeddings['glove'] is None:
            return jsonify({'error': 'Failed to load GloVe embeddings'}), 500
        
        if word not in embeddings['glove']:
            return jsonify({'error': f'Word "{word}" not found in GloVe embeddings'}), 404
        
        embedding_vector = embeddings['glove'][word].tolist()
        similar_words = find_most_similar_glove(word, embeddings['glove'])
        
    elif model_type == 'word2vec':
        if embeddings['word2vec'] is None:
            return jsonify({'error': 'Failed to load Word2Vec embeddings'}), 500
        
        try:
            embedding_vector = embeddings['word2vec'][word].tolist()
            similar = embeddings['word2vec'].most_similar(word, topn=12)
            similar_words = [(w, float(score)) for w, score in similar]
        except KeyError:
            # Try lowercase as fallback
            try:
                word = word.lower()
                embedding_vector = embeddings['word2vec'][word].tolist()
                similar = embeddings['word2vec'].most_similar(word, topn=12)
                similar_words = [(w, float(score)) for w, score in similar]
            except KeyError:
                return jsonify({'error': f'Word "{word_input}" not found in Word2Vec embeddings'}), 404
    
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    
    return jsonify({
        'word': word,
        'embedding': embedding_vector,
        'similar_words': similar_words
    })


if __name__ == '__main__':
    # Create embeddings directory if it doesn't exist
    os.makedirs('embeddings', exist_ok=True)
    
    # Get configuration from environment variables
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    
    app.run(debug=debug_mode, host=host, port=port)
