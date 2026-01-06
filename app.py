"""
Flask application for word embedding visualization.
Allows users to explore and compare GloVe and Word2Vec-10k-Public embeddings.
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from gensim.models import KeyedVectors
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration for embedding file paths
GLOVE_PATH = 'embeddings/glove.6B.50d.txt'
# Word2Vec-10k-Public: Word2Vec embeddings with 10k vocabulary for public use
WORD2VEC_10K_PUBLIC_PATH = 'embeddings/word2vec-10k-public.bin'
# Custom Word Embeddings: Placeholder for future custom embeddings (not yet implemented)
# This path is reserved for future use when custom embeddings loading is implemented
CUSTOM_EMBEDDINGS_PATH = 'embeddings/custom-embeddings.bin'
GLOVE_MAX_WORDS = int(os.environ.get('GLOVE_MAX_WORDS', '40000'))

# Global variables to store loaded embeddings
embeddings = {
    'glove': None,
    'word2vec_10k_public': None,
    'custom': None  # Placeholder for future implementation
}


def load_glove_embeddings(file_path, max_words=None):
    """
    Load GloVe embeddings from a text file.
    
    Args:
        file_path: Path to the GloVe embedding file
        max_words: Maximum number of words to load (to manage memory).
                   If None, uses GLOVE_MAX_WORDS from environment (default: 400000)
    
    Returns:
        Dictionary with word as key and embedding vector as value
    """
    if max_words is None:
        max_words = GLOVE_MAX_WORDS
    
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


def load_word2vec_10k_public_embeddings(file_path):
    """
    Load Word2Vec-10k-Public embeddings using gensim.
    Word2Vec-10k-Public is a Word2Vec model with 10k vocabulary for public use.
    
    Args:
        file_path: Path to the Word2Vec-10k-Public binary file
    
    Returns:
        KeyedVectors object
    """
    try:
        model = KeyedVectors.load_word2vec_format(file_path, binary=True)
        logging.info(f"Loaded Word2Vec-10k-Public embeddings: {len(model.index_to_key)} words")
        return model
    except FileNotFoundError:
        logging.error(f"Word2Vec-10k-Public file not found: {file_path}")
        return None


def load_custom_embeddings(file_path):
    """
    Placeholder function for loading custom word embeddings.
    This function is not yet implemented and is reserved for future development.
    
    Args:
        file_path: Path to the custom embeddings file
    
    Returns:
        None (not yet implemented)
    """
    # TODO: Implement custom embeddings loading logic
    logging.info("This feature is not yet implemented")
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
    API endpoint to get word embeddings and similar words from all supported models.
    
    Modified to return results from multiple embedding models simultaneously:
    - GloVe (6B tokens, 50-dimensional)
    - Word2Vec-10k-Public (100-dimensional)
    - Custom Word Embeddings (placeholder for future implementation)
    
    Expects JSON with:
        - word: The word to look up
    
    Returns JSON with:
        - word: The queried word
        - results: Dictionary with results from each model (glove, word2vec_10k_public, custom)
          Each result contains:
            - embedding: The embedding vector
            - similar_words: List of similar words with scores
            - error: Error message if the word is not found in that model
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON in request body'}), 400
    
    word_input = data.get('word', '').strip()
    
    if not word_input:
        return jsonify({'error': 'Please provide a word'}), 400
    
    # Dictionary to store results from all models
    results = {}
    
    # Process GloVe embeddings
    word_glove = word_input.lower()  # GloVe embeddings are typically lowercase
    
    if embeddings['glove'] is None:
        if os.path.exists(GLOVE_PATH):
            embeddings['glove'] = load_glove_embeddings(GLOVE_PATH)
        else:
            results['glove'] = {
                'error': f'GloVe embeddings not found at {GLOVE_PATH}'
            }
    
    if embeddings['glove'] is not None:
        if word_glove in embeddings['glove']:
            embedding_vector = embeddings['glove'][word_glove].tolist()
            similar_words = find_most_similar_glove(word_glove, embeddings['glove'])
            results['glove'] = {
                'embedding': embedding_vector,
                'similar_words': similar_words
            }
        else:
            results['glove'] = {
                'error': f'Word "{word_glove}" not found in GloVe embeddings'
            }
    
    # Process Word2Vec-10k-Public embeddings
    word_w2v = word_input
    
    if embeddings['word2vec_10k_public'] is None:
        if os.path.exists(WORD2VEC_10K_PUBLIC_PATH):
            embeddings['word2vec_10k_public'] = load_word2vec_10k_public_embeddings(WORD2VEC_10K_PUBLIC_PATH)
        else:
            results['word2vec_10k_public'] = {
                'error': f'Word2Vec-10k-Public embeddings not found at {WORD2VEC_10K_PUBLIC_PATH}'
            }
    
    if embeddings['word2vec_10k_public'] is not None:
        try:
            embedding_vector = embeddings['word2vec_10k_public'][word_w2v].tolist()
            similar = embeddings['word2vec_10k_public'].most_similar(word_w2v, topn=12)
            similar_words = [(w, float(score)) for w, score in similar]
            results['word2vec_10k_public'] = {
                'embedding': embedding_vector,
                'similar_words': similar_words
            }
        except KeyError:
            # Try lowercase as fallback
            try:
                word_w2v = word_input.lower()
                embedding_vector = embeddings['word2vec_10k_public'][word_w2v].tolist()
                similar = embeddings['word2vec_10k_public'].most_similar(word_w2v, topn=12)
                similar_words = [(w, float(score)) for w, score in similar]
                results['word2vec_10k_public'] = {
                    'embedding': embedding_vector,
                    'similar_words': similar_words
                }
            except KeyError:
                results['word2vec_10k_public'] = {
                    'error': f'Word "{word_input}" not found in Word2Vec-10k-Public embeddings'
                }
    
    # Process Custom embeddings (placeholder - not yet implemented)
    # Note: Custom embeddings are not yet implemented. When implemented, this should
    # follow the same pattern as other embeddings (check file existence, load, process)
    if embeddings['custom'] is None:
        # Placeholder: Always show a message that custom embeddings are not yet implemented
        # When implementing, check if CUSTOM_EMBEDDINGS_PATH exists before showing error
        results['custom'] = {
            'error': 'Custom word embeddings feature is not yet implemented'
        }
    
    # Return error if no results from any model or results is empty
    if not results or all('error' in result for result in results.values()):
        return jsonify({
            'error': 'Word not found in any embedding model',
            'results': results
        }), 404
    
    return jsonify({
        'word': word_input,
        'results': results
    })


if __name__ == '__main__':
    # Create embeddings directory if it doesn't exist
    os.makedirs('embeddings', exist_ok=True)
    
    # Get configuration from environment variables
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    
    app.run(debug=debug_mode, host=host, port=port)
