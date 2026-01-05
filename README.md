# Word Embedding Web Application

A Flask-based web application for exploring and visualizing word embeddings using popular pre-trained models like GloVe and Word2Vec.

## Features

- 🔍 Interactive web interface for exploring word embeddings
- 📊 Visualize embedding vectors for any word
- 🔗 Find the 12 most similar words using cosine similarity
- 🎯 Support for multiple embedding models:
  - GloVe (6B tokens, 50-dimensional)
  - Word2Vec (Google News, 300-dimensional)
- 💅 Clean and intuitive user interface

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/conoromahony/word-embedding-webapp.git
cd word-embedding-webapp
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Downloading Word Embeddings

The application requires pre-trained word embedding files. Create an `embeddings` directory and download the models:

### GloVe Embeddings

1. Download GloVe embeddings from Stanford NLP:
```bash
mkdir -p embeddings
cd embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ..
```

You'll need the `glove.6B.50d.txt` file (50-dimensional vectors).

### Word2Vec Embeddings

1. Download the Google News Word2Vec model from one of these sources:

**Option 1: Using gdown (recommended)**
```bash
pip install gdown
cd embeddings
gdown 0B7XkCwpI5KDYNlNUTTlSS21pQmM
gunzip GoogleNews-vectors-negative300.bin.gz
cd ..
```

**Option 2: Direct download**
- Visit the [official Word2Vec page](https://code.google.com/archive/p/word2vec/)
- Or download from [Kaggle Datasets](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)
- Or use this [alternative mirror](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)

Place the `GoogleNews-vectors-negative300.bin` file in the `embeddings/` directory.

**Note:** The Word2Vec file is approximately 1.5 GB, so the download may take some time.

## Configuration

You can customize the application using environment variables:

**Embedding File Paths:**
```bash
export GLOVE_PATH=/path/to/your/glove.6B.50d.txt
export WORD2VEC_PATH=/path/to/your/GoogleNews-vectors-negative300.bin
export GLOVE_MAX_WORDS=400000  # Maximum words to load from GloVe (default: 400000)
```

By default, the application looks for:
- GloVe: `embeddings/glove.6B.50d.txt`
- Word2Vec: `embeddings/GoogleNews-vectors-negative300.bin`

**Performance Note:** Loading 400,000 GloVe embeddings requires significant memory (~1-2 GB). Reduce `GLOVE_MAX_WORDS` if you have memory constraints.

**Server Configuration:**
```bash
export FLASK_DEBUG=true        # Enable debug mode (default: false, NOT recommended for production)
export FLASK_HOST=0.0.0.0     # Bind to all interfaces (default: 127.0.0.1)
export FLASK_PORT=8080        # Custom port (default: 5000)
```

**Security Note:** Never run with `FLASK_DEBUG=true` and `FLASK_HOST=0.0.0.0` in production environments.

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. The application will automatically create the `embeddings` directory if it doesn't exist.

## Usage

1. **Select an embedding model** from the dropdown menu (GloVe or Word2Vec)
2. **Enter a word** you want to explore (e.g., "king", "computer", "happy")
3. **Click "Search"** to see results
4. View:
   - The word's embedding vector (numerical representation)
   - The 12 most similar words with their similarity scores

## Project Structure

```
word-embedding-webapp/
├── app.py                 # Flask application and embedding logic
├── templates/
│   └── index.html        # Frontend HTML with CSS and JavaScript
├── embeddings/           # Directory for embedding files (not committed)
│   ├── glove.6B.50d.txt
│   └── GoogleNews-vectors-negative300.bin
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## How It Works

### Backend (Flask)
- **Load Embeddings**: Loads GloVe and Word2Vec embeddings on demand
- **Cosine Similarity**: Computes similarity between word vectors
- **API Endpoint**: `/get_embedding` receives word queries and returns:
  - The embedding vector
  - Top 12 similar words with scores

### Frontend (HTML/JavaScript)
- **User Interface**: Clean form for selecting models and entering words
- **AJAX Requests**: Sends queries to the Flask backend
- **Dynamic Display**: Shows embedding vectors and similar words

### Word Embeddings
- **GloVe**: Pre-trained on 6 billion tokens with 50-dimensional vectors
- **Word2Vec**: Pre-trained on Google News corpus with 300-dimensional vectors
- Both capture semantic relationships between words

## Example Queries

Try these words to see interesting relationships:
- **king** → queen, monarch, prince, etc.
- **computer** → software, hardware, laptop, etc.
- **happy** → glad, joyful, pleased, etc.
- **berlin** → germany, munich, vienna, etc.

## Technical Details

- **Flask**: Lightweight web framework for Python
- **NumPy**: Efficient numerical operations for vector computations
- **Gensim**: Library for loading and working with Word2Vec embeddings
- **Cosine Similarity**: Measures the cosine of the angle between two vectors (ranges from -1 to 1)

## Troubleshooting

**Error: "Embeddings not found"**
- Make sure you've downloaded the embedding files and placed them in the `embeddings/` directory
- Check that file names match exactly:
  - `embeddings/glove.6B.50d.txt`
  - `embeddings/GoogleNews-vectors-negative300.bin`

**Error: "Word not found"**
- The word might not be in the vocabulary of the selected model
- Try lowercase words for GloVe
- Try different forms of the word (e.g., singular vs. plural)

**Slow Performance**
- First load may take time as embeddings are loaded into memory
- Subsequent queries will be faster
- For GloVe, the default configuration loads 100,000 words; adjust `max_words` in `app.py` if needed

## License

MIT License - feel free to use this project for learning and development.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
