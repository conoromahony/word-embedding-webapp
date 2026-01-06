A Flask-based web application for exploring and comparing word embeddings across multiple popular pre-trained models simultaneously.

## Features

- 🔍 Interactive web interface for exploring word embeddings
- 📊 Visualize embedding vectors for any word
- 🔗 Find the 12 most similar words using cosine similarity
- 🎯 Support for multiple embedding models displayed side-by-side:
  - GloVe (6B tokens, 50-dimensional)
  - Word2Vec-10k-Public (100-dimensional)
  - Conor's Word Embeddings (placeholder for future implementation)
- 💅 Clean and intuitive user interface with table-based comparison view

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

### Word2Vec-10k-Public Embeddings

Word2Vec-10k-Public is a Word2Vec model with 10k vocabulary, useful for testing and development with lower memory requirements.

```bash
cd embeddings
import gensim.downloader as api
from gensim.models import Word2Vec

# Download a public corpus automatically
corpus = api.load("text8")

model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=10,  workers=4)
print("Vocabulary size:", len(model.wv))
model.wv.save_word2vec_format("word2vec-10k-public.bin", binary=True)

print("Word2Vec-10k-Public model created successfully!")
EOF
cd ..
```

**Note:** Word2Vec-10k-Public is meant to be a lightweight model. If you need production-quality embeddings, consider using the full Word2Vec Google News model or another pre-trained model and adapting the code accordingly.

### Conor's Word Embeddings (Placeholder)

This is a placeholder for future implementation.

## Configuration

You can customize the application using environment variables:

```bash
export GLOVE_MAX_WORDS=40000  # Maximum words to load from GloVe (default: 40000)
```

By default, the application looks for:
- GloVe: `embeddings/glove.6B.50d.txt`
- Word2Vec-10k-Public: `embeddings/word2vec-10k-public.bin`
- Custom Word Embeddings: `embeddings/custom-embeddings.bin` (placeholder, not yet implemented)

**Security Note:** Never run with `FLASK_DEBUG=true` and `FLASK_HOST=0.0.0.0` in production environments.

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. The application will automatically create the `embeddings` directory if it doesn't exist.

## Usage

1. **Enter a word** you want to explore (e.g., "king", "computer", "happy")
2. **Click "Search"** to see results from all embedding models simultaneously
3. View results in a table format:
   - **Most Similar Words**: Compare the 12 most similar words across all models
   - **Embedding Vectors**: See the numerical representation for each model
4. Each column shows results from a different embedding model:
   - GloVe (purple gradient)
   - Word2Vec-10k-Public (blue gradient)
   - Conor's Word Embeddings (teal gradient, placeholder)

## Project Structure

```
word-embedding-webapp/
├── app.py                 # Flask application and embedding logic
├── templates/
│   └── index.html        # Frontend HTML with CSS and JavaScript
├── embeddings/           # Directory for embedding files (not committed)
│   ├── glove.6B.50d.txt
│   └── word2vec-10k-public.bin
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## How It Works

### Backend (Flask)
- **Load Embeddings**: Loads GloVe and Word2Vec-10k-Public embeddings on demand
- **Cosine Similarity**: Computes similarity between word vectors
- **API Endpoint**: `/get_embedding` receives word queries and returns results from all models:
  - The embedding vector for each model
  - Top 12 similar words with scores for each model
  - Error messages for models where the word is not found

### Frontend (HTML/JavaScript)
- **User Interface**: Clean form for entering words
- **AJAX Requests**: Sends queries to the Flask backend
- **Dynamic Table Display**: Shows results from all models side-by-side in columns
- **Section Ordering**: Most Similar Words displayed above Embedding Vector for better UX

### Word Embeddings
- **GloVe**: Pre-trained on 6 billion tokens with 50-dimensional vectors
- **Word2Vec-10k-Public**: Word2Vec embeddings with 10k vocabulary (100-dimensional)
- **Custom Word Embeddings**: Placeholder for future custom embeddings support
- All models capture semantic relationships between words

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
  - `embeddings/word2vec-10k-public.bin`

**Error: "Word not found"**
- The word might not be in the vocabulary of the selected model(s)
- Try lowercase words for GloVe
- Try different forms of the word (e.g., singular vs. plural)
- Note: Different models may have different vocabularies, so a word might appear in some models but not others

**Slow Performance**
- First load may take time as embeddings are loaded into memory
- Subsequent queries will be faster
- For GloVe, the default configuration loads 400,000 words; adjust `GLOVE_MAX_WORDS` environment variable if needed

## License

MIT License - feel free to use this project for learning and development.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
