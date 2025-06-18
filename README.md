# Fashion Visual Search Engine

A content-based fashion image similarity search system that helps users find visually similar products (e.g., dresses, jeans) based on uploaded images or image URLs. It uses deep feature extraction, dimensionality reduction (PCA), and FAISS indexing for lightning-fast similarity queries.

---

## Features

- **Deep Feature Extraction** from fashion images (via `FashionFeatureExtractor`)
- **Similarity Search** using [Facebook's FAISS](https://github.com/facebookresearch/faiss)
- **Dimensionality Reduction** with PCA
- **API Endpoints** built with FastAPI
- **Search via Image URL or Upload**
- **Precomputed Feature Caching** for scalability
- **Dataset-agnostic** (easily extendable to other fashion categories)

---

## Project Structure
fashion_visual_search/
│
├── app/
│ ├── main.py # FastAPI app entry point
│ ├── models/
│ │ └── fashion_model.py # Deep fashion feature extractor
│ └── services/
│ └── search_service.py # Core visual search logic
│
├── data/
│ ├── dresses_bd_processed_data.csv
│ ├── jeans_bd_processed_data.csv
│ ├── precomputed_features.npy
│ └── valid_indices.npy
│
├── README.md
└── requirements.txt


---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Sajiiidddd/fashion-visual-search.git
cd fashion-visual-search
```

## 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Organize your dataset
Place the following CSV files in the specified base_path:
dresses_bd_processed_data.csv
jeans_bd_processed_data.csv
Each file should have a column feature_image_s3 with valid image URLs.

## 5. Run the Application

```bash
uvicorn app.main:app --reload
```

## How It Works
Dataset Load: Reads preprocessed fashion metadata CSVs.

Feature Extraction:
Extracts 512D visual features from each image using a deep model (FashionFeatureExtractor).
Normalization + PCA:
Applies StandardScaler then reduces to 128 dimensions using PCA.

Index Building:
Uses FAISS (L2 distance) for similarity matching.

Searching:
Accepts image URL or file.
Extracts query features → scales → PCA → FAISS search → top-K results.

## Tech Stack
FastAPI – API framework
FAISS – Fast Approximate Nearest Neighbors
Pandas / NumPy / Scikit-learn – Data processing
Pillow / Requests / Torch / torchvision – Image processing & model loading

## License
MIT License © 2025 Sajiiidddd






