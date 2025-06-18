import os
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from app.models.fashion_model import FashionFeatureExtractor

# === Constants & Paths ===
BASE_PATH = r"C:\ADYPU\fashion_visual_search"
FEATURES_PATH = os.path.join(BASE_PATH, "precomputed_features.npy")
INDICES_PATH = os.path.join(BASE_PATH, "valid_indices.npy")
DRESSES_CSV = os.path.join(BASE_PATH, 'dresses_bd_processed_data.csv')
JEANS_CSV = os.path.join(BASE_PATH, 'jeans_bd_processed_data.csv')

# === Global Variables ===
extractor = FashionFeatureExtractor()
scaler, pca, index, df, valid_indices = None, None, None, None, None


def load_and_prepare_data():
    """Loads image data and prepares the FAISS index."""
    global scaler, pca, index, df, valid_indices

    print("[INFO] Loading datasets...")
    dresses_df = pd.read_csv(DRESSES_CSV)
    jeans_df = pd.read_csv(JEANS_CSV)
    df = pd.concat([dresses_df, jeans_df], ignore_index=True)

    if os.path.exists(FEATURES_PATH) and os.path.exists(INDICES_PATH):
        print("[INFO] Loading precomputed features...")
        image_features = np.load(FEATURES_PATH)
        valid_indices = np.load(INDICES_PATH)
    else:
        print("[INFO] Extracting features from images...")
        image_features, valid_indices = extract_and_save_features(df)

    if image_features.size == 0 or len(valid_indices) == 0:
        raise ValueError("[FATAL] Feature arrays are empty or invalid.")

    print(f"[INFO] Feature shape before PCA: {image_features.shape}")
    scaler = StandardScaler()
    image_features = scaler.fit_transform(image_features)

    pca = PCA(n_components=128)
    image_features = pca.fit_transform(image_features)

    print("[INFO] Building FAISS index...")
    index = faiss.IndexFlatL2(image_features.shape[1])
    index.add(image_features)

    print(f"[INFO] FAISS index contains {index.ntotal} vectors.")


def extract_and_save_features(df):
    """Extract features from dataset images and save them to disk."""
    features = []
    valid_idx = []

    for idx, url in enumerate(df['feature_image_s3']):
        if isinstance(url, str) and url.strip():
            feature_vec = extractor.extract_features_from_url(url)
            if feature_vec is not None and isinstance(feature_vec, np.ndarray) and feature_vec.ndim == 1:
                features.append(feature_vec)
                valid_idx.append(idx)
            else:
                print(f"[WARN] Skipping idx {idx} — feature extraction failed.")
        else:
            print(f"[WARN] Invalid URL at idx {idx}")

    features = np.array(features)
    valid_idx = np.array(valid_idx)
    np.save(FEATURES_PATH, features)
    np.save(INDICES_PATH, valid_idx)

    return features, valid_idx


def find_similar_images_from_url(query_image_url: str, top_k: int = 5) -> list:
    """Find visually similar images using a query image URL."""
    try:
        print(f"[INFO] Extracting features from URL: {query_image_url}")
        features = extractor.extract_features_from_url(query_image_url)

        if features is None or features.ndim != 1:
            print("[ERROR] Invalid feature vector from URL.")
            return []

        return search_similar_images(features, top_k)
    except Exception as e:
        print(f"[ERROR] URL-based search failed: {e}")
        return []


def find_similar_images_from_path(image_path: str, top_k: int = 5) -> list:
    """Find visually similar images using a local file path."""
    try:
        print(f"[INFO] Extracting features from file: {image_path}")
        features = extractor.extract_features_from_path(image_path)

        if features is None or features.ndim != 1:
            print("[ERROR] Invalid feature vector from file.")
            return []

        return search_similar_images(features, top_k)
    except Exception as e:
        print(f"[ERROR] File-based search failed: {e}")
        return []


def search_similar_images(query_features: np.ndarray, top_k: int) -> list:
    """Runs a FAISS similarity search for the provided features."""
    query_features = query_features.reshape(1, -1)
    query_features = scaler.transform(query_features)
    query_features = pca.transform(query_features)

    distances, indices = index.search(query_features, top_k)
    print(f"[DEBUG] Indices: {indices.tolist()} | Distances: {distances.tolist()}")

    result_idx = [valid_indices[i] for i in indices[0]]
    return df.iloc[result_idx]['feature_image_s3'].tolist()


# === Initialize System on Module Load ===
print("[INIT] Starting Fashion Visual Search Service...")
load_and_prepare_data()








