import numpy as np
import pandas as pd
import json
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

class OutfitRecommender:
    def __init__(self, products_df: pd.DataFrame):
        self.df = products_df.copy()
        self._initialize_engine()

    def _initialize_engine(self):
        self._parse_style_features()
        self._encode_style_features()
        self._map_categories()
        self._define_compatibility()

    def _parse_style_features(self):
        def parse_attributes(x):
            try:
                if isinstance(x, str):
                    attr_dict = json.loads(x)
                    features = []
                    for k, v in attr_dict.items():
                        if isinstance(v, list):
                            features.extend([f"{k}_{item}" for item in v])
                        else:
                            features.append(f"{k}_{v}")
                    return features
                return []
            except Exception as e:
                warnings.warn(f"Failed to parse attributes: {x}. Error: {e}")
                return []

        self.df['style_features'] = self.df['style_attributes'].apply(parse_attributes)

    def _encode_style_features(self):
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.style_feature_matrix = self.mlb.fit_transform(self.df['style_features'])

    def _map_categories(self):
        self.category_to_indices = self.df.groupby('category_id').indices

    def _define_compatibility(self):
        self.compatibility_map = {
            1: [2, 3],  # Tops
            2: [1, 3],  # Bottoms
            3: [1, 2],  # Shoes
            # Extendable
        }

    def get_compatible_categories(self, category_id):
        return self.compatibility_map.get(category_id, [])

    def recommend(self, product_id, top_k=5):
        product_row = self.df[self.df['product_id'] == product_id]
        if product_row.empty:
            warnings.warn(f"Product id {product_id} not found in the dataset.")
            return []

        idx = product_row.index[0]
        product_cat = self.df.at[idx, 'category_id']
        compatible_cats = self.get_compatible_categories(product_cat)

        if not compatible_cats:
            warnings.warn(f"No compatible categories found for category {product_cat}.")
            return []

        query_vec = self.style_feature_matrix[idx]

        candidate_indices = []
        for cat in compatible_cats:
            candidate_indices.extend(self.category_to_indices.get(cat, []))

        if not candidate_indices:
            warnings.warn("No candidate products found for compatible categories.")
            return []

        candidate_matrix = self.style_feature_matrix[candidate_indices]

        sims = cosine_similarity(query_vec, candidate_matrix).flatten()
        top_indices_local = sims.argsort()[-top_k:][::-1]
        recommended_indices = [candidate_indices[i] for i in top_indices_local]

        results = self.df.loc[recommended_indices, [
            'product_id', 'product_name', 'category_id', 'brand', 'feature_image_s3'
        ]]
        results['similarity_score'] = sims[top_indices_local]

        return results.reset_index(drop=True)

