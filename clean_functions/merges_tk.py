from fuzzywuzzy import fuzz
from scipy.spatial import cKDTree
from typing import List, Tuple
import logging
from haversine import haversine
import pandas as pd
import numpy as np

# Función principal de fuzzy_merge
def fuzzy_merge(
    df_google: pd.DataFrame,
    df_yelp: pd.DataFrame,
    name_threshold: int = 70,
    distance_threshold: float = 0.5,
    address_threshold: int = 60,
    name_col: str = 'name',
    address_col: str = 'address',
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    comparison_method: str = 'ratio',
    log_level: str = 'INFO'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    df_google = df_google.reset_index(drop=True)
    df_yelp = df_yelp.reset_index(drop=True)

    required_cols = [name_col, address_col, lat_col, lon_col]
    for df, name in [(df_google, 'Google'), (df_yelp, 'Yelp')]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # Estandarizamos abreviaciones comunes en direcciones
        replacements = {
            'street': 'st',
            'avenue': 'ave',
            'boulevard': 'blvd',
            'road': 'rd',
            'drive': 'dr',
            'lane': 'ln',
            'first': '1st',
            'second': '2nd',
            'third': '3rd',
            'fourth': '4th',
            'fifth': '5th',
            'suite': 'ste',
            'apartment': 'apt',
            'building': 'bldg',
            'number': '#'
        }
        text = text.lower().strip()
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    df_google['clean_name'] = df_google[name_col].apply(clean_text)
    df_yelp['clean_name'] = df_yelp[name_col].apply(clean_text)
    df_google['clean_address'] = df_google[address_col].apply(clean_text)
    df_yelp['clean_address'] = df_yelp[address_col].apply(clean_text)

    coords_google = np.deg2rad(df_google[[lat_col, lon_col]].values)
    coords_yelp = np.deg2rad(df_yelp[[lat_col, lon_col]].values)
    tree = cKDTree(coords_yelp)
    R = 6371
    distance_rad = distance_threshold / R
    pairs = tree.query_ball_point(coords_google, distance_rad)

    logger.info(f"Found {sum(len(p) for p in pairs)} pairs within distance threshold")

    comparison_funcs = {
        'ratio': fuzz.ratio,
        'partial_ratio': fuzz.partial_ratio,
        'token_sort_ratio': fuzz.token_sort_ratio
    }
    compare_func = comparison_funcs.get(comparison_method, fuzz.ratio)

    matches = []
    partial_matches = []
    used_yelp_indices = set()

    for idx1, nearby_indices in enumerate(pairs):
        if not nearby_indices:
            continue

        name1 = df_google['clean_name'].iloc[idx1]
        address1 = df_google['clean_address'].iloc[idx1]

        similarities = []
        for idx2 in nearby_indices:
            name_sim = compare_func(name1, df_yelp['clean_name'].iloc[idx2])
            address_sim = compare_func(address1, df_yelp['clean_address'].iloc[idx2])
            if name_sim >= name_threshold and address_sim >= address_threshold:
                similarities.append((idx2, name_sim, address_sim))

        for idx2, name_sim, address_sim in similarities:
            coords1 = (df_google[lat_col].iloc[idx1], df_google[lon_col].iloc[idx1])
            coords2 = (df_yelp[lat_col].iloc[idx2], df_yelp[lon_col].iloc[idx2])
            distance = haversine(coords1, coords2)

            match_data = {
                'google_idx': idx1,
                'yelp_idx': idx2,
                'name_similarity': name_sim,
                'address_similarity': address_sim,
                'distance_km': distance
            }

            if idx2 not in used_yelp_indices:
                matches.append(match_data)
                used_yelp_indices.add(idx2)
            else:
                partial_matches.append(match_data)

    logger.info(f"Found {len(matches)} final matches and {len(partial_matches)} partial matches")

    if matches:
        matches_df = pd.DataFrame(matches)
        partial_matches_df = pd.DataFrame(partial_matches)

        df_google_matched = df_google.iloc[matches_df['google_idx']]
        df_yelp_matched = df_yelp.iloc[matches_df['yelp_idx']]

        df_merged = pd.concat([
            df_google_matched.reset_index(drop=True),
            df_yelp_matched.add_suffix('_yelp').reset_index(drop=True),
            matches_df[['name_similarity', 'address_similarity', 'distance_km']].reset_index(drop=True)
        ], axis=1)

        return df_merged, partial_matches_df

    return pd.DataFrame(), pd.DataFrame()

class RestaurantDataCleaner:
    def __init__(self, usa_lat_bounds=(24.396308, 49.384358), usa_lon_bounds=(-125.0, -66.93457)):
        self.valid_states = ['PA', 'FL', 'NC', 'NY', 'TN', 'GA', 'MA', 'VA', 'NJ', 'MD']
        self.usa_bounds = {'lat': usa_lat_bounds, 'lon': usa_lon_bounds}

    def clean_yelp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[
            (df["state"].isin(self.valid_states)) &
            (df["latitude"].between(*self.usa_bounds['lat'])) &
            (df["longitude"].between(*self.usa_bounds['lon']))
        ]
        df = df.rename(columns={"business_id": "id_Y"})
        df["id_G"] = np.nan
        df = df.drop(columns=["attributes", "stars", "review_count", "is_open"], errors='ignore')
        df["categories"] = df["categories"].apply(lambda x: [item.strip() for item in x.split(",")] if isinstance(x, str) else [])
        return df

    def clean_google_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={"gmap_id": "id_G"})
        df = df.drop(columns=["stars", "review_count"], errors='ignore')
        df["id_Y"] = np.nan
        return df

class FuzzyMatcher:
    def __init__(self, name_threshold=75, address_threshold=80, distance_threshold=0.05, comparison_method='token_sort_ratio'):
        self.thresholds = {
            'name': name_threshold,
            'address': address_threshold,
            'distance': distance_threshold
        }
        self.comparison_method = comparison_method

    def merge(self, df_google: pd.DataFrame, df_yelp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return fuzzy_merge(
            df_google, df_yelp,
            name_threshold=self.thresholds['name'],
            address_threshold=self.thresholds['address'],
            distance_threshold=self.thresholds['distance'],
            comparison_method=self.comparison_method
        )

class DataMerger:
    def __init__(self, cleaner: RestaurantDataCleaner, matcher: FuzzyMatcher):
        self.cleaner = cleaner
        self.matcher = matcher

    def merge_datasets(self, df_google: pd.DataFrame, df_yelp: pd.DataFrame) -> pd.DataFrame:
        df_yelp_clean = self.cleaner.clean_yelp_data(df_yelp)
        df_google_clean = self.cleaner.clean_google_data(df_google)

        df_merged, partial_matches = self.matcher.merge(df_google_clean, df_yelp_clean)

        matched_google_ids = set(df_merged['id_G'])
        matched_yelp_ids = set(df_merged['id_Y_yelp'])

        unmatched_google = df_google_clean[~df_google_clean['id_G'].isin(matched_google_ids)].copy()
        unmatched_google['id_Y'] = np.nan

        unmatched_yelp = df_yelp_clean[~df_yelp_clean['id_Y'].isin(matched_yelp_ids)].copy()
        unmatched_yelp['id_G'] = np.nan

        if not partial_matches.empty:
            partial_google = df_google.iloc[partial_matches['google_idx']]
            partial_yelp = df_yelp.iloc[partial_matches['yelp_idx']]
            partial_merged = pd.concat([
                partial_google.reset_index(drop=True),
                partial_yelp.add_suffix('_yelp').reset_index(drop=True),
                partial_matches[['name_similarity', 'address_similarity', 'distance_km']].reset_index(drop=True)
            ], axis=1)
        else:
            partial_merged = pd.DataFrame()

        result = pd.concat([df_merged, partial_merged, unmatched_google, unmatched_yelp], ignore_index=True)

        logger = logging.getLogger(__name__)
        logger.info(f"Final counts: Google={len(df_google)}, Yelp={len(df_yelp)}, Result={len(result)}")

        return result