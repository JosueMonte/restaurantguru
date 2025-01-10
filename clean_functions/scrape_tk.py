import pandas as pd
import numpy as np
from serpapi import GoogleSearch
import re
from time import sleep
import time
import json
from typing import Dict, List, Tuple
import scrapy
from datetime import datetime
import urllib.parse
from google.cloud import bigquery
from scrapy.utils.log import configure_logging
import requests
from google.cloud import bigquery

#############################################################################################################################################################################################

class GoogleMapsScraper:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def extract_address_components(self, address: str, name: str, city: str, state_id: str, zips: str) -> Tuple[str, str, str, str]:
        """
        Extract address components from a full address string
        """
        try:
            address = address.replace(name + ", ", "")
            # Check for full format: street, city, state, postal code, country
            match_full = re.match(r"^(.*),\s*(.*),\s*(\w{2})\s*(\d{5}),\s*(.*)$", address)
            if match_full:
                street, city, state, postal_code, country = match_full.groups()
                return street, city, state, postal_code

            # Check for shorter format: street, city, state, postal code
            match_short = re.match(r"^(.*),\s*(.*),\s*(\w{2})\s*(\d{5})$", address)
            if match_short:
                street, city, state, postal_code = match_short.groups()
                return street, city, state, postal_code
        except Exception:
            pass
        # Return fallback values if parsing fails
        return address, city, state_id, zips[:5]

    def extract_coordinates(self, gps_coords: Dict) -> Tuple[float, float]:
        """
        Extract latitude and longitude from GPS coordinates
        """
        try:
            return gps_coords['latitude'], gps_coords['longitude']
        except:
            return None, None

    def format_operating_hours(self, hours: Dict) -> List[List[str]]:
        """
        Format operating hours into a list of lists
        """
        if not hours:
            return None
        return [[day.capitalize(), period] for day, period in hours.items()]

    def extract_location_data(self, results: List[Dict], df_info_states: pd.DataFrame) -> List[Dict]:
        """
        Extract and transform required data from API results
        """
        processed_data = []

        for result in results:
            # Extract additional info from df_info_states
            city = df_info_states.loc[0, 'city']
            state_id = df_info_states.loc[0, 'state_id']
            zips = df_info_states.loc[0, 'zips']

            # Extract address components
            street, city, state, postal_code = self.extract_address_components(
                result.get('address', ''), 
                result.get('title', ''),
                city,
                state_id,
                zips
            )

            # Extract coordinates
            lat, lng = self.extract_coordinates(result.get('gps_coordinates', {}))

            # Format categories
            categories = result.get('types', [])

            # Create processed entry
            processed_entry = {
                'gmap_id': result.get('data_id'),
                'name': result.get('title'),
                'address': street,
                'city': city,
                'state': state,
                'postal_code': postal_code,
                'latitude': lat,
                'longitude': lng,
                'stars': result.get('rating'),
                'review_count': result.get('reviews'),
                'categories': categories,
                'hours': self.format_operating_hours(result.get('operating_hours'))
            }

            processed_data.append(processed_entry)

        return processed_data

    def scrape_google_maps(self, df_info_states: pd.DataFrame, min_results: int = 20) -> pd.DataFrame:
        """
        Scrape Google Maps data using SerpAPI
        """
        all_results = []

        for _, row in df_info_states.iterrows():
            lat, lng = row['lat'], row['lng']

            params = {
                "engine": "google_maps",
                "type": "search",
                "q": "restaurant",
                "ll": f"@{lat},{lng},14z",
                "hl": "en",
                "api_key": self.api_key
            }

            search = GoogleSearch(params)
            results = []

            while len(results) < min_results:
                try:
                    data = search.get_dict()
                    local_results = data.get('local_results', [])

                    if not local_results:
                        break

                    results.extend(local_results)

                    # Check if there's a next page
                    if 'next' not in data.get('serpapi_pagination', {}):
                        break

                    # Update parameters for next page
                    params.update({"start": len(results)})
                    search = GoogleSearch(params)

                    # Add delay to avoid rate limiting
                    sleep(2)

                except Exception as e:
                    print(f"Error occurred: {e}")
                    break

            processed_results = self.extract_location_data(results, df_info_states)
            all_results.extend(processed_results)

        # Create DataFrame and save to parquet
        df_results = pd.DataFrame(all_results)

        # Ensure all required columns are present
        required_columns = [
            'gmap_id', 'name', 'address', 'city', 'state', 'postal_code',
            'latitude', 'longitude', 'stars', 'review_count', 'categories', 'hours'
        ]

        for col in required_columns:
            if col not in df_results.columns:
                df_results[col] = None

        # Reorder columns
        df_results = df_results[required_columns]

        return df_results

#############################################################################################################################################################################################

class GoogleReviewsScraper:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def convert_to_unix(self, iso_date: str) -> int:
        """Convert ISO 8601 date to Unix timestamp."""
        try:
            return int(time.mktime(time.strptime(iso_date, "%Y-%m-%dT%H:%M:%SZ")))
        except Exception:
            return None

    def process_reviews(self, df_gmap_ids: pd.DataFrame) -> pd.DataFrame:
        """Fetch and process reviews from Google Maps for each gmap_id in the DataFrame."""
        all_reviews = []

        for _, row in df_gmap_ids.iterrows():
            gmap_id = row['gmap_id']

            params = {
                "engine": "google_maps_reviews",
                "data_id": gmap_id,
                "hl": "fr",
                "api_key": self.api_key,
                "sort_by": "newestFirst"
            }

            search = GoogleSearch(params)
            results = search.get_dict()
            reviews = results.get("reviews", [])

            for review in reviews:
                user_info = review.get("user", {})

                processed_review = {
                    "user_id": user_info.get("contributor_id"),
                    "name": user_info.get("name"),
                    "time": self.convert_to_unix(review.get("iso_date_of_last_edit")),
                    "rating": review.get("rating"),
                    "text": review.get("snippet", None),
                    "pics": None,
                    "resp": None,  # response field set to null
                    "gmap_id": gmap_id
                }

                all_reviews.append(processed_review)

        return pd.DataFrame(all_reviews)

    def save_reviews_to_parquet(self, df: pd.DataFrame, filename: str):
        """Save reviews DataFrame to a parquet file."""
        df.to_parquet(filename, index=False)
#############################################################################################################################################################################################

class YelpReviewsSpider(scrapy.Spider):
    name = "yelp_reviews"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configuración de BigQuery
        self.client = bigquery.Client()
        self.dataset_name = 'TPFDS'  # Cambia esto por el nombre de tu dataset
        self.business_table = 'Business'
        self.reviews_table = 'Reviews'
        self.page_limit = int(kwargs.get('page_limit', 3))  # Límite de páginas a scrapear

        # Configurar logging
        configure_logging({'LOG_LEVEL': 'INFO'})

    def start_requests(self):
        # Obtiene las URLs de los negocios desde BigQuery
        query = f"""
        SELECT y_url FROM `{self.client.project}.{self.dataset_name}.{self.business_table}`
        WHERE y_url IS NOT NULL
        LIMIT 2
        """
        query_job = self.client.query(query)

        for row in query_job:
            base_url = row['y_url'].split('?')[0]  # Remover parámetros de consulta
            yield scrapy.Request(
                url=f"{base_url}?page_src=related_bizes&sort_by=date_desc",
                callback=self.parse,
                meta={
                    'business_url': base_url,
                    'current_page': 1
                }
            )

    def parse(self, response):
        self.logger.info(f"Accessing page: {response.url}")
        business_url = response.meta['business_url']
        current_page = response.meta['current_page']

        reviews = response.css('li.y-css-1sqelp2')  # Ajusta este selector según el DOM inspeccionado

        if not reviews:
            self.logger.warning("No reviews found on the page. Check your selectors.")
            return

        rows_to_insert = []
        for review in reviews:
            # Extraer fecha de la review
            review_date_text = review.css('span.y-css-1d8mpv1::text').get()
            if not review_date_text:
                self.logger.warning("Review date not found. Skipping review.")
                continue

            try:
                review_date = datetime.strptime(review_date_text, '%b %d, %Y')
            except ValueError as e:
                self.logger.error(f"Error parsing date: {review_date_text} - {e}")
                continue

            # Extraer detalles de la review
            user_name = review.css('a.y-css-1x1e1r2::text').get()
            rating_text = review.css('div.y-css-dnttlc::attr(aria-label)').get()
            rating = int(float(rating_text.split(' ')[0])) if rating_text else None
            review_text = review.css('span.raw__09f24__T4Ezm::text').get()

            # Preparar fila para inserción
            rows_to_insert.append({
                'id': self.get_next_review_id(),
                'origin': 'yelp',
                'rating': rating,
                'review': review_text,
                'date': review_date,
                #'business_id': self.get_business_id_from_url(business_url)
            })

        if rows_to_insert:
            self.insert_reviews_into_bigquery(rows_to_insert)

        # Manejar paginación si no hemos alcanzado el límite
        if current_page < self.page_limit:
            next_page = f"{business_url}?page_src=related_bizes&sort_by=date_desc&start={current_page * 10}"
            yield scrapy.Request(next_page, callback=self.parse, meta={
                'business_url': business_url,
                'current_page': current_page + 1
            })

    def get_next_review_id(self):
        query = f"""
        SELECT MAX(id) AS max_id
        FROM `{self.client.project}.{self.dataset_name}.{self.reviews_table}`
        """
        query_job = self.client.query(query)
        results = list(query_job)
        return (results[0].max_id + 1) if results and results[0].max_id else 1

    def get_business_id_from_url(self, url):
        """Extrae un identificador único de la URL del negocio (por ejemplo, el segmento final)."""
        return url.split('/')[-1]

    def insert_reviews_into_bigquery(self, rows):
        table_id = f"{self.client.project}.{self.dataset_name}.{self.reviews_table}"
        # Convertir datetime a string en formato ISO 8601
        for row in rows:
            if isinstance(row['date'], datetime):
                row['date'] = row['date'].isoformat()  # Formato ISO 8601

        errors = self.client.insert_rows_json(table_id, rows)
        if errors:
            self.logger.error(f"Errors occurred while inserting rows: {errors}")
        else:
            self.logger.info(f"Successfully inserted {len(rows)} rows into {table_id}.")

# Ejecutor del script
if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess()
    process.crawl(YelpReviewsSpider, page_limit=3)  # Limita a 3 páginas por negocio
    process.start()

#############################################################################################################################################################################################

class YelpBusinessFetcher:
    def __init__(self, api_key, dataset, business_table):
        self.api_key = api_key
        self.base_url = 'https://api.yelp.com/v3/businesses/search'
        self.headers = {'Authorization': f'Bearer {api_key}'}
        self.dataset = dataset
        self.business_table = business_table
        self.client = bigquery.Client()

    def get_cities_from_bigquery(self):
        """Obtiene las ciudades desde BigQuery usando una query predefinida."""
        query = """
            SELECT
            Ciudades.nombre,
            Estado.Codigo_estado,
            Ciudades.id AS id_city
            FROM
                `clasificacion-renders-vs-fotos`.`TPFDS`.`Ciudades` AS Ciudades
                INNER JOIN `clasificacion-renders-vs-fotos`.`TPFDS`.`Estado` AS Estado ON Ciudades.id_estado = Estado.id
            ORDER BY
            Estado.id
            LIMIT 1;
        """
        query_job = self.client.query(query)
        return [(row.nombre, row.Codigo_estado, row.id_city) for row in query_job]

    def get_last_id_from_bigquery(self):
        """Obtiene el último ID de la tabla business en BigQuery."""
        query = f"""
        SELECT MAX(id) AS max_id
        FROM `{self.client.project}.{self.dataset}.{self.business_table}`
        """
        query_job = self.client.query(query)
        results = list(query_job)
        return results[0].max_id if results and results[0].max_id else 0

    def get_businesses(self, location, categories='restaurants', limit=50, offset=0):
        """Consulta a la API de Yelp para obtener negocios en una ubicación."""
        if offset >= 240:
            print(f"Reached API limit for {location}: offset {offset}")
            return None

        params = {
            'location': location,
            'categories': categories,
            'limit': min(50, 240 - offset),  # Ajusta el límite para no exceder 240
            'offset': offset
        }
        response = requests.get(self.base_url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.json()}")
            return None

    def is_business_in_database(self, y_id):
        """Verifica si un negocio ya está en la base de datos de BigQuery."""
        query = f"""
        SELECT y_id FROM `{self.client.project}.{self.dataset}.{self.business_table}`
        WHERE y_id = '{y_id}'
        """
        query_job = self.client.query(query)
        results = list(query_job)
        return len(results) > 0

    def insert_into_bigquery(self, businesses, id_city):
        """Inserta negocios en BigQuery."""
        rows_to_insert = []
        last_id = self.get_last_id_from_bigquery()

        for business in businesses:
            if not self.is_business_in_database(business['id']):
                last_id += 1
                rows_to_insert.append({
                    'id': last_id,  # Nuevo ID incremental
                    'y_id': business['id'],
                    'name': business['name'],
                    'address': business['location'].get('address1', ''),
                    'id_city': id_city,
                    'lat': business['coordinates']['latitude'] if 'coordinates' in business else None,
                    'long': business['coordinates']['longitude'] if 'coordinates' in business else None,
                    'post_code': business['location'].get('zip_code', None),
                    'review_count': business.get('review_count', 0),
                    'avg_rating': float(business.get('rating', 0.0)),  # Asegura que sea FLOAT
                    'y_last_review': None,
                    'hours': None,
                    'y_url': business.get('url', '')
                })

        # Inserta las filas en BigQuery
        if rows_to_insert:
            table_id = f"{self.client.project}.{self.dataset}.{self.business_table}"
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                print(f"Errors occurred while inserting rows: {errors}")
            else:
                print(f"Inserted {len(rows_to_insert)} rows into {table_id}.")

    def fetch_and_store_businesses(self):
        """Proceso principal para obtener y almacenar negocios."""
        cities = self.get_cities_from_bigquery()

        for city_name, state_code, id_city in cities:
            city = f"{city_name}, {state_code}"
            print(f"Searching businesses in {city}...")

            offset = 0
            while True:
                data = self.get_businesses(location=city, offset=offset)
                if not data or 'businesses' not in data:
                    break

                self.insert_into_bigquery(data['businesses'], id_city)

                # Incrementa el offset
                offset += len(data['businesses'])

                # Detener si no hay más negocios o si el bloque es menor a 50
                if len(data['businesses']) < 50:
                    break


# Uso del script encapsulado
if __name__ == "__main__":
    API_KEY = "API-KEY"
    DATASET = 'TPFDS'  # Cambia esto por el nombre de tu dataset
    BUSINESS_TABLE = 'Business'

    fetcher = YelpBusinessFetcher(api_key=API_KEY, dataset=DATASET, business_table=BUSINESS_TABLE)
    fetcher.fetch_and_store_businesses()
