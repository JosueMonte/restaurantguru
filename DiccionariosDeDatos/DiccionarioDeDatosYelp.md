# Google Yelp

Plataforma de reseñas y recomendaciones de negocios que permite a los usuarios buscar, calificar y escribir opiniones sobre diversos servicios locales, como restaurantes, tiendas, salones de belleza, hoteles y más. Aunque es diferente de Google Maps, ambos compiten en el ámbito de descubrimiento de negocios locales y se complementan en ciertos casos. Yelp se destaca especialmente en el sector de reseñas detalladas y comunidades de usuarios.

---

## Archivo `business.pkl` de Yelp

Contiene datos relacionados con los negocios disponibles en la plataforma. Este archivo, al estar en formato Pickle (.pkl), es un archivo binario de Python que almacena estructuras de datos serializadas, como DataFrames o diccionarios. Es común encontrar este tipo de archivos cuando se trabaja con datasets de gran tamaño, ya que son más rápidos de cargar y procesar en Python.

### Contenido típico de `business.pkl`

El archivo puede contener información estructurada sobre negocios de Yelp, similar a un DataFrame de Pandas. Las columnas principales incluyen:

- **Identificación del negocio**:
  - `business_id`: Identificador único asignado a cada negocio.

- **Información general del negocio**:
  - `name`: Nombre del negocio.
  - `address`: Dirección física.
  - `city`: Ciudad.
  - `state`: Estado o provincia.
  - `postal_code`: Código postal.
  - `latitude`, `longitude`: Coordenadas geográficas.

- **Clasificación del negocio**:
  - `categories`: Categorías a las que pertenece el negocio (e.g., "Restaurantes", "Cafeterías").
  - `stars`: Calificación promedio (1-5 estrellas).
  - `review_count`: Número total de reseñas.

- **Información adicional**:
  - `is_open`: Estado actual del negocio (1 = abierto, 0 = cerrado).
  - `hours`: Horarios de apertura y cierre.
  - `attributes`: Detalles adicionales (e.g., Wi-Fi, opciones veganas).

### Usos principales

- **Análisis exploratorio**: Distribución de negocios por ubicación o categoría.
- **Estudios de mercado**: Relación entre calificaciones y reseñas.
- **Sistemas de recomendación**: Sugerir negocios similares.
- **Mapeo geoespacial**: Visualización en mapas.
- **Predicción del éxito**: Modelos basados en calificaciones y atributos.

---

## Archivo `review.json` de Yelp

Contiene datos detallados sobre las reseñas de los usuarios en diversos negocios. Estas reseñas son clave para análisis de sentimiento, tendencias de comportamiento y sistemas de recomendación.

### Atributos comunes en `review.json`

- `review_id`: Identificador único de la reseña.
- `user_id`: Identificador único del usuario.
- `business_id`: Identificador único del negocio.
- `stars`: Calificación (1-5 estrellas).
- `date`: Fecha de publicación.
- `text`: Texto de la reseña.
- `useful`, `funny`, `cool`: Votos otorgados por otros usuarios.

### Usos principales

- **Análisis de Sentimiento**: Identificar sentimientos y palabras clave.
- **Estudios de Tendencias**: Calificaciones a lo largo del tiempo.
- **Sistema de Recomendación**: Basado en historial de reseñas.
- **Evaluación de Negocios**: Identificación de mejores y peores calificados.
- **Insights de Clientes**: Aspectos valorados por los usuarios.

---

## Archivo `user.parquet` de Yelp

Almacena información detallada sobre los usuarios de Yelp en un formato optimizado para análisis a gran escala.

### Contenido de `user.parquet`

- **Identificación y detalles**:
  - `user_id`: Identificador único.
  - `name`: Nombre del usuario.
  - `yelping_since`: Fecha de ingreso.

- **Actividad**:
  - `review_count`: Reseñas escritas.
  - `useful`, `funny`, `cool`: Votos recibidos.

- **Influencia**:
  - `fans`: Número de seguidores.
  - `average_stars`: Calificación promedio.

- **Social**:
  - `compliments`: Cumplidos recibidos.
  - `friends`: Lista de amigos.

### Usos principales

- **Análisis de actividad**: Identificar usuarios activos o influyentes.
- **Segmentación de usuarios**: Clasificación por comportamiento.
- **Redes sociales**: Análisis de conexiones y comunidades.
- **Sistemas de recomendación**: Predicción de preferencias futuras.
- **Tendencias**: Cambios en hábitos desde su ingreso.

---

## Archivo `checkin.json` de Yelp

Contiene información sobre los registros de visitas a negocios en la plataforma.

### Atributos principales

- `business_id`: Identificador del negocio.
- `date`: Fechas y horas específicas de check-ins.

### Usos principales

- **Patrones de visitas**: Identificación de días y horas populares.
- **Optimización de recursos**: Planificación de personal y recursos.
- **Análisis geográfico**: Comparación de check-ins por región.
- **Recomendaciones temporales**: Mejores horarios para visitar.

---

## Archivo `tip.json` de Yelp

Incluye comentarios breves o tips que los usuarios dejan sobre negocios, diferentes de las reseñas al no incluir calificaciones.

### Atributos principales

- `business_id`: Identificador del negocio.
- `user_id`: Identificador del usuario.
- `text`: Contenido del tip.
- `date`: Fecha del tip.
- `compliment_count`: Cumplidos recibidos (opcional).

### Usos principales

- **Análisis de contenido**: Temas frecuentes en tips.
- **Popularidad**: Interacciones informales de usuarios.
- **Análisis de sentimiento**: Relación con percepción del negocio.
- **Tendencias temporales**: Publicación de tips a lo largo del tiempo.
- **Recomendaciones personalizadas**: Sugerencias basadas en tips.
