# Análisis de Sentimiento Basado en Aspectos (ABSA)

La Identificación de Temas o Aspectos Clave es una de las aplicaciones principales del Análisis de Sentimiento Basado en Aspectos (Aspect-Based Sentiment Analysis, ABSA). Este enfoque no solo identifica el sentimiento general de una reseña, sino que también descompone el texto en aspectos específicos para analizar el sentimiento asociado a cada uno. Es particularmente útil en sectores como restaurantes, donde los clientes suelen comentar sobre múltiples facetas del servicio en una sola reseña.

## ¿Qué es el ABSA?

El ABSA es una técnica de análisis de sentimiento que tiene dos objetivos principales:

* Identificar los aspectos clave (temas): Determinar sobre qué temas o categorías está hablando el usuario (por ejemplo, comida, servicio, precio, ambiente).
* Determinar el sentimiento asociado a cada aspecto: Clasificar el sentimiento hacia cada aspecto como positivo, negativo o neutro.

## Ejemplo Práctico

Dada una reseña:
*"La comida estaba deliciosa, pero el servicio fue terrible y el lugar estaba muy ruidoso."*

El ABSA extraería:

### Aspectos identificados:
* Comida
* Servicio
* Ambiente

### Sentimientos asociados:
* Comida → Positivo
* Servicio → Negativo
* Ambiente → Negativo

## Pasos para Implementar ABSA

### 1. Preprocesamiento del Texto
* Eliminar ruido: caracteres especiales, HTML, etc.
* Tokenización: dividir el texto en palabras o frases
* Lematización o stemming: reducir las palabras a su forma base

### 2. Identificación de Aspectos (Aspect Extraction)

Métodos para extraer aspectos:

* **Basados en reglas**: Buscar palabras clave o sustantivos relacionados con aspectos predefinidos (por ejemplo, "comida", "servicio").
* **Basados en aprendizaje automático**: Entrenar modelos de clasificación o extracción (como CRF o BERT) para etiquetar los aspectos.
* **Basados en aprendizaje profundo**: Usar redes neuronales con arquitecturas como LSTM o Transformers para identificar aspectos automáticamente.

### 3. Análisis de Sentimiento por Aspecto

* **Métodos supervisados**: Entrenar un modelo de clasificación para determinar el sentimiento hacia cada aspecto. Ejemplo: Utilizar BERT fine-tuned en tareas de análisis de sentimiento.
* **Métodos no supervisados**: Análisis de palabras o frases con diccionarios de polaridad (p. ej., VADER, SentiWordNet).
* **Modelos combinados**: Usar Transformers como BERT o RoBERTa fine-tuned específicamente para ABSA.

### 4. Visualización de Resultados

Mostrar resultados en gráficos como:
* Histogramas de sentimientos por aspecto
* Comparaciones entre aspectos en diferentes ubicaciones
* Word clouds para palabras más mencionadas en cada aspecto

## Modelos y Herramientas para ABSA

### Modelos Preentrenados:
* BERT y sus variantes (como BERTweet o DistilBERT)
* RoBERTa (una versión mejorada de BERT)
* SpanBERT para tareas específicas de ABSA

### Librerías:
* NLTK o TextBlob: Para métodos básicos de análisis de sentimiento
* SpaCy: Para extracción de aspectos usando modelos preentrenados
* Transformers (Hugging Face): Para ajustar modelos como BERT en ABSA
* Stanford CoreNLP: Ofrece herramientas específicas para ABSA

## Aplicaciones de ABSA en Restaurantes

### Identificación de Problemas Comunes:
* ¿Qué aspectos reciben más críticas negativas (comida, precio, servicio)?

### Análisis Comparativo:
* Comparar diferentes restaurantes o sucursales según aspectos clave

### Mejora del Servicio:
* Priorizar mejoras en aspectos con mayor impacto negativo

### Detección de Cambios:
* Identificar si los cambios en el menú, la administración o el diseño impactaron positivamente en las reseñas


# Guía de Implementación: ABSA con Hugging Face Transformers

Esta guía incluye los pasos básicos para implementar el Análisis de Sentimiento Basado en Aspectos (ABSA) utilizando Hugging Face Transformers, desde la preparación de los datos hasta la ejecución del análisis.

## Requisitos Previos

### Instalación de librerías:

```bash
pip install transformers datasets torch scikit-learn pandas
```

### Dataset para ABSA:
* Datos personalizados (reseñas de restaurantes)

## Pasos para Implementar ABSA

### 1. Importar las Librerías Necesarias
```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
from datasets import Dataset
```

### 2. Preparar los Datos

Tu dataset debe contener:
* **text**: La reseña completa
* **aspect**: El aspecto sobre el cual se evalúa el sentimiento (ejemplo: comida, servicio)
* **sentiment**: La etiqueta del sentimiento asociado (positive, negative, neutral)

Ejemplo de DataFrame:
```python
data = {
    'text': [
        "La comida estaba deliciosa, pero el servicio fue lento.",
        "El ambiente era increíble, pero los precios son demasiado altos."
    ],
    'aspect': ['comida', 'servicio', 'ambiente', 'precio'],
    'sentiment': ['positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)
```

### 3. Tokenizar los Datos

Usaremos un modelo preentrenado como bert-base-uncased.

```python
# Cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenización
def tokenize_data(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=128)

# Convertir el DataFrame en Dataset de Hugging Face
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_data, batched=True)
```

### 4. Dividir los Datos

Dividimos los datos en conjuntos de entrenamiento y prueba:

```python
train_dataset, test_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()
```

### 5. Cargar un Modelo Preentrenado

Utilizamos bert-base-uncased fine-tuned para clasificación.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3
)
```
*Nota: Asegúrate de que num_labels=3 corresponda a las etiquetas: positive, negative, neutral.*

### 6. Configurar el Pipeline

Creamos un pipeline para el análisis:

```python
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)
```

### 7. Predecir Sentimientos por Aspecto

Pasamos el texto y el aspecto al pipeline para analizar el sentimiento.

```python
# Función para predecir el sentimiento por aspecto
def predict_sentiment(text, aspect):
    input_text = f"{aspect}: {text}"
    prediction = classifier(input_text)
    return prediction

# Ejemplo
for index, row in df.iterrows():
    result = predict_sentiment(row['text'], row['aspect'])
    print(f"Aspecto: {row['aspect']}, Resultado: {result}")
```

### 8. Evaluar el Modelo

Si estás entrenando el modelo, evalúa su desempeño:

```python
from sklearn.metrics import classification_report

# Supongamos que tienes predicciones y etiquetas reales
y_true = ["positive", "negative", "neutral"]
y_pred = ["positive", "negative", "positive"]

print(classification_report(y_true, y_pred))
```

### 9. Opcional: Entrenar tu Propio Modelo Fine-Tuned

Si deseas ajustar un modelo preentrenado para tu caso, puedes hacerlo:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

## Resultados y Visualización

Puedes resumir los resultados del ABSA:

- Tablas: Mostrar el porcentaje de sentimientos por aspecto.

- Gráficos: Histogramas o diagramas de barras para comparar aspectos.


### Ejemplo de visualización con Matplotlib:
```python
import matplotlib.pyplot as plt

# Resultados simulados
aspects = ['comida', 'servicio', 'ambiente', 'precio']
sentiments = [75, 50, 85, 30]

plt.bar(aspects, sentiments, color=['green', 'red', 'blue', 'orange'])
plt.title("Sentimientos por Aspecto")
plt.xlabel("Aspectos")
plt.ylabel("Porcentaje de Sentimiento Positivo")
plt.show()
```

## Siguientes Pasos

* Implementar esta guía con datos reales
* Ajustar el modelo para mejorar la precisión
* Explorar visualizaciones más avanzadas en Dashboards interactivos