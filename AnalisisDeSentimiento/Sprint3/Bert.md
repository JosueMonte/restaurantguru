# Análisis de Sentimientos en NLP

El análisis de sentimientos es una técnica del procesamiento del lenguaje natural (NLP) que clasifica las emociones o intenciones expresadas en un texto. Los modelos utilizados para este propósito varían en complejidad y enfoque.

## Modelos preentrenados basados en Transformers

Los más avanzados en la actualidad, capaces de capturar el contexto a nivel palabra, frase y documento.

### Ejemplos:

* **BERT (Bidirectional Encoder Representations from Transformers)**: Captura contexto bidireccional y es adecuado para clasificación de texto.
* **RoBERTa, DistilBERT, XLNet**: Variantes optimizadas de BERT.
* **GPT (Generative Pre-trained Transformer)**: Enfocado en generación de texto, pero también se adapta para clasificación.

## BERT-base-uncased

El modelo BERT-base-uncased es una variante del modelo BERT desarrollado por Google. Es ampliamente utilizado en tareas de procesamiento de lenguaje natural (NLP) debido a su capacidad para generar representaciones contextuales de palabras en un texto.

### Características de BERT-base-uncased

#### Tamaño del modelo:

BERT-base es la versión más pequeña del modelo estándar de BERT (en comparación con BERT-large).

Tiene:
* 12 capas (transformers)
* 768 unidades ocultas por capa
* 12 cabezas de atención
* 110 millones de parámetros

#### "Uncased":

* Indica que el texto de entrada se convierte a minúsculas antes de ser procesado.
* Las diferencias entre mayúsculas y minúsculas no se consideran (por ejemplo, "apple" y "Apple" se tratan como la misma palabra).
* Esto es útil cuando las distinciones entre mayúsculas y minúsculas no son relevantes para la tarea.

#### Entrenamiento bidireccional:

A diferencia de los modelos unidireccionales (como GPT), BERT utiliza un enfoque bidireccional para analizar el contexto completo de una palabra, considerando las palabras que están antes y después en la oración. Esto permite capturar relaciones semánticas más ricas.

### Entrenamiento previo (pretraining):

#### Tareas de entrenamiento previo:

* **Modelado de lenguaje enmascarado (MLM)**: Se ocultan palabras al azar (alrededor del 15%) en la entrada, y el modelo aprende a predecirlas basándose en el contexto.
* **Predicción de la siguiente oración (NSP)**: El modelo aprende a determinar si dos oraciones aparecen juntas en un texto o no.
* Entrenado en grandes corpus, como Wikipedia en inglés y BookCorpus.

### Tokenización:

* Utiliza un tokenizador basado en subpalabras (WordPiece).
* Las palabras se descomponen en subunidades, lo que ayuda al modelo a manejar palabras raras o desconocidas.

## Usos Comunes

### Análisis de Sentimiento:

* Clasificación de opiniones como positivas, negativas o neutras.

### Extracción de Entidades:

* Identificación de nombres, lugares, fechas, etc., en un texto.

### Respuesta a Preguntas:

* Dado un contexto y una pregunta, BERT puede encontrar la respuesta en el texto.

### Clasificación de Texto:

* Asignación de etiquetas o categorías a documentos.

### Traducción o Parafraseo:

* Mejora las tareas de transformación del texto.

## Ventajas

### Precisión elevada:

* Ofrece representaciones de palabras más ricas y precisas gracias a su naturaleza bidireccional.

### Adaptabilidad:

* Puede ser ajustado para tareas específicas mediante fine-tuning en datasets más pequeños.

### Amplio soporte:

* Compatible con bibliotecas populares como TensorFlow y PyTorch.

## Limitaciones

### Costoso en recursos:

* Entrenar y ejecutar BERT requiere una cantidad considerable de memoria y potencia computacional.
* Sin GPU, las tareas complejas pueden ser lentas.

### Dependencia del corpus de pre-entrenamiento:

* Su comprensión del lenguaje depende del texto utilizado para entrenarlo inicialmente.