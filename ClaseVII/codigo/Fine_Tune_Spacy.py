import spacy
import random
import pandas as pd
from spacy.util import minibatch
from sklearn.model_selection import train_test_split
from spacy.training.example import Example
from spacy.pipeline.textcat import Config, ConfigSchema, ConfigChoice

# Cargar el modelo preentrenado de spaCy con transformers (BERT)
# Primero se debe descargar el archivo al disco!!
nlp = spacy.load('en_core_web_trf')

# Definir las clases para la clasificación (por ejemplo, Sentimientos: positivo y negativo)
text_classifier = nlp.add_pipe('textcat', last=True)  # Añadir una nueva capa de clasificación
text_classifier.add_label('POSITIVE')
text_classifier.add_label('NEGATIVE')

# Cargar tu dataset. Asumimos que tienes un CSV con 'text' y 'label' como columnas.
data = pd.read_csv('your_dataset.csv')

# Asegurarte de que tienes solo las clases necesarias
data = data[data['label'].isin(['POSITIVE', 'NEGATIVE'])]

# Dividir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2)

# Preparar los datos de entrenamiento y prueba en el formato que spaCy requiere
train_examples = []
for text, label in zip(train_data['text'], train_data['label']):
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, {'cats': {label: 1.0, 'POSITIVE' if label == 'NEGATIVE' else 'NEGATIVE': 0.0}})
    train_examples.append(example)

test_examples = []
for text, label in zip(test_data['text'], test_data['label']):
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, {'cats': {label: 1.0, 'POSITIVE' if label == 'NEGATIVE' else 'NEGATIVE': 0.0}})
    test_examples.append(example)

# Configuración del optimizador
optimizer = nlp.begin_training()

# Entrenamiento del modelo con minibatch
for epoch in range(5):  # Número de épocas de entrenamiento
    random.shuffle(train_examples)
    losses = {}
    # Utilizamos minibatch para evitar cargar todos los datos a la vez
    batches = minibatch(train_examples, size=8)
    for batch in batches:
        # Preparar los ejemplos para el entrenamiento
        for batch_example in batch:
            # Actualizar los gradientes con el modelo
            nlp.update([batch_example], drop=0.5, losses=losses)
    print(f"Epoch {epoch + 1}, Losses: {losses}")

# Evaluación del modelo
test_texts = [example.text for example in test_examples]
test_labels = [example.annotation['cats'] for example in test_examples]
correct_predictions = 0

for i, doc in enumerate(nlp.pipe(test_texts)):
    predicted = doc.cats
    actual = test_labels[i]
    if predicted['POSITIVE'] > predicted['NEGATIVE'] and actual['POSITIVE'] == 1:
        correct_predictions += 1
    elif predicted['NEGATIVE'] > predicted['POSITIVE'] and actual['NEGATIVE'] == 1:
        correct_predictions += 1

accuracy = correct_predictions / len(test_examples)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
nlp.to_disk("sentiment_model_spacy")

