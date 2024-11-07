# Text data preprocessing and setup for Word2Vec style embeddings
import string
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Lambda, Dense
from tensorflow.keras.models import Sequential
import seaborn as sns
import matplotlib.pyplot as plt

# Input text
text = """
The speed of transmission is an important point of difference between the two viruses. 
Influenza has a shorter median incubation period (the time from infection to appearance of symptoms) 
and a shorter serial interval (the time between successive cases) than COVID-19 virus. 
The serial interval for COVID-19 virus is estimated to be 5-6 days, while for influenza virus, 
the serial interval is 3 days. This means that influenza can spread faster than COVID-19. 

Further, transmission in the first 3-5 days of illness, or potentially pre-symptomatic transmission 
–transmission of the virus before the appearance of symptoms – is a major driver of transmission for influenza. 
In contrast, while we are learning that there are people who can shed COVID-19 virus 24-48 hours prior to symptom onset, 
at present, this does not appear to be a major driver of transmission. 

The reproductive number – the number of secondary infections generated from one infected individual – 
is understood to be between 2 and 2.5 for COVID-19 virus, higher than for influenza. However, estimates for both 
COVID-19 and influenza viruses are very context and time-specific, making direct comparisons more difficult.  
"""

# Step 1: Tokenize text
dl_data = text.split()
text_data = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in dl_data]

# Tokenizer to convert words to indices
tk = Tokenizer()
tk.fit_on_texts(text_data)
w2idx = tk.word_index
idx2w = {v: k for k, v in w2idx.items()}
sentence = [w2idx.get(w) for w in text_data]

# Step 2: Prepare context and target pairs
target = []
context = []
context_size = 2

for i in range(context_size, len(sentence) - context_size):
    target.append(sentence[i])
    temp = sentence[i - context_size:i] + sentence[i + 1:i + 1 + context_size]
    context.append(temp)

x = np.array(context)
y = np.array(target)

# Model parameters
vocab_size = len(w2idx) + 1  # Adding 1 for padding token
embed_size = 100  # Embedding size

# Step 3: Define the neural network model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=2 * context_size),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

model.summary()

# Step 4: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(x, y, epochs=100, verbose=1)

# Step 6: Plot training history
sns.lineplot(data=history.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Epochs")
plt.show()

# Testing with new words
test_words = ["making", "direct", "comparisons", "more"]
test_indices = [w2idx.get(word) for word in test_words if word in w2idx]

# Reshape input for prediction
inp = np.array([test_indices])
pred = model.predict(inp)

# Output the most likely word in the prediction
predicted_index = pred.argmax()
print("Predicted word index:", predicted_index)
print("Predicted word:", idx2w.get(predicted_index, "Unknown"))

