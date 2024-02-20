from assignment_sample_dataset import scientific_facts
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Preprocess the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(scientific_facts).toarray()

words = set(word for sentence in scientific_facts for word in sentence.split())

word_to_id = {word: i for i, word in enumerate(words)}
id_to_word = {i: word for word, i in word_to_id.items()}

y = [word_to_id[sentence.split()[-1]] for sentence in scientific_facts]
y = to_categorical(y, num_classes=len(words))

# Initialize and train the neural network model
model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='relu'))  # 10 neurons in the first layer
model.add(Dense(len(words), activation='softmax'))  # Output layer with one neuron per unique word

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

# Example prediction
input_sentence = "An herbivore is an animal that eats only"
input_vector = vectorizer.transform([input_sentence]).toarray()
predicted_probabilities = model.predict(input_vector)
predicted_word_id = int(predicted_probabilities.argmax())

if predicted_word_id in id_to_word:
    predicted_word = id_to_word[predicted_word_id]
    print(f"The predicted last word after '{input_sentence}' is: {predicted_word}")
else:
    print(f"Predicted ID {predicted_word_id} not found in ID to Word Mapping.")
