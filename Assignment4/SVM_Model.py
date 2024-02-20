from assignment_sample_dataset import scientific_facts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# Preprocess the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(scientific_facts)
y = [sentence.split()[-1] for sentence in scientific_facts]

# Train the SVM model
svm_model = svm.SVC()
svm_model.fit(X, y)

# Example prediction
input_sentence = "Plants produce oxygen through a process called"
input_vector = vectorizer.transform([input_sentence])
predicted_word = svm_model.predict(input_vector)

print(f"The predicted last word after '{input_sentence}' is: {predicted_word[0]}")
