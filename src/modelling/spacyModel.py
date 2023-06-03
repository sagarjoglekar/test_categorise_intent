import spacy
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class IntentClassifierSpacy:

    def __init__(self, intents, best_params=None):
        self.nlp = spacy.load('en_core_web_md')
        if not best_params:
            self.mlp = MLPClassifier(random_state=42, early_stopping=True)
        else:
            self.mlp = MLPClassifier(**best_params, random_state=42, early_stopping=True)
        self.intents = intents
        self.intent_to_label = {intent: idx for idx, intent in enumerate(self.intents)}

    def fit(self, data, text_field='text', intent_field='intent'):
        # Convert intents to numerical labels
        data["labels"] = data[intent_field].apply(lambda x: self.intent_to_label[x])

        # Create Spacy embeddings
        data["spacy_embeddings"] = data[text_field].apply(lambda x: self.nlp(x).vector)

        # Train the MLP classifier
        self.mlp.fit(np.vstack(data['spacy_embeddings']), data['labels'])

    def predict(self, data, text_field='text'):
        # Create Spacy embeddings
        data["spacy_embeddings"] = data[text_field].apply(lambda x: self.nlp(x).vector)

        # Make predictions
        predicted_labels = self.mlp.predict(np.vstack(data['spacy_embeddings']))
        predicted_proba = self.mlp.predict_proba(np.vstack(data['spacy_embeddings']))
        predicted_intents = [self.intents[label] for label in predicted_labels]
        return predicted_labels, predicted_proba, predicted_intents

    def predict_one(self, text):
        # Create Spacy embedding
        embedding = self.nlp(text).vector.reshape(1, -1)

        # Make a prediction
        predicted_label = self.mlp.predict(embedding)[0]
        predicted_proba = self.mlp.predict_proba(embedding)
        predicted_intent = self.intents[predicted_label]

        return predicted_label, predicted_proba, predicted_intent

    def evaluate(self, data, text_field='text', intent_field='intent'):
        # Convert intents to numerical labels
        data["labels"] = data[intent_field].apply(lambda x: self.intent_to_label[x])

        # Make predictions
        pred, _proba, _intent = self.predict(data, text_field)

        # Evaluate the model
        return classification_report(data["labels"], pred, target_names=self.intents, output_dict=True)

    def tune_hyperparameters(self, data, param_grid, text_field='text', intent_field='intent'):
        # Convert intents to numerical labels
        self.intents = data[intent_field].unique().tolist()
        self.intent_to_label = {intent: idx for idx, intent in enumerate(self.intents)}

        # Convert intents to numerical labels
        data["labels"] = data[intent_field].apply(lambda x: self.intent_to_label[x])
        data["spacy_embeddings"] = data[text_field].apply(lambda x: self.nlp(x).vector)

        # Perform grid search
        grid_search = GridSearchCV(self.mlp, param_grid, cv=5, n_jobs=-1, verbose=3)
        grid_search.fit(np.vstack(data['spacy_embeddings']), data['labels'])

        # Update the MLP classifier with the best parameters
        self.mlp = grid_search.best_estimator_

        # Print the best hyperparameters
        print("Best hyperparameters:", grid_search.best_params_)
        print("Best accuracy:", grid_search.best_score_)
