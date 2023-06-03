import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


class TfidfModel:
    def __init__(self, model_file: str):
        self.train_mode = False
        # These params were found using grid search separately
        self.best_params = {'clf__C': 1, 'tfidf__max_features': 10000, 'tfidf__ngram_range': (1, 1)}

        if not model_file:
            self.train_mode = True
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=self.best_params['tfidf__max_features'],
                                          ngram_range=self.best_params['tfidf__ngram_range'])),
                ('clf', LinearSVC(C=self.best_params['clf__C'], class_weight='balanced'))
            ])
        else:
            self.model_file = model_file
            try:
                self.model = joblib.load(self.model_file)
            except FileNotFoundError:
                print("Model file not found")

    def predict(self, series: pd.Series):
        return self.model.predict(series)

    def predict_proba(self, series: pd.Series):
        return self.model.predict_proba(series)

    def predict_proba_one(self, text: str):
        proba = self.model.predict_proba([text])[0]
        return max(proba)

    def predict_one(self, text: str):
        intent = self.model.predict([text])[0]
        return intent

    def fit(self, df, text_field, label_field):
        self.pipeline.fit(df[text_field], df[label_field])
