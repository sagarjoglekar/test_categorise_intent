import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modelling.tfidfModel import TfidfModel
from modelling.spacyModel import IntentClassifierSpacy


def train_tfidf_model(data, text_field, label_field):
    model = TfidfModel(model_file=None)
    model.fit(data, text_field=text_field, label_field=label_field)
    return model


def train_spacy_model(data, text_field, label_field):
    intents = data[label_field].unique().tolist()
    model = IntentClassifierSpacy(intents=intents)
    model.fit(data, text_field=text_field, intent_field=label_field)
    return model


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Train intent classification model')

    # Add the command-line arguments
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('text_field', type=int, help='Field name corresponding to the text')
    parser.add_argument('label_field', type=int, help='Field name corresponding to the label')
    parser.add_argument('--output_dir', type=str, default='../models/', help='Output directory for model files')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the data from the CSV file
    data = pd.read_csv(args.csv_file , header=None)
    data.head()

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Train the model
    model_type = input("Enter the model type (1 for TfidfModel, 2 for Spacy Model): ")
    if model_type == '1':
        model = train_tfidf_model(train_data, args.text_field, args.label_field)
    elif model_type == '2':
        model = train_spacy_model(train_data, args.text_field, args.label_field)
    else:
        print("Invalid model type. Please choose either 1 or 2.")
        return

    # Make predictions on the validation set
    val_labels = model.predict(val_data[args.text_field])
    val_accuracy = accuracy_score(val_data[args.label_field], val_labels)
    print(f"Validation accuracy: {val_accuracy}")

    # Save the trained model
    output_file = os.path.join(args.output_dir, 'model.joblib')
    joblib.dump(model, output_file)
    print(f"Trained model saved to: {output_file}")


if __name__ == '__main__':
    main()
