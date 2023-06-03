import requests

def download_file(url, local_path):
    """
    Download model files for spacy and tfidf
    Args:
        url (str): The URL of the file to download.
        local_path (str): The local file path to save the downloaded file.
    """
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, 'wb') as file:
        file.write(response.content)

if __name__ == '__main__':
    # Define the HTTPS URLs for the train and test datasets
    spacy_model = 'https://www.dropbox.com/s/a5x3epz6njqh42y/spacy_classifier.joblib?dl=0'
    tfidf_model = 'https://www.dropbox.com/s/0e109y5di4z326m/TfIdfClassifier.joblib?dl=0'

    # Define the local file paths to save the downloaded files
    spacy_local_path = 'models/spacy_model.joblib'
    tfidf_local_path = 'data/tfidf_model.joblib'

    print(f'Downloading models')
    download_file(spacy_model, spacy_local_path)
    print('Spacy model downloaded successfully!')

    download_file(tfidf_model, tfidf_local_path)
    print('TfIDF model downloaded successfully!')
