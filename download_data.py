import requests

def download_file(url, local_path):
    """
    Download a files for train and test

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
    train_data_url = 'https://s3-eu-west-1.amazonaws.com/adthena-ds-test/trainSet.csv'
    test_data_url = 'https://s3-eu-west-1.amazonaws.com/adthena-ds-test/candidateTestSet.txt'

    # Define the local file paths to save the downloaded files
    train_data_local_path = 'data/train_data.csv'
    test_data_local_path = 'data/test_data.csv'

    # Download the train dataset
    print(f'Downloading train data from {train_data_url}...')
    download_file(train_data_url, train_data_local_path)
    print('Train data downloaded successfully!')

    # Download the test dataset
    print(f'Downloading test data from {test_data_url}...')
    download_file(test_data_url, test_data_local_path)
    print('Test data downloaded successfully!')
