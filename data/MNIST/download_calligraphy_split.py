import requests
import zipfile
import os


def download_file_from_google_drive(id, destination):
    URL = 'https://drive.google.com/uc?id=10XH6uRWzIxIgbqjmz39HLUiRqh8tH4lu'
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_file(path_to_zip_file, destination):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination)


def cleanup(path_to_zip_file):
    os.remove(path_to_zip_file)


def main():
    file_id = '10XH6uRWzIxIgbqjmz39HLUiRqh8tH4lu'
    path_to_zip_file = './data/MNIST/7_calligraphy_split.zip'
    destination = './data/MNIST/test/'
    download_file_from_google_drive(file_id, path_to_zip_file)
    unzip_file(path_to_zip_file, destination)
    cleanup(path_to_zip_file)


if __name__ == "__main__":
    main()
