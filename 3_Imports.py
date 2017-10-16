import os
import requests

import zipfile
import numpy as np
from bs4 import BeautifulSoup


def download(url, dest_path):
    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, 'wb') as fd:
        for chunk in req.iter_content(chunk_size=2**20):
            fd.write(chunk)


def get_data(data_home, url, dest_subdir, dest_filename):

    data_dir = os.path.join(os.path.abspath(data_home), dest_subdir)

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    dest_path = os.path.join(data_dir, dest_filename)

    if not os.path.isfile(dest_path):
        download(url, dest_path)

    return dest_path


def _read_raw_data(path):
    with zipfile.ZipFile(path) as datafile:
        return(datafile.read('jester_ratings.dat').decode().split('\n'),
               datafile.read('jester_items.dat').decode().split('\n\n'))


def _parse_jokes(raw):
    from bs4 import BeautifulSoup

    jokes_dict = {}
    for line in raw:

        if not line:
            continue

        # line.strip()
        # print(line)
        (number, joke) = line.split|(':\n')
        # jokes_dict[number] = joke.rstrip()
        jokes_dict[int(number)] = BeautifulSoup(joke.rstrip(), 'lxml').text.strip('\r\n').replace('\r', '')

    return jokes_dict


def _parse_data(raw):
    # return [[(int(uid), int(iid), float(rating)) for (uid, iid, rating) in line.split('\t\t')] for line in raw if line]
    def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)

    return ([num(n) for n in line.split('\t\t')] for line in raw if line)


def fetch_jester():
    zip_path = get_data('data',
                        'http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip',
                        'jester',
                        'jester.zip')

    (data_raw, jokes_raw) = _read_raw_data(zip_path)

    jokes_dict = _parse_jokes(jokes_raw)
    data = _parse_data(data_raw)


fetch_jester()
