import pickle
from rdkit import Chem
import sys
import requests
import time
from multiprocessing import Pool
from bs4 import BeautifulSoup, Comment
import pandas as pd
import nltk
from urllib.request import urlopen
from bs4 import BeautifulSoup
from requests_html import HTMLSession
from requests import get  
from urllib.request import urlopen


url = 'https://doi.org/10.1002/anie.201709125'

#response = get(url)               

f = urlopen(url)
myfile = f.read()
print(myfile)

