from __future__ import print_function
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import pandas as pd
import pickle
import threading
import sys
import os
os.chdir("C:\\Users\\dabel\\Documents\\Natural_Language_Processing_MPCS\\project")

df = pd.read_pickle("df_minutes.pickle")
df2 = df[df.index > '2006-01-01']

test = df2.statements[0]
test
test.replace('\n',' ').split('For immediate release')
df2 = df.copy()
df2.statements