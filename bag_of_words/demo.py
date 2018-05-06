from sklearns.externals import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy

# Separating 10 samples for presenting a demo

df = pd.read_csv('title_category_table.csv')

df = df.sample(frac=1)

df = df[0:10]

df.to_csv('demo.csv')
