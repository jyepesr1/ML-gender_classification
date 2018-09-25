#!/usr/bin/env python3

import fileinput
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

dv = DictVectorizer()
NBclassifier = MultinomialNB()
get_features = lambda name: {
    'first-letter': name[0],
    'first2-letters': name[0:2],
    'first3-letters': name[0:3],
    'last_letter': name[-1],
    'last2_letters': name[-2:],
    'last3_letters': name[-3:],
}

def predictor(name):
  global dv, NBclassifier, get_features
  test_name = [name.rstrip().lower()]
  transform_dv = dv.transform(get_features(test_name))
  vector = transform_dv.toarray()
  if NBclassifier.predict(vector) == 0:
    print("Female")
  else:
    print("Male")

def main():
  global dv, NBclassifier, get_features
  df = pd.read_csv('./datasets/names_dataset_medium.csv')
  df.drop(df[df.Sexo == 'A'].index, inplace = True)
  df.Sexo.replace({'F':0,'M':1},inplace=True)
  df.Nombre = df.Nombre.str.lower()
  get_features = np.vectorize(get_features)
  X = get_features(df.Nombre)
  y = df.Sexo
  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  #X_feat = dv.fit_transform(X_train)
  #print("Accuracy of Model {}{}".format(NBclassifier.score(dv.transform(X_test),y_test)*100,"%"))
  X_feat = dv.fit_transform(X)
  NBclassifier.fit(X_feat, y)
  #test = ['Cristina', 'Vanesa', 'Andrea', 'Raúl', 'Pedro', 'jose', 'Jose']

  print("Enter a name:")

  print("> ", end='')
  for line in fileinput.input():
    predictor(line)
    print("> ", end='')

if __name__ == "__main__":
    main()
