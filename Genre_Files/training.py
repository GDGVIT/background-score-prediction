import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

action = pd.read_csv('action.csv')
comedy = pd.read_csv('comedy.csv')
horror = pd.read_csv('horror.csv')
romantic = pd.read_csv('romantic.csv')

df = pd.concat([action, comedy, horror, romantic], ignore_index = False)

df['Genre'].unique()
# array([1, 2, 0, 3])  0 -> Horror 1 -> Action 2 -> Comedy 3 -> Romantic

df['Genre'].value_counts()
# 3    106
# 2    106
# 1    106
# 0     97
# Name: Genre, dtype: int64


X = df.iloc[:, 0 : 9].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rfc = RandomForestClassifier(n_estimators = 100, random_state = 0)

rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)

print(accuracy_score(y_test, predictions))

filename = 'pre-processing.sav'
pickle.dump(sc, open(filename, 'wb'))

filename = 'finalized_model.sav'
pickle.dump(rfc, open(filename, 'wb'))