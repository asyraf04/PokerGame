
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

df = pd.read_csv('poker-hand-training.csv')

X3 = df.iloc[:, :6]
y = df.iloc[:, -1]

X4 = df.iloc[:, :8]
y = df.iloc[:, -1]

X_train_3, X_test_3, y_train, y_test = train_test_split(X3, y, test_size=0.2, random_state=42)
X_train_4, X_test_4, y_train, y_test = train_test_split(X4, y, test_size=0.2, random_state=42)

model3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model3.fit(X_train_3, y_train)

pickle.dump(model3, open('poker-model3.sav', 'wb'))

model4 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model4.fit(X_train_4, y_train)

pickle.dump(model4, open('poker-model4.sav', 'wb'))
