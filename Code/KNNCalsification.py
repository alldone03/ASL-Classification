import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"C:\Users\Aldan\Desktop\ASLClassification\hand_features.csv")
X = df.drop('label', axis=1)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))