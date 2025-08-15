from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

# load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save the model using joblib
joblib.dump(model, 'Computer_vision/Packaging_model/rf_model.pkl')