from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder
# accuracy of neural net with all fights: 0.5535714285714286
# accuracy of neural net after filtering out old fights: 0.48364153627311524

data = pl.read_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/elo_training_data.csv")

#removing old fights




print(f"dataset size before filtering out old fights: {data.shape}")

data = data.filter(pl.col("date") >= 14600)
print(f"dataset size after filtering out old fights: {data.shape}")
enc = LabelEncoder()
data = data.with_columns(
    pl.col("result").map_batches(enc.fit_transform).alias("result")
)

X = data.drop("result")
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=1000)


model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")'''

#trying new method using hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [1000, 1500, 2000],
    'early_stopping': [True],
    'validation_fraction': [0.1]
}

grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")

best_model = grid_search.best_estimator_