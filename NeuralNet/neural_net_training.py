from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder
# accuracy of neural net with all fights: 0.5535714285714286
# accuracy of neural net after filtering out old fights: 0.48364153627311524

data = pl.read_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/elofightstats5122025.csv")

#removing old fights


reference_date = pl.date(1970, 1, 1)
data = data.with_columns(
    pl.col("date").str.to_datetime("%b. %d, %Y").alias("date")
)

data = data.with_columns(
    (pl.col("date")-reference_date).dt.total_days().alias("date")
)
#print(f"dataset size before filtering out old fights: {data.shape}")

#data = data.filter(pl.col("date") >= 14600)

#print(f"dataset size after filtering out old fights: {data.shape}")
#removing missing weight values
'''data = data.filter(
    pl.col("fighter_weight") != "--", 
    pl.col("opponent_weight") != "--"
)'''

'''data = data.with_columns(
    pl.when(pl.col("fighter_weight") != "--").then(pl.col("fighter_weight").str.slice(0, 3).cast(pl.Int32, strict=False)).otherwise(None).alias("fighter_weight")
)
data = data.with_columns(
    pl.when(pl.col("opponent_weight") != "--").then(pl.col("opponent_weight").str.slice(0, 3).cast(pl.Int32, strict=False)).otherwise(None).alias("opponent_weight")
)

data = data.with_columns(
    pl.col("fighter_weight").fill_null(-1)
)

data = data.with_columns(
    pl.col("opponent_weight").fill_null(-1)
)'''




'''print(f"average fighter weight: {data["fighter_weight"].mean()}")
print(f"average opponent weight: {data["opponent_weight"].mean()}")'''

data = data.drop_nans(subset=[
        "fighter_kd_differential",
        "fighter_str_differential",
        "fighter_td_differential",
        "fighter_sub_differential",
        "fighter_winstreak",
        "fighter_losestreak",
        "fighter_age_deviation",
        "fighter_titlefights",
        "fighter_titlewins",
        "fighter_elo",
        "fighter_opp_avg_elo",
        "opponent_kd_differential",
        "opponent_str_differential",
        "opponent_td_differential",
        "opponent_sub_differential",
        "opponent_winstreak",
        "opponent_losestreak",
        "opponent_age_deviation",
        "opponent_titlefights",
        "opponent_titlewins" 
    ]
)
data = data.drop_nulls(subset=[
    "date",
    "fighter_kd_differential",
    "fighter_str_differential",
    "fighter_td_differential",
    "fighter_sub_differential",
    "fighter_winstreak",
    "fighter_losestreak",
    "fighter_age_deviation",
    "fighter_titlefights",
    "fighter_titlewins",
    "fighter_elo",
    "fighter_opp_avg_elo",
    "opponent_kd_differential",
    "opponent_str_differential",
    "opponent_td_differential",
    "opponent_sub_differential",
    "opponent_winstreak",
    "opponent_losestreak",
    "opponent_age_deviation",
    "opponent_titlefights",
    "opponent_titlewins",
    "opponent_elo",
    "opponent_opp_avg_elo"
])

selected_columns = [
    "fighter_kd_differential",
    "fighter_str_differential",
    "fighter_td_differential",
    "fighter_sub_differential",
    "fighter_winstreak",
    "fighter_losestreak",
    "fighter_age_deviation",
    "fighter_titlefights",
    "fighter_titlewins",
    "fighter_elo",
    "fighter_opp_avg_elo",
    "opponent_kd_differential",
    "opponent_str_differential",
    "opponent_td_differential",
    "opponent_sub_differential",
    "opponent_winstreak",
    "opponent_losestreak",
    "opponent_age_deviation",
    "opponent_titlefights",
    "opponent_titlewins",
    "opponent_elo",
    "opponent_opp_avg_elo",
    "result",
    "fighter_age",
    "opponent_age",
]







#fixing dates

reference_date = pl.date(1970, 1, 1)
data = data.with_columns(
    pl.col("fighter_dob").str.to_datetime("%b %d, %Y", strict=False))

data = data.with_columns(
    (pl.col("fighter_dob")-reference_date).dt.total_days().alias("fighter_age")
)


data = data.with_columns(
    pl.col("opponent_dob").str.to_datetime("%b %d, %Y", strict=False))

data = data.with_columns(
    (pl.col("opponent_dob")-reference_date).dt.total_days().alias("opponent_age")
)


data = data[selected_columns]


enc = LabelEncoder()
data = data.with_columns(
    pl.col("result").map_batches(enc.fit_transform).alias("result")
)

print(f"size of data after dropping nans: {data.shape}")
data = data.drop_nans()
data = data.drop_nulls()
print(f"size of data after dropping nans: {data.shape}")
X = data.drop("result")
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

enc = LabelEncoder()
data = data.with_columns(
    pl.col("result").map_batches(enc.fit_transform).alias("result")
)


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