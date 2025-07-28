import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
data = pl.read_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/elofightstats5122025.csv")

#removing old fights to see if performance improves: it doesnt, slightly worse

reference_date = pl.date(1970, 1, 1)
data = data.with_columns(
    pl.col("date").str.to_datetime("%b. %d, %Y").alias("date")
)

data = data.with_columns(
    (pl.col("date")-reference_date).dt.total_days().alias("date")
)

#removing missing weight values
data = data.with_columns(
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
)

print(data["fighter_weight"].head())
# Creating new column for weight class. 
#   115-135: lightest, encoded as 0
#   145-155: light, encoded as 1
#   170-185: mid, encoded as 2
#   205-265+: heavy, encoded as 3
data = data.with_columns(
    pl.when(
        (pl.col("fighter_weight") != -1) & (pl.col("fighter_weight") >= 115) & (pl.col("fighter_weight") <= 136)
    ).then(0)  # lightest
    .when(
        (pl.col("fighter_weight") != -1) & (pl.col("fighter_weight") > 136) & (pl.col("fighter_weight") <= 156)
    ).then(1)  # light
    .when(
        (pl.col("fighter_weight") != -1) & (pl.col("fighter_weight") > 156) & (pl.col("fighter_weight") <= 186)
    ).then(2)  # mid
    .when(
        (pl.col("fighter_weight") != -1) & (pl.col("fighter_weight") > 186) & (pl.col("fighter_weight") <= 266)
    ).then(3)  # heavy
    .otherwise(-1)
    .alias("weight_class")
)
print(data["weight_class"].head())



print(f"average fighter weight: {data["fighter_weight"].mean()}")
print(f"average opponent weight: {data["opponent_weight"].mean()}")

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
    "opponent_age"
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

# save the data for later use
#data.write_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/elo_training_data.csv")
enc = LabelEncoder()
data = data.with_columns(
    pl.col("result").map_batches(enc.fit_transform).alias("result")
)
#encode other variables if you want to predict method and round as well
'''data = data.with_columns(
    pl.col("method").map_batches(enc.fit_transform).alias("method")
)
data = data.with_columns(
    pl.col("round").map_batches(enc.fit_transform).alias("round")
)'''


X = data.drop("result")
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
# uncomment 5 lines below to predict multiple variables: result, method, and round
'''data = data.drop_nans()
data = data.drop_nulls()
X = data.drop(["result", "method", "round"])
y = data["result", "method", "round"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
'''
# uncomment line below to try multi output classifier wrapper if predicting multiple variables
#model = MultiOutputClassifier(RandomForestClassifier()).fit(X_train, y_train)
model = RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=1, max_features='log2', max_depth=10, bootstrap=True)


model.fit(X_train, y_train)



y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#accuracy metric is different for predicting multiple variables. Uncomment and use function below to asses multi output accuracy
'''def exact_match(y_true, y_pred):
    y_true_np = y_true.to_numpy()
    #y_pred_np = y_pred.to_numpy()
    matches = np.all(y_true_np == y_pred, axis=1)
    return np.mean(matches)'''
print("MODEL TRAINED ON FIGHTS OF ALL WEIGHT CLASSES\n")
print(f"Train set size: {X_train.shape}")
print(f"Test set size: {y_test.shape}")
print(f"Accuracy: {accuracy}")

# feature importances 

feature_importances = model.feature_importances_

feature_importance_df = pl.DataFrame(
    {"Feature": X.columns, "Importance": feature_importances}
)

feature_importance_df = feature_importance_df.sort("Importance", descending=True)

plt.figure(figsize=(15, 10))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances of RandomForest Classifier Features")
plt.show()

# the RandomForest classifier, trained on all fight data from 1993-present, is the most accurate, with an accuracy score of 0.70 on test set of size 2138 fights.



num_right = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        num_right += 1

print(f"\nNUMBER OF CORRECT PREDICTIONS FOR MODEL TRAINED ON ALL WEIGHT CLASSES: {num_right}")








