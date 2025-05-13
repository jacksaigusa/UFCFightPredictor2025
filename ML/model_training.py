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

#removing old fights to see if performance improves

reference_date = pl.date(1970, 1, 1)
data = data.with_columns(
    pl.col("date").str.to_datetime("%b. %d, %Y").alias("date")
)

data = data.with_columns(
    (pl.col("date")-reference_date).dt.total_days().alias("date")
)
print(f"dataset size before filtering out old fights: {data.shape}")

data = data.filter(pl.col("date") >= 14600)

print(f"dataset size after filtering out old fights: {data.shape}")
print(f"most recent fight in dataset occurred {data["date"].max()} days after jan 1st 1970")
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
        "opponent_titlewins",
        "opponent_elo",
        "opponent_opp_avg_elo"
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



'''data = data.drop("event")
#data = data.drop("date")
data = data.drop("fighter_name")
data = data.drop("opponent_name")
data = data.drop("time")
data = data.drop("fighter_weight")
data = data.drop("fighter_height")
data = data.drop("fighter_reach")
data = data.drop("fighter_record")
data = data.drop("opponent_weight")
data = data.drop("opponent_height")
data = data.drop("opponent_reach")
data = data.drop("opponent_record")'''



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

from xgboost import XGBClassifier
#pyarrow needed to use xgb with polars df
import pyarrow
from sklearn.ensemble import AdaBoostClassifier
# trying to predict multiple variables: result, method, and round
'''data = data.drop_nans()
data = data.drop_nulls()
X = data.drop(["result", "method", "round"])
y = data["result", "method", "round"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
'''
# multi output classifier wrapper if predicting multiple variables
#model = MultiOutputClassifier(RandomForestClassifier()).fit(X_train, y_train)
model = RandomForestClassifier()
#model = AdaBoostClassifier()
#model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
#accuracy metric is different for predicting multiple variables
'''def exact_match(y_true, y_pred):
    y_true_np = y_true.to_numpy()
    #y_pred_np = y_pred.to_numpy()
    matches = np.all(y_true_np == y_pred, axis=1)
    return np.mean(matches)'''

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










