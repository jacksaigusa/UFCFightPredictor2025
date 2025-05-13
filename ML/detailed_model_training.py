import csv
import os
import json
import polars as pl
import sys
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import log_loss
import optuna
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
data = pl.read_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/detailed_fights.csv")

# fixing dates if we ever want to filter out old fights
reference_date = pl.date(1970, 1, 1)
data = data.with_columns(
    pl.col("Date").str.to_datetime("%B %d, %Y").alias("Date")
)
data = data.with_columns(
    (pl.col("Date")-reference_date).dt.total_days().alias("Date")
)
print(data["Date"].head())
enc = LabelEncoder()
data = data.with_columns(
        pl.col("Result").map_batches(enc.fit_transform).alias("Result")
)
to_drop = ["Date", "Title", "Red Fighter", "Blue Fighter"]
data = data.drop(to_drop)
selected_columns = data.columns

print(f"selected columns: {selected_columns}")
print(data.select(selected_columns).dtypes)
data = data.select(selected_columns).drop_nans()
data = data.select(selected_columns).drop_nulls()

'''def prune_features(selected_columns):
        corr_matrix = data.select(selected_columns).fill_null(0).corr()
        upper_tri = corr_matrix.filter(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        data.drop(to_drop)
        selected_columns = [column for column in selected_columns if column not in to_drop]
        return selected_columns

selected_columns = prune_features(selected_columns)'''
data = data[selected_columns]
X = data.drop("Result")
x_cols = X.columns
y = data["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.select(x_cols).to_numpy()


model = lgb.LGBMClassifier()


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy}")




