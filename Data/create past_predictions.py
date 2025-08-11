import polars as pl
from termcolor import colored
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
past_predictions = pl.read_csv("/Users/jacksaigusa/Downloads/UFCPredictor2025/Data/Predictions/dolidze_vs_hernandez_complete.csv")

past_predictions = past_predictions.drop("date")
print(past_predictions.head())
past_predictions = past_predictions.to_dicts()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///past_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db4 = SQLAlchemy(app)

class PastPrediction(db4.Model):
    id = db4.Column(db4.Integer, primary_key = True, index = True)
    Event = db4.Column(db4.String(100))
    
    FighterName = db4.Column(db4.String(100))
    OpponentName = db4.Column(db4.String(100))
    Prediction = db4.Column(db4.String(20))
    Confidence = db4.Column(db4.Float)
    Correct = db4.Column(db4.Boolean)


with app.app_context():
    #db4.drop_all() #<- uncomment for clearing and creating fresh database
    db4.create_all()
    


with app.app_context():
    try:
        for pred in past_predictions:
            past_prediction = PastPrediction(
                Event = pred['event'],
                FighterName = pred['fighter_name'],
                OpponentName = pred['opponent_name'],
                Prediction = pred['prediction'],
                Confidence = pred['confidence'],
                Correct = pred["correct"]
            )
            db4.session.add(past_prediction)
        db4.session.commit()
    except Exception as e:
        db4.session.rollback()
        print(colored(f"Error storing predictions: {e}", "red"))
    finally:
        db4.session.close()