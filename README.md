MMA Fight Predictor
A machine learning system that predicts UFC fight outcomes using historical fighter statistics and performance metrics.

📌 Overview
This project scrapes, processes, and analyzes UFC fighter data to:

Build predictive models for fight outcomes

Track historical model performance

Provide predictions for upcoming fights

Best Performing Model: RandomForest Classifier with 70.44% accuracy on a test set of 2,138 fights

🛠️ Project Structure
Data Pipeline
text
WebScrape/
├── scrape_fighter_stats.py       # Scrapes all fighter stats from ufcstats.com (1993-present)
├── scrape_new_fights.py          # Scrapes upcoming event data
└── allfighters.txt               # Output of scraped fighter data

ML/
├── process_fights.py             # Creates training data by matching fighter stats with historical fights
├── model_training.py             # Main model training and testing
├── weight_class_model_training.py # Weight-class specific models (+1% improvement)
├── neural_net_training.py        # (Experimental) Neural network approach
└── predict_new_fights.py         # Generates predictions for upcoming fights
Data Storage
text
instance/
├── detailed_fighter_stats.db     # SQLite database of all fighter stats
└── elo_fightstats.db            # Processed fight data for model training
Web Application
text
WebApp/                           # Full-stack prediction platform
├── backend/                      # Python Flask API
└── frontend/                     # React.js interface
🔍 Key Findings
Temporal Analysis: Models trained on pre-2010 data performed slightly worse, contrary to the hypothesis that recent data would be more predictive

Weight Class Specialization: Weight-class specific models improved accuracy by ~1%

Model Comparison: RandomForest outperformed neural network approaches

🚀 Live Application
View predictions and model metrics at:
🌐 https://mmamath.netlify.app/

🔄 Workflow
Data Collection:

bash
python WebScrape/scrape_fighter_stats.py
python WebScrape/scrape_new_fights.py
Data Processing:

bash
python ML/process_fights.py
Model Training:

bash
python ML/model_training.py
# or for weight-class models:
python ML/weight_class_model_training.py
Generate Predictions:

bash
python ML/predict_new_fights.py
📈 Key Metrics
Test set size: 2,138 fights

Baseline accuracy: 50%

Model accuracy: 70.44% (RandomForest)

Weight-class models: ~71.5% accuracy

📚 Data Sources
All data scraped from ufcstats.com, covering:

Fight results from 1993 to present

650+ fighter statistics

50+ performance metrics per fighter

🤝 Contributing
This project welcomes contributions. Please open an issue to discuss potential improvements.