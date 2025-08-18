MMA Fight Prediction System
A comprehensive machine learning system for predicting UFC fight outcomes using historical fighter statistics and advanced modeling techniques.
🥊 Project Overview
This project scrapes fighter statistics from UFCStats.com, processes historical fight data, and uses machine learning models to predict fight outcomes. The system achieved 70.44% accuracy on a test set of 2,138 fights using a RandomForest classifier with weight class-specific modeling.
Live Web App: mmamath.netlify.app
📊 Performance Metrics

Best Model: RandomForest Classifier with Weight Class Specialization
Accuracy: 70.44% on test set (2,138 fights)
Improvement: ~1% performance gain with weight class-specific models
Data Coverage: Fighter statistics from 1993-present

🏗️ Project Structure
├── WebScrape/
│   ├── scrape_fighter_stats.py    # Scrapes fighter stats from UFCStats.com
│   └── scrape_new_fights.py       # Scrapes upcoming fight data
├── ML/
│   ├── process_fights.py          # Processes historical fight data
│   ├── model_training.py          # Main ML model training and testing
│   ├── neural_net_training.py     # Neural network experiments
│   ├── weight_class_model_training.py  # Weight class-specific models
│   └── predict_new_fights.py      # Generates predictions for upcoming fights
├── WebApp/                        # Full-stack web application
│   ├── backend/                   # Python Flask backend
│   └── frontend/                  # React.js frontend
├── instance/
│   ├── detailed_fighter_stats.db  # Fighter statistics database
│   ├── elo_fightstats.db          # Processed fight data
│   └── predictions.db             # Fight predictions database
├── allfighters.txt                # Fighter data output
└── elofightstats5122025.csv      # Training dataset
🔧 Installation & Setup
Prerequisites
bashpip install pandas scikit-learn sqlite3 requests beautifulsoup4 flask
npm install  # For React frontend
Database Setup

Run scrape_fighter_stats.py to populate the fighter statistics database
Execute process_fights.py to create the processed fight dataset
Use model_training.py to train and evaluate models

📈 Data Pipeline
1. Data Collection

scrape_fighter_stats.py: Scrapes comprehensive fighter statistics from UFCStats.com (1993-present)
Output: detailed_fighter_stats.db SQLite database and allfighters.txt

2. Data Processing

process_fights.py: Matches fighter statistics with historical fight outcomes
Output: elo_fightstats.db database and elofightstats5122025.csv training file

3. Model Training

model_training.py: Primary model training and evaluation
weight_class_model_training.py: Weight class-specific model optimization
neural_net_training.py: Neural network experiments (lower performance)

4. Prediction Generation

scrape_new_fights.py: Collects upcoming fight data
predict_new_fights.py: Generates predictions and updates prediction database

🧠 Model Architecture
RandomForest Classifier
The best-performing model uses a RandomForest classifier with the following key features:

Weight Class Specialization: Separate models trained for different weight divisions
Feature Engineering: Advanced fighter statistics including knockdown differentials, win streaks, and historical performance metrics
Cross-Validation: Robust testing methodology to prevent overfitting

Key Features

Fighter career statistics
Historical performance metrics
Weight class-specific attributes
Knockdown differentials (fighter_kd_differential)
Win/loss streaks and patterns

🔬 Research Findings
Temporal Analysis
Hypothesis: Removing older fight data would improve model performance due to MMA evolution.
Result: Model performance slightly decreased when trained only on fights from 2010+.
Insight: Historical data remains valuable, though sport evolution effects may still impact future predictions.
Weight Class Specialization
Hypothesis: Weight class-specific models would perform better due to different statistical importance across divisions.
Result: ~1% performance improvement confirmed.
Example: fighter_kd_differential has higher predictive value in heavyweight division compared to women's divisions due to knockout frequency differences.
🌐 Web Application
The full-stack web application provides:

Upcoming Fight Predictions: Real-time predictions for scheduled fights
Historical Performance: Model accuracy metrics and analysis
Interactive Visualizations: Fight outcome probabilities and trends

Technology Stack:

Backend: Python Flask
Frontend: React.js
Database: SQLite
Deployment: Netlify

📊 Usage Examples
Generate New Predictions
python# Scrape upcoming fights
python WebScrape/scrape_new_fights.py

# Generate predictions
python ML/predict_new_fights.py
Train New Models
python# Train standard models
python ML/model_training.py

# Train weight class-specific models  
python ML/weight_class_model_training.py
Update Fighter Database
python# Refresh fighter statistics
python WebScrape/scrape_fighter_stats.py

# Reprocess fight data
python ML/process_fights.py
🎯 Future Improvements

Real-time Model Updates: Implement continuous learning from new fight results
Advanced Feature Engineering: Incorporate fight camp information, injury history
Ensemble Methods: Combine multiple model types for improved accuracy
Betting Market Integration: Compare predictions against market odds
Style Matchup Analysis: Factor in fighting style compatibility

📝 Contributing

Fork the repository
Create a feature branch
Make your improvements
Add tests for new functionality
Submit a pull request

⚠️ Disclaimer
This system is for educational and research purposes only. Fight predictions should not be used for gambling or betting purposes. Past performance does not guarantee future results.
📄 License
This project is open source and available under the MIT License.