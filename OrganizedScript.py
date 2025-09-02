#test for algorithm
import yfinance as yf

from dotenv import load_dotenv
import os

api_key = os.getenv("API_KEY")

# initializing ticker class for a single symbol index
sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period = "max")

print("=== S&P 500 Data Loaded ===")
print(f"Date range: {sp500.index[0]} to {sp500.index[-1]}")
print(f"Total days: {len(sp500)}")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import numpy as np

def fetch_sp500_news(api_key, query="S&P 500", page_size=20):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    return [article["title"] + ". " + article.get("description", "") for article in data.get("articles", [])]

def analyze_sentiments(news_list):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(news)["compound"] for news in news_list]
    return np.mean(scores) if scores else 0

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

sp500 = sp500.loc["1990-01-01":].copy()

print(f"\n=== Data After 1990 Filter ===")
print(f"Total days: {len(sp500)}")

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split = 100, random_state = 1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score
preds = model.predict(test[predictors])

import pandas as pd
preds = pd.Series(preds,index=test.index)

print(f"\n=== Initial Model Performance ===")
print(f"Test period: {test.index[0]} to {test.index[-1]}")
print(f"Precision: {precision_score(test['Target'], preds):.4f}")
print(f"Predictions - Up: {(preds == 1).sum()}, Down: {(preds == 0).sum()}")

def predict (train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

def backtest(data, model, predictors, start = 2500, step = 250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)

print(f"\n=== Backtesting Results (Basic Features) ===")
print(f"Total predictions: {len(predictions)}")
print(f"Predicted UP: {predictions['Predictions'].value_counts().get(1, 0)}")
print(f"Predicted DOWN: {predictions['Predictions'].value_counts().get(0, 0)}")
print(f"Precision: {precision_score(predictions['Target'], predictions['Predictions']):.4f}")
print(f"Actual UP days: {(predictions['Target'] == 1).sum()} ({(predictions['Target'] == 1).sum() / len(predictions) * 100:.2f}%)")

horizons = [2, 5, 60, 250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]

sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])

# Add sentiment analysis
sp500["sentiment"] = 0.0
NEWS_API_KEY = api_key
news_articles = fetch_sp500_news(NEWS_API_KEY)
today_sentiment = analyze_sentiments(news_articles)
sp500.loc[sp500.index[-1], "sentiment"] = today_sentiment

print(f"\n=== Sentiment Analysis ===")
print(f"Articles analyzed: {len(news_articles)}")
print(f"Today's sentiment score: {today_sentiment:.4f}")
print("Sample headlines:")
for i, article in enumerate(news_articles[:3]):
    print(f"  {i+1}. {article[:100]}...")

new_predictors.append("sentiment")

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = backtest(sp500, model, new_predictors)

print(f"\n=== Enhanced Model Performance (60% threshold) ===")
print(f"Total predictions: {len(predictions)}")
print(f"Predicted UP: {predictions['Predictions'].value_counts().get(1, 0)}")
print(f"Predicted DOWN: {predictions['Predictions'].value_counts().get(0, 0)}")
print(f"Precision when predicting UP: {precision_score(predictions['Target'], predictions['Predictions']):.4f}")

# Feature importance
model.fit(sp500[new_predictors].iloc[:-1], sp500["Target"].iloc[:-1])
feature_importance = pd.DataFrame({
    'feature': new_predictors,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n=== Feature Importance ===")
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Latest prediction with details
latest_data = sp500.iloc[[-1]]
latest_prediction_proba = model.predict_proba(latest_data[new_predictors])[0]
latest_prediction = model.predict(latest_data[new_predictors])[0]

print(f"\n=== TOMORROW'S PREDICTION ===")
print(f"Date: {latest_data.index[0]}")
print(f"Current Close: ${latest_data['Close'].values[0]:.2f}")
print(f"Probability UP: {latest_prediction_proba[1]:.4f}")
print(f"Probability DOWN: {latest_prediction_proba[0]:.4f}")
print(f"Prediction: {'📈 UP' if latest_prediction == 1 else '📉 DOWN'}")

# Recent performance
recent_predictions = predictions.tail(20)
recent_correct = (recent_predictions['Target'] == recent_predictions['Predictions']).sum()
print(f"\n=== Last 20 Days Performance ===")
print(f"Correct predictions: {recent_correct}/20 ({recent_correct/20*100:.1f}%)")
print(f"When predicted UP: {len(recent_predictions[recent_predictions['Predictions'] == 1])} times")
print(f"Correct UP predictions: {((recent_predictions['Predictions'] == 1) & (recent_predictions['Target'] == 1)).sum()}")