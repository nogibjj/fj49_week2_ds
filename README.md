# Python - Basic descriptive statistics using pandas library


[![CiCd](https://github.com/nogibjj/fj49_week2_ds/actions/workflows/cicd.yml/badge.svg)](https://github.com/nogibjj/fj49_week2_ds/actions/workflows/cicd.yml)

# Spotify Data Analysis

This Python project analyzes data from the Spotify API, which is stored in a CSV file named `spotify.csv`. It provides insights into song lengths and identifies the top 10 artists with the most chart-topping hits between 2010 and 2022.

## Features

Descriptive statistics on song lengths (in milliseconds) to showcase the variation:

- `Mean = 226033`
- `Median = 221653`
- `Mode = 236133`
- `Std = 42063`

Visualization of the top 10 artists with the most chart-topping hits.

Here is the visualization:

<img width="1580" alt="Screenshot 2023-09-10 at 7 11 13 PM" src="https://github.com/nogibjj/fj49_week2_ds/assets/101464414/cfc958df-4041-4c8f-be86-ab6885a69074">




## CI/CD Integration

This repository is integrated with a CI/CD template for automatic deployment of Python projects within a virtual environment. 

You can find the template [here] (https://github.com/farazjawedd/python-template-ids706). Feel free to use the template for other projects!

## Development Environment

- The repository includes a `.devcontainer` folder with configurations for the VS Code remote container development environment.
- The `.github/workflows/cicd.yml` file defines the Continuous Integration (CI) workflow using GitHub Actions.

Explore the code and data to gain insights into the world of music with Spotify! 

1. Data Collection and Preparation
Steps:

Collect Data: Gather synchronized minute-by-minute price data for RTY, SPX, and TY.
Data Cleaning: Handle missing values, outliers, and ensure time alignment across all datasets.
Data Segmentation: Extract the data specifically between 9 AM and 10 AM for each trading day for all three instruments.
2. Exploratory Data Analysis (EDA)
Steps:

Visualization: Plot the price series and returns for RTY, SPX, and TY.
Descriptive Statistics: Calculate key statistics (mean, median, std, skewness, kurtosis) for the price changes and returns of each instrument.
Correlation Analysis: Compute and visualize the correlation matrix between RTY, SPX, and TY returns.
3. Volatility and Stochastic Calculus Models
Steps:

GARCH Model: Apply the GARCH model to estimate and analyze volatility clustering in RTY returns.
Stochastic Differential Equations (SDEs): Model the price dynamics using SDEs, such as the Geometric Brownian Motion (GBM) or more complex jump-diffusion models.
Volatility Surface: Construct and analyze the implied volatility surface for RTY using option prices if available.
4. Cross-Instrument Normalization and Influence Analysis
Steps:

Normalization: Normalize RTY returns using SPX and TY returns. This can involve regression analysis or machine learning techniques to isolate the impact of SPX and TY on RTY.
Principal Component Analysis (PCA): Use PCA to identify the main drivers of returns and volatility across the three instruments.
Granger Causality Test: Perform Granger causality tests to determine if past values of SPX and TY returns can predict RTY returns.
5. Machine Learning and Pattern Recognition
Steps:

Feature Engineering: Create features such as lagged returns, rolling statistics, technical indicators, and volatility measures for RTY, SPX, and TY.
Clustering: Apply clustering algorithms (e.g., K-means, DBSCAN) to group similar price patterns.
Supervised Learning: Use classification algorithms (e.g., Random Forest, SVM) to predict significant price movements based on the engineered features.
Deep Learning: Implement LSTM or GRU neural networks to capture complex temporal dependencies in the data.
6. Statistical Testing and Validation
Steps:

Hypothesis Testing: Formulate and test hypotheses about the identified patterns using statistical tests (e.g., t-tests, ANOVA).
Cross-Validation: Use cross-validation techniques to evaluate the robustness and predictive power of the machine learning models.
7. Backtesting and Strategy Development
Steps:

Strategy Formulation: Develop trading strategies based on the identified patterns and predictive models.
Backtesting: Simulate the performance of these strategies on historical data to evaluate their profitability and risk.
Risk Management: Incorporate risk management techniques, such as stop-loss and position sizing, into the strategies.
Example Workflow
Here's an enhanced Python workflow incorporating the above steps:

python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Load data
data_rty = pd.read_csv('rty_futures_minute_data.csv', parse_dates=['timestamp'])
data_spx = pd.read_csv('spx_futures_minute_data.csv', parse_dates=['timestamp'])
data_ty = pd.read_csv('ty_futures_minute_data.csv', parse_dates=['timestamp'])

# Set timestamp as index
data_rty.set_index('timestamp', inplace=True)
data_spx.set_index('timestamp', inplace=True)
data_ty.set_index('timestamp', inplace=True)

# Extract data between 9 AM and 10 AM
data_rty_9_10 = data_rty.between_time('09:00', '10:00')
data_spx_9_10 = data_spx.between_time('09:00', '10:00')
data_ty_9_10 = data_ty.between_time('09:00', '10:00')

# Compute returns
data_rty_9_10['returns'] = data_rty_9_10['price'].pct_change()
data_spx_9_10['returns'] = data_spx_9_10['price'].pct_change()
data_ty_9_10['returns'] = data_ty_9_10['price'].pct_change()

# GARCH model for RTY volatility
garch_model = arch_model(data_rty_9_10['returns'].dropna(), vol='Garch', p=1, q=1)
garch_results = garch_model.fit(disp='off')
data_rty_9_10['garch_volatility'] = garch_results.conditional_volatility

# Granger Causality Test
granger_test_spx = grangercausalitytests(data_rty_9_10[['returns', 'spx_returns']].dropna(), maxlag=5)
granger_test_ty = grangercausalitytests(data_rty_9_10[['returns', 'ty_returns']].dropna(), maxlag=5)

# PCA for dimensionality reduction
features = data_rty_9_10[['returns', 'garch_volatility']].dropna().values
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Clustering
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(pca_features)
data_rty_9_10['cluster'] = np.nan
data_rty_9_10.loc[data_rty_9_10.dropna().index, 'cluster'] = clusters

# Supervised Learning
# Create labels based on significant price movements
data_rty_9_10['label'] = (data_rty_9_10['returns'].shift(-1) > 0.001).astype(int)

# Feature set and labels
X = data_rty_9_10[['returns', 'garch_volatility', 'spx_returns', 'ty_returns']].dropna()
y = data_rty_9_10['label'].loc[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
cross_val_scores = cross_val_score(clf, X, y, cv=5)
print("Cross-Validation Scores:", cross_val_scores)

# Plot clustered price patterns
plt.figure(figsize=(12, 8))
plt.scatter(data_rty_9_10.index, data_rty_9_10['price'], c=data_rty_9_10['cluster'])
plt.title('Clustered Price Patterns for RTY')
plt.show()
Conclusion
This comprehensive plan leverages advanced statistical techniques, machine learning, and stochastic calculus to discern patterns in RTY futures prices. By normalizing RTY data with SPX and TY, and using sophisticated models and algorithms, you can uncover complex dependencies and develop robust trading strategies.

