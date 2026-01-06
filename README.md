# AlphaStream Portfolio Engine
A python project which uses mean variance optimisation to return optimal capital allocation to each asset and risk management methods. Integrates yfinance and FRED API to optimise the Sharpe Ratio for selected assets. Includes correlation heatmaps, Sortino analysis and 10,000 iteration simulations to forecast VaR, which are all exported onto a PDF.

# Getting Started
The engine requires a Python 3.8+ environment. To ensure the mathematical solvers and data scrapers function correctly, install the following Python libraries:
> pip install yfinance pandas numpy scipy matplotlib seaborn fredapi
- yfinance: Utilised to scrape historical market data directly from Yahoo Finance. It serves as the primary data ingestion tool, pulling "Adjusted Close" prices which account for corporate actions like dividends and stock splits.
- pandas: Acts as the core data manipulation engine. It is used to structure raw price data into Time-Series DataFrames, handle missing values via forward-filling (ffill), and resample data for annualised risk metrics.
- NumPy: Provides high-performance mathematical functions for vectorised matrix operations. It is essential for calculating the portfolio variance ($w^T \Sigma w$) and generating the random normal distributions required for Monte Carlo simulations.
- SciPy (optimize): Specifically the minimize function using the SLSQP algorithm. It functions as the "solver" that iterates through thousands of weight combinations to find the mathematically optimal allocation that maximises the Sharpe Ratio.
- fredapi: Facilitates a direct connection to the Federal Reserve Economic Data (FRED). It is used to pull the most recent 10-Year Treasury Yield, ensuring the "Risk-Free Rate" in the Sharpe and Sortino calculations is based on real-time macroeconomic data.
- matplotlib: The primary plotting library used to generate the visual components of the project, including cumulative growth charts and bar graphs. It also powers the PdfPages backend to automate the export of the multi-page executive report.
- seaborn: A statistical data visualisation layer built on top of Matplotlib. It is used to generate the Correlation Heatmap, leveraging its ability to dynamically scale and colour-code complex relationship matrices between 100+ assets.

# Methodology \& Key Features
This project identifies the Tangency Portfolio on the Efficient Frontier. It works by maximising the Sharpe Ratio, balancing the expected returns against the volatility to achieve the highest risk-adjusted returns.
