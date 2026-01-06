# AlphaStream Portfolio Engine
A python project which uses mean variance optimisation to return optimal capital allocation to each asset and risk management methods. Integrates yfinance and FRED API to optimise the Sharpe Ratio for selected assets. Includes correlation heatmaps, Sortino analysis and 10,000 iteration simulations to forecast VaR, which are all exported onto a PDF.

This engine pulls ticker data from Yahoo Finance and macroeconomic indicators from the St. Louis Fed to carry out Mean-Variance Optimisation (MVO) on a multi-asset portfolio.

# Project Motivation
Modern portfolios often suffer from "over-diversification" or hidden sector correlations that drive systemic risk. This project was developed to provide a systematic, data-driven approach to capital allocation that prioritises risk-adjusted returns over simple growth, removing human bias through mathematical optimisation and stochastic stress-testing.

# Getting Started
The engine requires a Python 3.8+ environment. To ensure the mathematical solvers and data scrapers function correctly, install the following Python libraries:
> pip install yfinance pandas numpy scipy matplotlib seaborn fredapi
## Library Rationale
- yfinance: Utilised to scrape historical market data directly from Yahoo Finance. It serves as the primary data ingestion tool, pulling "Adjusted Close" prices which account for corporate actions like dividends and stock splits.
- pandas: Acts as the core data manipulation engine. It is used to structure raw price data into Time-Series DataFrames, handle missing values via forward-filling (ffill), and resample data for annualised risk metrics.
- NumPy: Provides high-performance mathematical functions for vectorised matrix operations. It is essential for calculating the portfolio variance ($w^T \Sigma w$) and generating the random normal distributions required for Monte Carlo simulations.
- SciPy (optimize): Specifically the minimize function using the SLSQP algorithm. It functions as the "solver" that iterates through thousands of weight combinations to find the mathematically optimal allocation that maximises the Sharpe Ratio.
- fredapi: Facilitates a direct connection to the Federal Reserve Economic Data (FRED). It is used to pull the most recent 10-Year Treasury Yield, ensuring the "Risk-Free Rate" in the Sharpe and Sortino calculations is based on real-time macroeconomic data.
- matplotlib: The primary plotting library used to generate the visual components of the project, including cumulative growth charts and bar graphs. It also powers the PdfPages backend to automate the export of the multi-page executive report.
- seaborn: A statistical data visualisation layer built on top of Matplotlib. It is used to generate the Correlation Heatmap, leveraging its ability to dynamically scale and colour-code complex relationship matrices between 100+ assets.

## Additional Steps to Setup
- API Integration: Obtain a free API key from FRED and update SECTION 3 of the code.
- Define Universe: Modify the tickers list in SECTION 1.
- Run: Execute the script to perform the optimisation and generate the Portfolio_Analysis_Report.pdf.

# Methodology \& Key Features

## Mean-Variance Optimisation (MVO)
This project identifies the Tangency Portfolio on the Efficient Frontier. It works by maximising the Sharpe Ratio, balancing the expected returns against the volatility to achieve the highest risk-adjusted returns.

The formula for Sharpe Ratio is:
> $$Sharpe\ Ratio = \frac{E[R_p] - R_f}{\sigma_p}$$

Downside risk can also be found, which quantifies returns while avoiding loss by avoiding upside volatility.

The formula for Sortino Ratio is:
>$$Sortino\ Ratio = \frac{E[R_p] - R_f}{\sigma_{down}}$$

## Stochastic Risk Forecasting
Unlike static backtests, the engine executes a 10,000-path Monte Carlo Simulation. This generates a probabilistic "cloud" of outcomes, allowing users to visualise the Value at Risk (VaR) and 95% confidence intervals for 1-year forward-looking projections.

## Automated Reporting
The project concludes with the generation of a multi-page PDF Report. This includes:
- Dynamic Heatmaps: Scaled correlation matrices that remain legible even with 100+ tickers.
- Annual Sortino Analysis: Downside risk metrics resampled by calendar year.
- Backtesting: Comparative growth analysis against the S&P 500 benchmark.
- Stochastic Risk Forecasting: Visualises 95% confidence intervals for 1-year forward-looking projections.

# Personal Reflection
This project represents a significant milestone in my transition into quantitative finance. Having taught myself Python independently, I built this engine to apply theoretical financial concepts to a scalable, real-world technical framework.

Key learnings included:
- Mathematical Implementation: Translating the abstract math of the Efficient Frontier and Matrix Covariance into efficient, vectorised Python code.
- Data Integrity: Developing strategies to handle high-dimensional datasets, such as dynamic chart scaling and forward-filling missing data for 100+ tickers.
- Risk Perspective: Shifting from a "return-only" mindset to a "risk-adjusted" perspective, specifically understanding the nuances of the Sortino Ratio and Value at Risk (VaR).
- Communication: Learning how to condense complex stochastic outputs into an automated PDF that is legible for stakeholders and investment committees.

Future Steps to Implement
- Black-Litterman Model Integration: Combining historical data with subjective investor views.
- Transaction Cost Modelling: Factoring in bid-ask spreads and commissions for more realistic Alpha.
- ESG Integration: Filtering assets based on Environmental, Social, and Governance scores.
