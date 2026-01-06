import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from fredapi import Fred
import seaborn as sns

# --- SECTION 1: CONFIGURATION & DATA ACQUISITION ---
# Large universe of 100+ tickers
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "JNJ", "V", "UNH", "PG", "HD", "MA", "DIS", "BAC", "ADBE", "CRM", "NFLX", "PYPL", "INTC", "CMCSA", "PEP", "CSCO", "TMO", "AVGO", "ABT", "COST", "WMT", "MRK", "ACN", "MCD", "AMD", "TXN", "NKE", "LIN", "IBM", "ORCL", "QCOM", "CVX", "UPS", "MDT", "SBUX", "GS", "CAT", "BA", "HON", "AXP", "GE", "MMM", "INTU", "AMGN", "ISRG", "NOW", "BLK", "PLD", "LOW", "SCHW", "RTX", "DE", "UNP", "TJX", "SPGI", "LMT", "T", "VZ", "BMY", "GILD", "F", "C", "WFC", "PFE", "DUK", "SO", "NEE", "D", "SNOW", "DDOG", "CRWD", "NET", "SHOP", "ZM", "TEAM", "MDB", "OKTA", "ROKU", "DOCU", "PLTR", "ASML", "TSM", "ADSK", "SNPS", "CDNS", "KLAC", "LRCX", "AMAT", "FTNT", "CHTR", "BABA", "JD", "PDD", "TCEHY", "SAP", "NSRGY", "NVO", "NVS", "AZN", "SAN", "RY", "TD", "ENB", "BP", "SHEL", "TTE", "SIEGY", "SPY", "VOO", "IVV", "QQQ", "DIA", "IWM", "VTI", "BND", "VEA", "VWO", "VGK", "GLD", "SLV", "TLT", "LQD", "HYG", "VNQ", "XLK", "XLV", "XLF", "XLI", "XLP", "XLY", "XLE", "XLU", "IBB", "IYR", "ITB", "XHB", "PFF", "VIG", "VYM", "SCHD", "ARKK", "ICLN", "TAN", "LIT", "URA", "MJ", "ESPO", "BOTZ"]

end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days=3*365)

# Download data - adjusted for potential failed downloads in large lists
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
adj_close_df = data['Close'].ffill() # Ensure no gaps in large datasets

# Ensure our ticker list matches the downloaded columns (yfinance may drop some)
valid_tickers = adj_close_df.columns.tolist()

# --- SECTION 2: RETURN & RISK CALCULATIONS ---
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252

def standard_deviation(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

# --- SECTION 3: ECONOMIC DATA (FRED API) ---
fred = Fred(api_key='618a28f6794406e6316e79788f1f81b7')
ten_year_treasury_rate = fred.get_series('GS10') / 100
risk_free_rate = ten_year_treasury_rate.iloc[-1]

# --- SECTION 4: MEAN-VARIANCE OPTIMIZATION ---
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.15) for _ in range(len(valid_tickers))]
initial_weights = np.array([1/len(valid_tickers)] * len(valid_tickers))

optimized_results = minimize(neg_sharpe_ratio, initial_weights, 
                             args=(log_returns, cov_matrix, risk_free_rate), 
                             method='SLSQP', constraints=constraints, bounds=bounds)

optimal_weights = optimized_results.x
optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

# --- SECTION 5: BACKTESTING & BENCHMARKING ---
benchmark = "SPY"
benchmark_data = yf.download(benchmark, start=start_date, end=end_date, auto_adjust=True)['Close']
if isinstance(benchmark_data, pd.DataFrame):
    benchmark_data = benchmark_data.iloc[:, 0]

spy_daily_return = benchmark_data.pct_change().dropna()
portfolio_daily_return = adj_close_df.pct_change().dropna().dot(optimal_weights)
portfolio_cum = (1 + portfolio_daily_return).cumprod()
spy_cum = (1 + spy_daily_return).cumprod()

total_port_return = (portfolio_cum.iloc[-1] - 1) * 100
total_spy_return = (spy_cum.iloc[-1] - 1) * 100
alpha_gen = total_port_return - total_spy_return

# --- SECTION 6: ANNUALISED RISK ANALYSIS ---
def calculate_annual_sortino(returns_series, rf_rate_annual):
    annual_return = returns_series.mean() * 252
    downside_returns = returns_series[returns_series < 0]
    if len(downside_returns) < 2:
        return np.nan
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    return (annual_return - rf_rate_annual) / downside_deviation

daily_port_series = pd.Series(portfolio_daily_return, index=adj_close_df.index[1:])
annual_sortinos = daily_port_series.groupby(daily_port_series.index.year).apply(
    lambda x: calculate_annual_sortino(x, risk_free_rate)
)

# --- SECTION 7: MONTE CARLO SIMULATION ---
num_simulations = 10000
num_days = 252 
port_mean_daily = log_returns.mean() @ optimal_weights
port_std_daily = standard_deviation(optimal_weights, cov_matrix) / np.sqrt(252)
simulated_daily_returns = np.random.normal(port_mean_daily, port_std_daily, (num_days, num_simulations))
sim_growth_paths = np.cumprod(1 + simulated_daily_returns, axis=0)

final_values = sim_growth_paths[-1, :]
expected_final_value = np.mean(final_values)
percentile_5 = np.percentile(final_values, 5)
percentile_95 = np.percentile(final_values, 95)

# --- SECTION 8: PDF EXPORT WITH DYNAMIC SCALING ---
pdf_filename = "Portfolio_Analysis_Report.pdf"

with PdfPages(pdf_filename) as pdf:
    # PAGE 1: EXECUTIVE SUMMARY
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    summary_text = (
        f"PORTFOLIO OPTIMISATION EXECUTIVE SUMMARY\n"
        f"{'='*40}\n\n"
        f"Universe Size:                 {len(valid_tickers)} Assets\n"
        f"Risk-Free Rate (10Y Treasury): {risk_free_rate:.2%}\n"
        f"Expected Annual Return:        {optimal_portfolio_return:.2%}\n"
        f"Expected Annual Volatility:    {optimal_portfolio_volatility:.2%}\n"
        f"Portfolio Sharpe Ratio:        {optimal_sharpe_ratio:.2f}\n\n"
        f"[BENCHMARK VS. S&P 500]\n"
        f"Total Portfolio Return:        {total_port_return:.2f}%\n"
        f"Total SPY Return:              {total_spy_return:.2f}%\n"
        f"Alpha Generated:               {alpha_gen:.2f}%\n\n"
        f"[DOWNSIDE RISK]\n"
        f"Avg Annual Sortino Ratio:      {annual_sortinos.mean():.2f}\n\n"
        f"[1-YEAR MONTE CARLO FORECAST]\n"
        f"Expected Value of $1:          ${expected_final_value:.2f}\n"
        f"Value at Risk (5th Pct):       ${(1 - percentile_5):.2f} loss per $1\n"
        f"95% Confidence Interval:       ${percentile_5:.2f} - ${percentile_95:.2f}\n"
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11, family='monospace', verticalalignment='top')
    pdf.savefig(fig)
    plt.close()

    # PAGE 2: ASSET ALLOCATION (Filtered for significant weights)
    weight_series = pd.Series(optimal_weights, index=valid_tickers)
    significant_weights = weight_series[weight_series > 0.001].sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    significant_weights.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Significant Asset Allocation (>{0.001*100}%)", fontsize=14)
    plt.ylabel("Weighting (%)")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # PAGE 3: BACKTEST
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_cum, color='black', lw=2, label="Optimised Portfolio")
    plt.plot(spy_cum, color='red', alpha=0.7, label="S&P 500 (SPY)")
    plt.title("Historical Backtest: Growth of $1", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    pdf.savefig()
    plt.close()

    # PAGE 4: CORRELATION HEATMAP (Dynamic Scaling for 100+ Tickers)
    # Increase dimensions based on ticker count to prevent overlapping labels
    fig_dim = max(12, len(valid_tickers) // 6)
    plt.figure(figsize=(fig_dim, fig_dim * 0.8))
    
    # annot=False because 100x100 numbers would be unreadable
    sns.heatmap(log_returns.corr(), annot=False, cmap='coolwarm', 
                linewidths=0, xticklabels=True, yticklabels=True)
    
    # Adjust font size for labels based on total number of tickers
    label_font_size = max(2, 10 - (len(valid_tickers) // 20))
    plt.xticks(rotation=90, fontsize=label_font_size)
    plt.yticks(fontsize=label_font_size)
    plt.title(f"Correlation Heatmap: {len(valid_tickers)} Assets", fontsize=16)
    pdf.savefig()
    plt.close()

    # PAGE 5: MONTE CARLO SIMULATION
    plt.figure(figsize=(10, 6))
    plt.plot(sim_growth_paths[:, :100], color='royalblue', alpha=0.2)
    plt.axhline(1.0, color='red', linestyle='--', lw=1)
    plt.title("Monte Carlo: 10,000 Simulated Future Paths", fontsize=14)
    pdf.savefig()
    plt.close()


print(f"Successfully exported report to: {pdf_filename}")
