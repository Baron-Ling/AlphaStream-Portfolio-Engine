import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fredapi import Fred
import seaborn as sns

# --- SECTION 1: CONFIGURATION & DATA ACQUISITION ---
# Defining the split: 6 years total. 
today = dt.datetime.today()
year_3_ago = today - dt.timedelta(days=3*365)
year_6_ago = today - dt.timedelta(days=6*365)

# Combined Universe (Truncated for stability, but supports your 100+ list)
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "JNJ", "V", 
           "UNH", "PG", "HD", "MA", "DIS", "BAC", "ADBE", "CRM", "NFLX", "PYPL", "INTC"]

# Download data for full 6-year period
data = yf.download(tickers, start=year_6_ago, end=today, auto_adjust=True)['Close']
adj_close_df = data.ffill().dropna(axis=1)
valid_tickers = adj_close_df.columns.tolist()

# Split datasets for In-Sample (IS) and Out-of-Sample (OOS)
is_data = adj_close_df.loc[year_6_ago:year_3_ago]
oos_data = adj_close_df.loc[year_3_ago:today]

# --- SECTION 2: ECONOMIC DATA (FRED API) ---
fred = Fred(api_key='618a28f6794406e6316e79788f1f81b7')
ten_year_treasury_rate = fred.get_series('GS10') / 100
risk_free_rate = ten_year_treasury_rate.iloc[-1]

# --- SECTION 3: MVO OPTIMISATION (TRAINING ON YEARS 6 TO 3) ---
is_returns = np.log(is_data / is_data.shift(1)).dropna()
cov_matrix = is_returns.cov() * 252
mean_returns = is_returns.mean() * 252

def sharpe_ratio(weights, mean_returns, cov_matrix, rf):
    p_ret = np.sum(mean_returns * weights)
    p_std = np.sqrt(weights.T @ cov_matrix @ weights)
    return (p_ret - rf) / p_std

def neg_sharpe(weights, mean_returns, cov_matrix, rf):
    return -sharpe_ratio(weights, mean_returns, cov_matrix, rf)

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = [(0, 0.15) for _ in range(len(valid_tickers))]
init_weights = [1/len(valid_tickers)] * len(valid_tickers)

opt_results = minimize(neg_sharpe, init_weights, args=(mean_returns, cov_matrix, risk_free_rate), 
                       method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_results.x

# --- SECTION 4: OUT-OF-SAMPLE PERFORMANCE (TESTING ON LAST 3 YEARS) ---
oos_daily_returns = oos_data.pct_change().dropna()
portfolio_oos_returns = oos_daily_returns.dot(optimal_weights)

benchmark = yf.download("SPY", start=year_3_ago, end=today, auto_adjust=True)['Close']
spy_oos_returns = benchmark.pct_change().dropna().squeeze()

performance_df = pd.concat([portfolio_oos_returns.rename('Portfolio'), spy_oos_returns.rename('SPY')], axis=1).dropna()
performance_df['Port_Cum'] = (1 + performance_df['Portfolio']).cumprod()
performance_df['SPY_Cum'] = (1 + performance_df['SPY']).cumprod()

total_port_return = (performance_df['Port_Cum'].iloc[-1] - 1) * 100
total_spy_return = (performance_df['SPY_Cum'].iloc[-1] - 1) * 100
alpha_gen = total_port_return - total_spy_return

# --- SECTION 5: RISK ANALYSIS & MONTE CARLO ---
def calculate_annual_sortino(returns_series, rf):
    downside_returns = returns_series[returns_series < 0]
    if len(downside_returns) < 2: return np.nan
    downside_dev = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    return (returns_series.mean() * 252 - rf) / downside_dev

def calculate_annual_sharpe(returns_series, rf):
    return (returns_series.mean() * 252 - rf) / (returns_series.std() * np.sqrt(252))

# Calculate ratios for both portfolio and SPY
sortino_port = calculate_annual_sortino(performance_df['Portfolio'], risk_free_rate)
sortino_spy = calculate_annual_sortino(performance_df['SPY'], risk_free_rate)
sharpe_port = calculate_annual_sharpe(performance_df['Portfolio'], risk_free_rate)
sharpe_spy = calculate_annual_sharpe(performance_df['SPY'], risk_free_rate)

# Statistical Significance Testing (Two-sample t-test on daily returns)
t_stat, p_value = stats.ttest_ind(performance_df['Portfolio'], performance_df['SPY'])
is_significant = p_value < 0.05
significance_text = "Statistically Significant" if is_significant else "Not Statistically Significant"

# Monte Carlo (1-year forecast based on OOS volatility)
num_simulations, num_days = 10000, 252
port_mean = performance_df['Portfolio'].mean()
port_std = performance_df['Portfolio'].std()
sim_returns = np.random.normal(port_mean, port_std, (num_days, num_simulations))
sim_paths = np.cumprod(1 + sim_returns, axis=0)

# Monte Carlo Statistics
mc_mean = np.mean(sim_paths[-1])
mc_median = np.median(sim_paths[-1])
mc_5th = np.percentile(sim_paths[-1], 5)
mc_95th = np.percentile(sim_paths[-1], 95)

# --- SECTION 6: PDF EXPORT ---
pdf_filename = "Pictet_Style_Portfolio_Report.pdf"
with PdfPages(pdf_filename) as pdf:
    # Page 1: Executive Summary
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis('off')
    summary = (f"QUANTITATIVE EQUITY STRATEGY REPORT\n{'='*50}\n\n"
               f"Methodology: In-Sample (Y6-3) Optimisation | Out-of-Sample (Y3-0) Test\n"
               f"Annual Risk-Free Rate: {risk_free_rate:.2%}\n\n"
               f"OUT-OF-SAMPLE PERFORMANCE (Last 3 Years)\n{'-'*50}\n"
               f"Portfolio Total Return: {total_port_return:.2f}%\n"
               f"SPY Total Return: {total_spy_return:.2f}%\n"
               f"Alpha Generated: {alpha_gen:.2f}%\n\n"
               f"RISK-ADJUSTED PERFORMANCE\n{'-'*50}\n"
               f"Portfolio Sharpe Ratio: {sharpe_port:.3f}\n"
               f"SPY Sharpe Ratio: {sharpe_spy:.3f}\n"
               f"Sharpe Difference: {sharpe_port - sharpe_spy:.3f}\n\n"
               f"Portfolio Sortino Ratio: {sortino_port:.3f}\n"
               f"SPY Sortino Ratio: {sortino_spy:.3f}\n"
               f"Sortino Difference: {sortino_port - sortino_spy:.3f}\n\n"
               f"STATISTICAL SIGNIFICANCE TEST\n{'-'*50}\n"
               f"T-Statistic: {t_stat:.3f}\n"
               f"P-Value: {p_value:.4f}\n"
               f"Result: {significance_text} (α=0.05)\n\n"
               f"MONTE CARLO SIMULATION (1-Year Forward, $1 Initial)\n{'-'*50}\n"
               f"Expected Value: ${mc_mean:.2f}\n"
               f"Median Value: ${mc_median:.2f}\n"
               f"5th Percentile (VaR): ${mc_5th:.2f}\n"
               f"95th Percentile: ${mc_95th:.2f}")
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10, family='monospace', 
            verticalalignment='top', linespacing=1.5)
    pdf.savefig(); plt.close()

    # Page 2: Composition (Pie Chart)
    fig, ax = plt.subplots(figsize=(10, 7))
    weight_df = pd.Series(optimal_weights, index=valid_tickers)
    filtered = weight_df[weight_df > 0.02] # Show only weights > 2%
    ax.pie(filtered, labels=filtered.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    ax.set_title("Optimal Portfolio Composition (Trained on Year 6-3 Data)", fontsize=14, fontweight='bold')
    pdf.savefig(); plt.close()

    # Page 3: Backtest (Last 3 Years)
    plt.figure(figsize=(10, 6))
    plt.plot(performance_df['Port_Cum'], label='Optimised Strategy (OOS)', color='black', linewidth=2)
    plt.plot(performance_df['SPY_Cum'], label='S&P 500', color='red', linestyle='--', linewidth=2)
    plt.fill_between(performance_df.index, performance_df['Port_Cum'], performance_df['SPY_Cum'], 
                     alpha=0.1, color='green')
    plt.title("3-Year Out-of-Sample Growth of $1", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Value ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    pdf.savefig(); plt.close()

    # Page 4: Sharpe & Sortino Ratio Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sharpe Ratio Comparison
    strategies = ['Portfolio', 'SPY']
    sharpe_values = [sharpe_port, sharpe_spy]
    colors_sharpe = ['darkgreen' if sharpe_port > sharpe_spy else 'darkred', 'gray']
    bars1 = ax1.bar(strategies, sharpe_values, color=colors_sharpe, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    for i, (bar, val) in enumerate(zip(bars1, sharpe_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Sortino Ratio Comparison
    sortino_values = [sortino_port, sortino_spy]
    colors_sortino = ['darkgreen' if sortino_port > sortino_spy else 'darkred', 'gray']
    bars2 = ax2.bar(strategies, sortino_values, color=colors_sortino, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Sortino Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Sortino Ratio Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    for i, (bar, val) in enumerate(zip(bars2, sortino_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.suptitle(f'Risk-Adjusted Performance Metrics\nSignificance Test: {significance_text} (p={p_value:.4f})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # Page 5: Monte Carlo Simulation Paths
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Simulation Paths
    for i in range(min(100, num_simulations)):
        ax1.plot(sim_paths[:, i], color='blue', alpha=0.05, linewidth=0.5)
    
    # Plot percentiles
    ax1.plot(np.percentile(sim_paths, 50, axis=1), color='black', linewidth=2, label='Median (50th)')
    ax1.plot(np.percentile(sim_paths, 5, axis=1), color='red', linewidth=2, 
             linestyle='--', label='5th Percentile (VaR)')
    ax1.plot(np.percentile(sim_paths, 95, axis=1), color='green', linewidth=2, 
             linestyle='--', label='95th Percentile')
    
    ax1.fill_between(range(num_days), 
                     np.percentile(sim_paths, 5, axis=1),
                     np.percentile(sim_paths, 95, axis=1),
                     alpha=0.2, color='gray', label='90% Confidence Interval')
    
    ax1.set_title('Monte Carlo Simulation: 1-Year Portfolio Value Projection (10,000 Paths)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Trading Days', fontsize=11)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    # Bottom: Distribution of Final Values
    final_values = sim_paths[-1, :]
    ax2.hist(final_values, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(mc_mean, color='black', linestyle='-', linewidth=2, label=f'Mean: ${mc_mean:.2f}')
    ax2.axvline(mc_median, color='orange', linestyle='--', linewidth=2, label=f'Median: ${mc_median:.2f}')
    ax2.axvline(mc_5th, color='red', linestyle='--', linewidth=2, label=f'5th %ile: ${mc_5th:.2f}')
    ax2.axvline(mc_95th, color='green', linestyle='--', linewidth=2, label=f'95th %ile: ${mc_95th:.2f}')
    
    ax2.set_title('Distribution of 1-Year Final Portfolio Values', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Final Portfolio Value ($)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    pdf.savefig(); plt.close()

    # Page 6: Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(is_returns.corr(), annot=False, cmap='coolwarm', center=0)
    plt.title("Asset Correlation Matrix (Training Period)", fontsize=14, fontweight='bold')
    pdf.savefig(); plt.close()

print(f"✓ Report Generated Successfully: {pdf_filename}")
print(f"\nKey Findings:")
print(f"  • Portfolio Sharpe Ratio: {sharpe_port:.3f} vs SPY: {sharpe_spy:.3f}")
print(f"  • Portfolio Sortino Ratio: {sortino_port:.3f} vs SPY: {sortino_spy:.3f}")
print(f"  • Statistical Significance: {significance_text} (p={p_value:.4f})")
print(f"  • Monte Carlo Expected 1-Year Value: ${mc_mean:.2f}")
print(f"  • Alpha Generated: {alpha_gen:.2f}%")
