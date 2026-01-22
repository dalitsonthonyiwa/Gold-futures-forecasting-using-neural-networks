# Predicting Gold Returns Using Neural Networks (Macro + Gold Futures)

This project builds a simple machine learning pipeline to **predict monthly returns of Gold** using a small set of **macroeconomic indicators from FRED** and **Gold futures data from Yahoo Finance**.  
Models are trained using TensorFlow/Keras and evaluated with MAE/MSE. A basic trading rule is then tested using the predicted return sign.

---

## Data Sources

### FRED (Macroeconomic Indicators)
Pulled using `pandas_datareader`:
- **PAYEMS** — Nonfarm Payrolls
- **UNRATE** — Unemployment Rate
- **PPIACO** — Producer Price Index
- **CES0500000003** — Average Hourly Earnings
- **CPILFESL** — Core CPI (ex food & energy)
- **FEDFUNDS** — Fed Funds Rate

### Yahoo Finance (Gold)
Pulled using `yfinance`:
- **GC=F** — Gold Futures (monthly interval)

---

## Target Definition

The target variable is **monthly gold returns**:

\[
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
\]

Implemented as:

```python
df["Adj Close"] = df["Gold futures"].pct_change()
df.dropna(inplace=True)
