# S&P 500 Options Analysis Tool - Technical Documentation

## 1. Architecture Overview

This application follows a client-server architecture with both components running locally:

- **Backend**: Python Flask server that processes data and serves API endpoints
- **Frontend**: HTML/JavaScript client that provides the user interface

```
sp500_analysis/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── run.sh                 # Launch script
├── sp500_stocks.py        # Static stock data
├── data_cache.json        # Generated local data cache
├── templates/             # HTML templates
│   └── index.html         # Main UI template
└── README.md              # Documentation
```

## 2. Core Components

### 2.1 Data Collection (`app.py`)

The application fetches data from Yahoo Finance using the `yfinance` library:

- **Stock list**: Retrieved from Wikipedia or falls back to a static list
- **Price data**: Historical data for calculating technical indicators
- **Options chains**: Call and put options for various expiration dates

### 2.2 Technical Analysis (`app.py`)

Several indicators are calculated to assess stock conditions:

- **RSI (Relative Strength Index)**: Momentum oscillator indicating overbought/oversold
- **Moving Averages**: MA50 calculation for trend identification
- **Fear/Greed**: Sentiment indicator derived from RSI

### 2.3 Options Analysis (`app.py`)

The heart of the application is the options analysis engine:

- **OTM Probability**: Calculated using implied volatility and normal distribution
- **Premium Per Day**: Normalizes premium by days to expiration for comparison
- **Classification**: "Best," "Better," "Good" categories based on risk/reward

### 2.4 API Endpoints (`app.py`)

The Flask server provides these key endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the main application UI |
| `/api/refresh-data` | POST | Triggers a background refresh of stock data |
| `/api/data` | POST | Returns filtered options data for selected stocks |
| `/api/status` | GET | Reports current refresh status and progress |
| `/api/stocks` | GET | Returns the list of available S&P 500 stocks |

### 2.5 Web Interface (`templates/index.html`)

The frontend provides an intuitive interface for:

- Stock selection with multi-select dropdown
- Parameter controls for filtering options
- Interactive data table with sorting and filtering
- Progress indicators for data refresh operations

## 3. Key Algorithms

### 3.1 OTM Probability Calculation

```python
# Calculate probability (uses normal distribution approximation)
time_to_expiry = days_to_expiry / 365.0
log_moneyness = math.log(strike / current_price)

# Standard deviation for the move
std_dev = iv * math.sqrt(time_to_expiry)

# Z-score for the probability
z_score = log_moneyness / std_dev if std_dev > 0 else 0

# Probability of remaining OTM (above strike for calls)
otm_prob = norm.cdf(z_score)
```

### 3.2 Option Categorization

```python
def categorize_option(premium_per_day, otm_probability):
    if premium_per_day >= 0.40 and otm_probability >= 90:
        return "Best"
    elif premium_per_day >= 0.25 and otm_probability >= 90:
        return "Better"
    elif otm_probability >= 90:
        return "Good"
    return "Not Recommended"
```

## 4. Data Flow

1. **Initialization**:
   - Load cached data if available and fresh
   - If no cache, wait for user to trigger refresh

2. **Data Refresh**:
   - User clicks "Refresh Data" button
   - Background thread fetches data for all S&P 500 stocks
   - Progress is reported via `/api/status` endpoint
   - Results are cached to `data_cache.json`

3. **Options Analysis**:
   - User selects stocks and sets filtering criteria
   - Frontend sends POST request to `/api/data`
   - Backend filters options based on criteria
   - Results are returned to frontend for display

## 5. Performance Considerations

### 5.1 API Rate Limiting

Yahoo Finance imposes strict rate limits that can lead to "429 Too Many Requests" errors. This application implements a robust solution for handling these limits:

#### How the Rate Limiting Solution Works

- **Custom Browser Headers**: The application rotates through different User-Agent strings to mimic regular browser traffic instead of appearing as an automated script.
- **Session Management**: Uses persistent sessions with browser-like headers for all requests.
- **Implementation Location**: The `get_ticker_with_headers()` function in `app.py` manages this solution.

#### If You Encounter Rate Limiting Again

1. **Check Headers**: Ensure all API requests are using the custom headers through the `get_ticker_with_headers()` function.
2. **Expand User-Agents**: Add more variety to the `USER_AGENTS` list in `app.py`.
3. **Add Delays**: Increase delays between API requests, especially in batch processing.
4. **Enhance Caching**: Implement more aggressive caching to reduce the number of API calls.
5. **Try Different Networks**: If possible, use a different network or VPN if your current IP is rate-limited.

#### Code Example

```python
def get_ticker_with_headers(ticker_symbol):
    """Create a yfinance Ticker object with custom headers to avoid rate limiting."""
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }
    
    session = requests.Session()
    session.headers.update(headers)
    
    return yf.Ticker(ticker_symbol, session=session)
```

This solution has been proven effective in bypassing Yahoo Finance's rate limiting mechanisms and should be maintained in any future development.

### 5.2 Memory Management

- Process one stock at a time
- Don't hold full dataset in memory unnecessarily
- Use generators where applicable to conserve memory

## 6. Future Enhancements

1. **Additional Technical Indicators**: Add MACD, Bollinger Bands, etc.
2. **Portfolio Integration**: Allow tracking of positions and P&L
3. **Strategy Backtesting**: Test options strategies against historical data
4. **Option Chain Visualization**: Add charts and visualizations
5. **Multi-threading**: Potentially improve performance with parallel processing

## 7. Development Workflow

To contribute to this project:

1. Fork the repository
2. Make your changes
3. Test locally with `python app.py`
4. Submit a pull request with a detailed description of changes

## 8. Troubleshooting

Common issues and solutions:

- **No options data**: Some stocks may not have options available
- **Slow loading**: Expected due to API rate limiting
- **Port conflicts**: If port 8086 is in use, the app will try 8087 