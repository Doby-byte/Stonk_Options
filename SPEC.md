# S&P 500 Moving Average & Options Analysis Tool: Technical Specification

## System Architecture

### Overview
This application follows a client-server architecture where both components run locally:
- **Backend**: Python Flask server that processes data and serves API endpoints
- **Frontend**: HTML/JavaScript client that provides the user interface

### Components

```
sp500_analysis/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── data_cache.json        # Local data cache file
├── templates/             # HTML templates
│   └── index.html         # Main UI template
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Custom CSS styles
│   └── js/
│       └── main.js        # Frontend JavaScript
└── README.md              # Documentation
```

## Technology Stack

### Backend
- **Python 3.7+**: Core programming language
- **Flask**: Web framework for serving the application
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Yahoo Finance API wrapper
- **threading**: For background processing of data refresh

### Frontend
- **HTML5/CSS3**: Structure and styling
- **JavaScript**: Client-side interactivity
- **jQuery**: DOM manipulation and AJAX
- **Tabulator**: Interactive table component
- **Chart.js**: Data visualization

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the main application UI |
| `/api/refresh-data` | POST | Triggers a background refresh of stock data |
| `/api/data` | GET | Returns the current dataset |

## Data Model

### Stock Data Object
```json
{
  "ticker": "AAPL",
  "company": "Apple Inc.",
  "price": 150.25,
  "marketCap": 2500000000000,
  "ma5": 151.20,
  "ma5Ratio": -0.0063,
  "ma20": 148.30,
  "ma20Ratio": 0.0131,
  "ma50": 145.50,
  "ma50Ratio": 0.0326,
  "ma100": 140.75,
  "ma100Ratio": 0.0674,
  "ma200": 135.80,
  "ma200Ratio": 0.1063,
  "otmCall": 2.35,
  "otmPut": 1.85
}
```

## Core Functionality

### 1. Data Collection
- Retrieves S&P 500 ticker list from Wikipedia
- Fetches historical price data for each stock from Yahoo Finance
- Implements batch processing with delays to respect API rate limits

```python
def get_sp500_tickers():
    """Get the list of S&P 500 tickers"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = pd.read_html(url)
    df = data[0]
    return df[['Symbol', 'Security']].set_index('Symbol').to_dict()['Security']
```

### 2. Moving Average Calculation
- Calculates 5, 20, 50, 100, and 200-day moving averages for each stock
- Computes the percentage deviation from each MA to current price

```python
# Calculate moving averages
for period in ma_periods:
    hist[f'MA{period}'] = hist['Close'].rolling(window=period).mean()
    
# Get the last values
last_row = hist.iloc[-1]
current_price = last_row['Close']

# Calculate ratios
for period in ma_periods:
    ma_value = last_row[f'MA{period}']
    ratio = (current_price / ma_value - 1)
    result[f'ma{period}'] = ma_value
    result[f'ma{period}Ratio'] = ratio
```

### 3. Options Premium Calculation
- Fetches available options expirations for each stock
- Filters to next 4 weeks of expirations
- Finds options strikes with approximately 90% OTM probability
- Retrieves premium data for these options

```python
def calculate_options_data(stock):
    """Calculate options data for 90% OTM probability"""
    # Get expirations for next 4 weeks
    expirations = stock.options
    
    # Filter to next 4 weeks
    now = datetime.now()
    filtered_exps = []
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        if exp_date > now and (exp_date - now).days <= 28:  # 4 weeks
            filtered_exps.append(exp)
    
    # Get options chain
    if filtered_exps:
        expiry = filtered_exps[0]
        options = stock.option_chain(expiry)
        
        # Find 90% OTM options...
        # [Implementation details]
```

### 4. Data Caching
- Saves processed data to local JSON file
- Loads cached data on startup to avoid unnecessary API calls
- Includes timestamp for tracking data freshness

```python
def save_data_to_cache(data):
    with open('data_cache.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'data': data
        }, f)
```

### 5. Background Processing
- Uses threading to handle data refresh in background
- Prevents UI blocking during long data fetching operations
- Implements locking to prevent concurrent refresh operations

```python
@app.route('/api/refresh-data', methods=['POST'])
def refresh_data():
    # Start data refresh in background
    thread = threading.Thread(target=calculate_ma_ratios)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Data refresh started'
    })
```

## UI Components

### 1. Control Panel
- Refresh button with status indicator
- Moving average period selector
- Sort criteria selector

### 2. Data Table
- Interactive table with paging capabilities
- Sortable columns
- Formatted values (currency, percentages)
- Color-coded cells based on values

### 3. Data Visualization
- Distribution histogram of MA deviations
- Top "Greed" and "Fear" stocks bar charts

## API Rate Limiting Strategy

To stay within Yahoo Finance API limits:

1. **Batch Processing**: Process stocks in small batches (20 at a time)
2. **Inter-batch Delays**: Add 5-second delays between batches
3. **Manual Refresh**: Only fetch data when explicitly requested by user
4. **Local Caching**: Save results to avoid unnecessary fetches
5. **Progressive Loading**: Process and display stocks as they're fetched

## Optimization Techniques

1. **Parallel Processing Avoidance**: Deliberately process sequentially to avoid triggering rate limits
2. **Minimal Data Fetching**: Only fetch required date range for MA calculations
3. **Memory Efficiency**: Process one stock at a time, avoid holding full dataset in memory
4. **Progressive UI Updates**: Update UI as data becomes available rather than waiting for all stocks

## Security Considerations

1. **Local Processing**: All data processing happens locally, no external servers
2. **No Authentication**: Since this is a local application, no user authentication is required
3. **No Sensitive Data**: No personal or account information is stored or transmitted

## Performance Expectations

- **Initial Load**: < 2 seconds
- **Data Refresh**: 5-10 minutes (intentionally slow to respect API limits)
- **UI Responsiveness**: Immediate for all interactions except data refresh
- **Memory Usage**: < 500MB during normal operation

## Future Enhancements

1. **Additional Technical Indicators**: RSI, MACD, Bollinger Bands
2. **Custom Stock Lists**: Allow user to define and save custom stock lists
3. **Historical Backtesting**: Test strategies against historical data
4. **Enhanced Options Analysis**: Full options chain visualization
5. **Sector/Industry Analysis**: Group stocks by sector/industry for analysis 