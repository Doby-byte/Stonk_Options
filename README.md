# Doby Trades: S&P 500 Options Analysis Tool

![License](https://img.shields.io/badge/license-MIT-blue.svg)

A powerful web-based tool for analyzing S&P 500 stocks and finding optimal options trading opportunities, with a focus on Cash-Secured Puts (CSP) and Covered Calls strategies.

## 🔍 Overview

This application helps traders identify potential options plays by:

1. Fetching real-time stock data and options chains from Yahoo Finance
2. Calculating key technical indicators like RSI, moving averages
3. Estimating probability of options expiring Out-of-The-Money (OTM)
4. Filtering options based on user-defined criteria:
   - Minimum OTM probability (safety)
   - Maximum budget (capital constraints)
   - Days to expiration (time horizon)
5. Categorizing opportunities as "Best," "Better," or "Good" based on risk/reward

## ✨ Features

- **Real-time Data**: Fetches current stock prices, RSI, and options chains
- **Probability Analysis**: Calculates OTM probabilities based on implied volatility
- **Risk Management**: Filters by probability and budget
- **Fear/Greed Analysis**: Identifies market sentiment for each stock
- **Interactive UI**: User-friendly web interface with sorting and filtering
- **Data Caching**: Reduces API calls and improves performance
- **Background Processing**: Fetches data without blocking the UI
- **Independent AI Recommendations**: Get AI-powered recommendations for Cash-Secured Put strategies with objective analysis independent of internal scoring systems

## 📊 Screenshots

*(Screenshots would be placed here)*

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Option 1: Setup using the run script
1. Clone the repository:
   ```
   git clone https://github.com/Doby-byte/Stonk_Options.git
   cd Stonk_Options
   ```
2. Run the setup/launch script:
   ```
   cd sp500_analysis
   chmod +x run.sh
   ./run.sh
   ```

### Option 2: Manual setup
1. Clone the repository:
   ```
   git clone https://github.com/Doby-byte/Stonk_Options.git
   cd Stonk_Options
   ```
2. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r sp500_analysis/requirements.txt
   ```
4. Run the application:
   ```
   cd sp500_analysis
   python app.py
   ```

The application will automatically open in your default web browser at http://localhost:8086.

## 🔧 Usage

1. **Select stocks** from the dropdown menu
2. Set your **filtering criteria**:
   - Minimum OTM probability (default: 90%)
   - Maximum budget per contract (default: $10,000)
   - Maximum days to expiration (default: 30 days)
3. Click **"Refresh Data"** to fetch and analyze options
4. View results in the table, sorted by premium per day
5. Click column headers to sort by different criteria
6. Use the **"Independent AI"** button in the recommendations section to get AI-powered analysis of the best CSP opportunities based on your selected risk tolerance

## 💻 Technical Details

### Architecture
- **Backend**: Flask web server with Python data processing
- **Frontend**: HTML/JS web interface with Bootstrap
- **Data Source**: Yahoo Finance API via yfinance library
- **Caching**: Local JSON file for data persistence

### Key Components
- **app.py**: Main Flask application with routing and data processing
- **sp500_stocks.py**: Static list of S&P 500 stocks
- **templates/index.html**: Web UI template
- **requirements.txt**: Python dependencies
- **run.sh**: Helper script to set up and launch the application

## Technical Documentation

### Architecture Overview

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

### Core Components

#### Data Collection (`app.py`)

The application fetches data from Yahoo Finance using the `yfinance` library:

- **Stock list**: Retrieved from Wikipedia or falls back to a static list
- **Price data**: Historical data for calculating technical indicators
- **Options chains**: Call and put options for various expiration dates

#### Technical Analysis (`app.py`)

Several indicators are calculated to assess stock conditions:

- **RSI (Relative Strength Index)**: Momentum oscillator indicating overbought/oversold
- **Moving Averages**: MA50 calculation for trend identification
- **Fear/Greed**: Sentiment indicator derived from RSI

#### Options Analysis (`app.py`)

The heart of the application is the options analysis engine:

- **OTM Probability**: Calculated using implied volatility and normal distribution
- **Premium Per Day**: Normalizes premium by days to expiration for comparison
- **Classification**: "Best," "Better," "Good" categories based on risk/reward

#### API Endpoints (`app.py`)

The Flask server provides these key endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the main application UI |
| `/api/refresh-data` | POST | Triggers a background refresh of stock data |
| `/api/data` | POST | Returns filtered options data for selected stocks |
| `/api/status` | GET | Reports current refresh status and progress |
| `/api/stocks` | GET | Returns the list of available S&P 500 stocks |
| `/api/independent-ai-recommendation` | POST | Returns AI-generated CSP recommendations based on current market data and risk tolerance |

#### Web Interface (`templates/index.html`)

The frontend provides an intuitive interface for:

- Stock selection with multi-select dropdown
- Parameter controls for filtering options
- Interactive data table with sorting and filtering
- Progress indicators for data refresh operations

### Key Algorithms

#### OTM Probability Calculation

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

#### Option Categorization

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

### Data Flow

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

### Performance Considerations

#### API Rate Limiting

Yahoo Finance imposes strict rate limits that can lead to "429 Too Many Requests" errors. This application implements a robust solution for handling these limits:

##### How the Rate Limiting Solution Works

- **Custom Browser Headers**: The application rotates through different User-Agent strings to mimic regular browser traffic instead of appearing as an automated script.
- **Session Management**: Uses persistent sessions with browser-like headers for all requests.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and informational purposes only. It is not intended to provide financial advice. Always conduct your own research and consult with a licensed financial advisor before making investment decisions.

The probability calculations are estimates based on statistical models and may not accurately predict actual market outcomes. Trade at your own risk.

### Recent Updates

#### UI Improvements (April 2025)
- **Streamlined Interface**: Removed duplicate refresh button for a cleaner user experience
- **Reorganized Components**: Moved the Independent AI button to the recommendations section for better logical grouping
- **Enhanced AI Recommendations**: Simplified the AI recommendation system to focus on objective analysis independent of internal scoring metrics
- **New Metrics**: Added OTM probability, option premium amount, and daily premium yield to AI recommendations

#### Technical Enhancements
- Consolidated refresh functionality to reduce code duplication
- Improved error handling in the data fetching process
- Added consistency in recommendation approach by standardizing on the independent AI model
