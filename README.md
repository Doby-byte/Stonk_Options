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

## 📊 Screenshots

*(Screenshots would be placed here)*

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Option 1: Setup using the run script
1. Clone the repository:
   ```
   git clone https://github.com/your-username/doby-trades.git
   cd doby-trades
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
   git clone https://github.com/your-username/doby-trades.git
   cd doby-trades
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