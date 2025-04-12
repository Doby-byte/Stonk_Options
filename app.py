"""
S&P 500 Options Analysis Tool
=============================

This Flask web application analyzes S&P 500 stocks and identifies potential options trading opportunities,
particularly for Cash-Secured Puts (CSP) and Covered Calls.

Key Features:
- Fetches real-time stock data and options chains via yfinance
- Calculates moving averages, RSI, and other technical indicators
- Estimates probability of options expiring Out-of-The-Money (OTM)
- Categorizes options based on risk/reward metrics
- Provides a web interface for interactive analysis
- Caches data to minimize API calls and improve performance

Architecture:
- Backend: Flask server with Python data processing
- Frontend: HTML/JS web interface using Bootstrap, Tabulator, and jQuery
- Data Source: Yahoo Finance API via yfinance library
- Caching: Local JSON file storage

Author: Josh Boyd
Version: 1.0
"""

from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import threading
import json
import os
import webbrowser
from threading import Thread
import traceback
import sys
import requests
from scipy.stats import norm
from flask_cors import CORS
import logging
from sp500_stocks import get_stock_list, get_stock_by_ticker
import math
import dateutil.parser as parser
import random
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
# Configure API key for Google Generative AI
genai_api_key = os.getenv("GENAI_API_KEY")
if genai_api_key:
    genai.configure(api_key=genai_api_key)
else:
    logger.warning("GENAI_API_KEY not found in environment variables. AI recommendation feature will not work.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
data_lock = threading.Lock()  # Thread synchronization for shared data access
last_updated = None  # Timestamp of last data refresh
sp500_data = []  # Main data store for processed stock information
is_refreshing = False  # Flag to prevent concurrent refreshes
refresh_progress = 0  # Progress indicator for frontend (0-100)

# Cache for storing data
data_cache = {
    'stocks': [],
    'last_updated': None
}

#######################################################################
# RATE LIMITING SOLUTION
#######################################################################
# IMPORTANT: Yahoo Finance API has strict rate limits that can cause
# "429 Too Many Requests" errors and data retrieval failures.
#
# The solution implemented below uses custom browser User-Agent headers
# to bypass rate limiting. This works because Yahoo Finance treats
# requests from browsers differently than those from automated scripts.
# 
# If you encounter rate limiting issues again:
# 1. Ensure these custom headers are being used in all API requests
# 2. Try adding more/different User-Agent strings to the rotation
# 3. Add delays between requests (especially batch requests)
# 4. Implement more aggressive caching to reduce API calls
# 5. Consider using a different VPN server if available
#
# This solution was implemented on [DATE] and resolved persistent
# "429 Too Many Requests" errors.
#######################################################################

# List of user agents to rotate through
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
]

def get_ticker_with_headers(ticker_symbol):
    """
    Create a yfinance Ticker object with custom headers to avoid rate limiting.
    
    This function is a critical component of the rate limiting solution.
    Yahoo Finance uses request patterns and headers to detect and block
    automated scripts. By using rotating browser User-Agents and other
    browser-like headers, we can avoid being identified as a bot.
    
    When Yahoo Finance rate-limits requests, it returns HTTP 429 errors,
    making the application unable to fetch data. This approach has been
    proven to bypass those limitations.
    
    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., "AAPL")
        
    Returns:
        yfinance.Ticker: Ticker object with custom session
        
    Note:
        If rate limiting persists, try expanding the USER_AGENTS list or
        adding random delays between requests.
    """
    # Set up custom headers to avoid rate limiting
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }
    
    # Create a custom session with the headers
    session = requests.Session()
    session.headers.update(headers)
    
    # Return yfinance Ticker with the custom session
    return yf.Ticker(ticker_symbol, session=session)

def get_sp500_tickers():
    """
    Retrieve a dictionary of S&P 500 stock tickers mapped to company names.
    
    This function tries multiple approaches to get the current S&P 500 constituents:
    1. First attempts to scrape data from Wikipedia
    2. Falls back to a hardcoded list of major companies if the scrape fails
    
    Returns:
        dict: Dictionary mapping ticker symbols (keys) to company names (values)
    
    Example:
        {'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', ...}
    """
    try:
        print("Fetching S&P 500 tickers...")
        # First try to get the list from Wikipedia
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            
            # Set custom headers to avoid scraping blocks
            headers = {
                'User-Agent': random.choice(USER_AGENTS),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            
            # Use requests with custom headers to get the page
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                tables = pd.read_html(response.text)
                df = tables[0]
                tickers = df.set_index('Symbol')['Security'].to_dict()
                print(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia")
                return tickers
            else:
                print(f"Failed to fetch from Wikipedia: status code {response.status_code}")
                
        except Exception as e:
            print(f"Error fetching from Wikipedia: {e}")
            
        # Fallback to a hardcoded list of major S&P 500 companies
        print("Using fallback list of major S&P 500 companies")
        return {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "AMZN": "Amazon.com, Inc.",
            "NVDA": "NVIDIA Corporation",
            "GOOGL": "Alphabet Inc. (Class A)",
            "GOOG": "Alphabet Inc. (Class C)",
            "META": "Meta Platforms, Inc.",
            "TSLA": "Tesla, Inc.",
            "JPM": "JPMorgan Chase & Co.",
            "V": "Visa Inc.",
            "AVGO": "Broadcom Inc.",
            "LLY": "Eli Lilly and Company",
            "MA": "Mastercard Incorporated",
            "PG": "Procter & Gamble Company",
            "HD": "Home Depot, Inc.",
            "COST": "Costco Wholesale Corporation",
            "ABBV": "AbbVie Inc.",
            "MRK": "Merck & Co., Inc.",
            "AMD": "Advanced Micro Devices, Inc.",
            "BAC": "Bank of America Corporation",
            "KO": "Coca-Cola Company",
            "PEP": "PepsiCo, Inc.",
            "ADBE": "Adobe Inc.",
            "WMT": "Walmart Inc.",
            "CRM": "Salesforce, Inc.",
            "CSCO": "Cisco Systems, Inc.",
            "MCD": "McDonald's Corporation",
            "PFE": "Pfizer Inc.",
            "TMO": "Thermo Fisher Scientific Inc.",
            "NFLX": "Netflix, Inc.",
            "PLTR": "Palantir Technologies Inc."  # Added
        }
    except Exception as e:
        print(f"Error getting S&P 500 tickers: {e}")
        return {"TSLA": "Tesla, Inc."}  # Fallback to just Tesla if all else fails

def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a series of prices.
    
    RSI is a momentum oscillator that measures the speed and change of price movements.
    It ranges from 0 to 100, with values above 70 generally considered overbought
    and values below 30 considered oversold.
    
    Args:
        prices (pandas.Series): Time series of price data
        period (int): The lookback period for the calculation, typically 14 days
        
    Returns:
        pandas.Series: RSI values for the given price series
        
    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over the specified period
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_fear_greed_indicator(rsi):
    """
    Convert RSI value to a simplified "Fear/Greed" indicator.
    
    This function translates the RSI technical indicator into a more
    intuitive market sentiment indicator that's easier for users to interpret.
    
    Args:
        rsi (float): RSI value (0-100)
        
    Returns:
        str: "Fear" (RSI < 30), "Greed" (RSI > 70), or "Neutral" (RSI between 30-70)
        
    The Fear/Greed indicator helps traders understand if a stock might be:
    - Oversold (Fear): Potentially undervalued, good for buying
    - Overbought (Greed): Potentially overvalued, good for selling
    """
    if rsi < 30:
        return "Fear"
    elif rsi > 70:
        return "Greed"
    return "Neutral"

def calculate_premium_per_day(premium, days_to_expiry):
    """
    Calculate the premium earned per day for an option position.
    
    This metric helps compare options with different expiration dates by
    normalizing the premium value to a per-day basis.
    
    Args:
        premium (float): Option premium in dollars
        days_to_expiry (int): Number of days until the option expires
        
    Returns:
        float: Premium earned per day ($/day)
    """
    return premium / days_to_expiry if days_to_expiry > 0 else 0

def categorize_option(premium_per_day, otm_probability):
    """
    Categorize an option trade opportunity based on its risk/reward profile.
    
    This function applies predefined criteria to classify option opportunities
    into different quality tiers for easy evaluation.
    
    Args:
        premium_per_day (float): Option premium per day ($/day)
        otm_probability (float): Probability the option will expire OTM (0-100%)
        
    Returns:
        str: Category classification ("Best", "Better", "Good", or "Not Recommended")
        
    Categories:
        - "Best": Highest premium per day with high OTM probability
        - "Better": Good premium per day with high OTM probability
        - "Good": At least high OTM probability
        - "Not Recommended": Doesn't meet minimum criteria
    """
    if premium_per_day >= 0.40 and otm_probability >= 90:
        return "Best"
    elif premium_per_day >= 0.25 and otm_probability >= 90:
        return "Better"
    elif otm_probability >= 90:
        return "Good"
    return "Not Recommended"

def fetch_stock_data(ticker):
    """
    Fetch comprehensive stock data including price history, technical indicators, and options chain.
    
    This is a key function that retrieves all necessary data for a stock from Yahoo Finance,
    calculates relevant metrics, and structures the data for analysis.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL")
        
    Returns:
        dict: Dictionary containing the following data (or None if error occurs):
            - ticker: Stock ticker symbol
            - price: Current stock price (latest close)
            - rsi: Current Relative Strength Index value
            - ma50: 50-day moving average price
            - historical_volatility: Current 30-day historical volatility
            - options: Dictionary containing:
                - calls: List of call option contracts
                - puts: List of put option contracts
                
    Notes:
        - RSI is calculated using a 14-day period
        - Options data is limited to the next 6 expiration dates to manage data volume
        - Each option contract includes expiration date, strike price, and premium
    """
    try:
        logger.info(f"Fetching stock data for {ticker}")
        
        # Get Ticker with custom headers to avoid rate limiting
        stock = get_ticker_with_headers(ticker)
        
        # Get historical data for last year
        hist = stock.history(period="1y")
        
        if hist.empty:
            logger.warning(f"No historical data found for {ticker}")
            return None
            
        # Calculate current price (latest close)
        current_price = hist['Close'].iloc[-1]
        logger.info(f"{ticker} current price: ${current_price:.2f}")
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        logger.info(f"{ticker} RSI: {current_rsi:.2f}")
        
        # Calculate Historical Volatility (30-day)
        try:
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate rolling standard deviation (30-day window)
            hv_30 = returns.rolling(window=30).std().iloc[-1]
            
            # Annualize the volatility (multiply by sqrt of trading days in a year)
            historical_volatility = hv_30 * (252 ** 0.5)
            
            logger.info(f"{ticker} 30-day Historical Volatility: {historical_volatility:.4f}")
        except Exception as e:
            logger.warning(f"Error calculating historical volatility for {ticker}: {str(e)}")
            historical_volatility = 0.3  # Default value
        
        # Calculate MA50
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        logger.info(f"{ticker} MA50: ${ma50:.2f}")
        
        # Calculate max movement metrics
        try:
            max_daily_move, max_daily_date = calculate_max_daily_move(ticker)
            max_weekly_move, start_date, end_date = calculate_max_weekly_move(ticker)
            logger.info(f"{ticker} Max 1D Move: {max_daily_move:.2f}% on {max_daily_date}")
            logger.info(f"{ticker} Max 5D Move: {max_weekly_move:.2f}% from {start_date} to {end_date}")
        except Exception as e:
            logger.warning(f"Error calculating max movement metrics for {ticker}: {str(e)}")
            max_daily_move, max_daily_date = None, None
            max_weekly_move, start_date, end_date = None, None, None
        
        # Get options chain
        try:
            # Get all available expiration dates
            expirations = stock.options
            
            if not expirations or len(expirations) == 0:
                logger.warning(f"No option expiration dates found for {ticker}")
                # Return data without options
                return {
                    'ticker': ticker,
                    'price': current_price,
                    'rsi': current_rsi,
                    'ma50': ma50,
                    'historical_volatility': historical_volatility,
                    'max_daily_move': max_daily_move,
                    'max_daily_move_date': max_daily_date,
                    'max_weekly_move': max_weekly_move,
                    'max_weekly_move_start': start_date,
                    'max_weekly_move_end': end_date,
                    'options': {'calls': [], 'puts': []}
                }
                
            logger.info(f"Found {len(expirations)} expiration dates for {ticker}")
            
            # Get options for each expiration
            calls = []
            puts = []
            
            # Limit to next 6 expiration dates to avoid too much data
            for exp_date in expirations[:6]:
                try:
                    logger.debug(f"Fetching options for {ticker} expiring on {exp_date}")
                    opt = stock.option_chain(exp_date)
                    
                    # Add expiration date and ticker to each option
                    for call in opt.calls.to_dict('records'):
                        call['expiration'] = exp_date
                        call['ticker'] = ticker
                        calls.append(call)
                        
                    for put in opt.puts.to_dict('records'):
                        put['expiration'] = exp_date
                        put['ticker'] = ticker
                        puts.append(put)
                        
                except Exception as e:
                    logger.warning(f"Error fetching options for {ticker} expiring on {exp_date}: {str(e)}")
                    continue
                    
            logger.info(f"Fetched {len(calls)} calls and {len(puts)} puts for {ticker}")
            
            return {
                'ticker': ticker,
                'price': current_price,
                'rsi': current_rsi,
                'ma50': ma50,
                'historical_volatility': historical_volatility,
                'max_daily_move': max_daily_move,
                'max_daily_move_date': max_daily_date,
                'max_weekly_move': max_weekly_move,
                'max_weekly_move_start': start_date,
                'max_weekly_move_end': end_date,
                'options': {
                    'calls': calls,
                    'puts': puts
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching options for {ticker}: {str(e)}")
            # Return data without options
            return {
                'ticker': ticker,
                'price': current_price,
                'rsi': current_rsi,
                'ma50': ma50,
                'historical_volatility': historical_volatility,
                'max_daily_move': max_daily_move,
                'max_daily_move_date': max_daily_date,
                'max_weekly_move': max_weekly_move,
                'max_weekly_move_start': start_date,
                'max_weekly_move_end': end_date,
                'options': {'calls': [], 'puts': []}
            }
            
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def process_options_data(options_data, current_price, min_otm_prob, max_budget, min_days=7, max_days=60):
    """
    Process options data to find the best OTM (Out-of-The-Money) options based on user criteria.
    
    This function analyzes both call and put options to identify optimal trading opportunities
    that meet the specified probability, budget, and timeframe requirements.
    
    Args:
        options_data (dict): Dictionary containing 'calls' and 'puts' lists of option contracts
        current_price (float): Current stock price
        min_otm_prob (float): Minimum probability (0.0-1.0) of option expiring OTM
        max_budget (float): Maximum budget in dollars for the option contract
        min_days (int): Minimum days to expiration (default: 7)
        max_days (int): Maximum days to expiration (default: 60)
        
    Returns:
        dict: Dictionary containing:
            - best_otm_call: Best call option meeting criteria (or None)
            - best_otm_put: Best put option meeting criteria (or None)
            
    Each option result contains:
        - strike: Strike price
        - premium: Option premium
        - expiration: Expiration date
        - days_to_expiry: Days until expiration
        - probability: Estimated probability of expiring OTM
        
    Notes:
        - OTM probability is calculated using implied volatility and normal distribution
        - For calls, "best" means closest to the target min_otm_prob (default 90%)
        - For puts, the same criteria applies
    """
    logger.info(f"Processing options data with current price: ${current_price:.2f}")
    
    # Get current date
    now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Initialize results
    best_otm_call = None
    best_otm_put = None
    
    # Separate calls and puts
    calls = options_data.get('calls', [])
    puts = options_data.get('puts', [])
    
    logger.info(f"Processing {len(calls)} calls and {len(puts)} puts")
    
    # Process calls for OTM probability
    for call in calls:
        try:
            # Validate required fields
            if 'strike' not in call or 'lastPrice' not in call:
                continue
                
            # Get or convert necessary fields
            strike = float(call['strike'])
            premium = float(call['lastPrice'])
            
            # Skip deep ITM calls or invalid premiums
            if strike < current_price * 0.9 or premium <= 0:
                continue
                
            # Get expiration date - try different field names
            exp_date_str = None
            for field in ['expiration', 'expiry', 'lastTradeDate']:
                if field in call and call[field]:
                    exp_date_str = call[field]
                    break
                    
            if not exp_date_str:
                logger.warning(f"No expiration date found for call option with strike {strike}")
                continue
                
            # Parse expiration date
            try:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
            except ValueError:
                try:
                    # Try alternative format
                    exp_date = parser.parse(exp_date_str).replace(hour=0, minute=0, second=0, microsecond=0)
                except Exception as e:
                    logger.warning(f"Failed to parse expiration date {exp_date_str}: {str(e)}")
                    continue
            
            # Calculate days to expiry
            days_to_expiry = (exp_date - now).days
            
            # Skip if outside desired range
            if days_to_expiry < min_days or days_to_expiry > max_days:
                continue
                
            # Get implied volatility or use a default
            iv = float(call.get('impliedVolatility', 0.4))
            if iv <= 0:
                iv = 0.4  # Default implied volatility
                
            # Calculate probability (uses normal distribution approximation)
            time_to_expiry = days_to_expiry / 365.0
            log_moneyness = math.log(strike / current_price)
            
            # Standard deviation for the move
            std_dev = iv * math.sqrt(time_to_expiry)
            
            # Z-score for the probability
            z_score = log_moneyness / std_dev if std_dev > 0 else 0
            
            # Probability of remaining OTM (above strike for calls)
            otm_prob = norm.cdf(z_score)
            
            # Skip if probability doesn't meet minimum threshold
            if otm_prob < min_otm_prob:
                continue
                
            # If premium is within budget and better than current best
            if premium <= max_budget and (
                best_otm_call is None or 
                abs(otm_prob - min_otm_prob) < abs(best_otm_call['probability'] - min_otm_prob)
            ):
                best_otm_call = {
                    'strike': strike,
                    'premium': premium,
                    'expiration': exp_date_str,
                    'days_to_expiry': days_to_expiry,
                    'probability': otm_prob
                }
                
        except Exception as e:
            logger.warning(f"Error processing call option: {str(e)}")
            continue
            
    # Process puts for OTM probability
    for put in puts:
        try:
            # Validate required fields
            if 'strike' not in put or 'lastPrice' not in put:
                continue
                
            # Get or convert necessary fields
            strike = float(put['strike'])
            premium = float(put['lastPrice'])
            
            # Skip deep ITM puts or invalid premiums
            if strike > current_price * 1.1 or premium <= 0:
                continue
                
            # Get expiration date - try different field names
            exp_date_str = None
            for field in ['expiration', 'expiry', 'lastTradeDate']:
                if field in put and put[field]:
                    exp_date_str = put[field]
                    break
                    
            if not exp_date_str:
                logger.warning(f"No expiration date found for put option with strike {strike}")
                continue
                
            # Parse expiration date
            try:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
            except ValueError:
                try:
                    # Try alternative format
                    exp_date = parser.parse(exp_date_str).replace(hour=0, minute=0, second=0, microsecond=0)
                except Exception as e:
                    logger.warning(f"Failed to parse expiration date {exp_date_str}: {str(e)}")
                    continue
            
            # Calculate days to expiry
            days_to_expiry = (exp_date - now).days
            
            # Skip if outside desired range
            if days_to_expiry < min_days or days_to_expiry > max_days:
                continue
                
            # Get implied volatility or use a default
            iv = float(put.get('impliedVolatility', 0.4))
            if iv <= 0:
                iv = 0.4  # Default implied volatility
                
            # Calculate probability (uses normal distribution approximation)
            time_to_expiry = days_to_expiry / 365.0
            log_moneyness = math.log(strike / current_price)
            
            # Standard deviation for the move
            std_dev = iv * math.sqrt(time_to_expiry)
            
            # Z-score for the probability
            z_score = log_moneyness / std_dev if std_dev > 0 else 0
            
            # Probability of remaining OTM (below strike for puts)
            otm_prob = 1 - norm.cdf(z_score)
            
            # Skip if probability doesn't meet minimum threshold
            if otm_prob < min_otm_prob:
                continue
                
            # If premium is within budget and better than current best
            if premium <= max_budget and (
                best_otm_put is None or 
                abs(otm_prob - min_otm_prob) < abs(best_otm_put['probability'] - min_otm_prob)
            ):
                best_otm_put = {
                    'strike': strike,
                    'premium': premium,
                    'expiration': exp_date_str,
                    'days_to_expiry': days_to_expiry,
                    'probability': otm_prob
                }
                
        except Exception as e:
            logger.warning(f"Error processing put option: {str(e)}")
            continue
    
    logger.info(f"Best OTM call found: {best_otm_call}")
    logger.info(f"Best OTM put found: {best_otm_put}")
    
    return {
        'best_otm_call': best_otm_call,
        'best_otm_put': best_otm_put
    }

def calculate_ma_ratios():
    """Calculate moving averages and ratios for S&P 500 stocks"""
    global sp500_data, last_updated, is_refreshing, refresh_progress
    
    with data_lock:
        is_refreshing = True
        refresh_progress = 0
    
    try:
        print(f"Fetching S&P 500 data")
            
        # Get S&P 500 tickers
        tickers = get_sp500_tickers()
        total_tickers = len(tickers)
        
        print(f"Processing {total_tickers} stocks...")
        
        # Update progress
        with data_lock:
            refresh_progress = 5
            
        # Initialize results array
        results = []
        processed_count = 0
        batch_size = 10  # Process in batches to avoid rate limiting
        
        # Convert dictionary to list of tuples for batch processing
        ticker_items = list(tickers.items())
        
        # Process in batches
        for batch_index in range(0, len(ticker_items), batch_size):
            batch_num = batch_index // batch_size + 1
            total_batches = (len(ticker_items) + batch_size - 1) // batch_size
            
            print(f"\nProcessing batch {batch_num}/{total_batches}")
            
            # Process each ticker in the batch
            batch = ticker_items[batch_index:batch_index + batch_size]
            for ticker, company_name in batch:
                try:
                    print(f"\n{'='*10}")
                    print(f"Processing {ticker} ({company_name})")
                    print(f"{'='*10}")
                    
                    # Fetch data for this stock
                    stock_result = fetch_stock_data(ticker)
                    
                    if stock_result:
                        print(f"✅ Successfully got data for {ticker}")
                        results.append(stock_result)
                    else:
                        print(f"Not enough data for {ticker}, skipping")
                    
                except Exception as e:
                    print(f"Failed to get ticker '{ticker}' reason: {e}")
                    print(f"{ticker}: No timezone found, symbol may be delisted")
                
                # Update progress
                processed_count += 1
                progress_pct = int(5 + (processed_count / total_tickers * 95))
                
                with data_lock:
                    refresh_progress = progress_pct
                    sp500_data = results  # Update data as we go, so partial results are available
            
            # Add a small delay between batches to avoid rate limiting
            if batch_num < total_batches:
                print("Waiting 3 seconds before next batch...")
                time.sleep(3)
        
        # Update global data
        with data_lock:
            sp500_data = results
            last_updated = datetime.now()
            is_refreshing = False
            refresh_progress = 100
        
        # Save results to cache
        save_data_to_cache(results)
        
        print(f"Data saved to cache")
        return results
        
    except Exception as e:
        print(f"Error in data refresh: {e}")
        traceback.print_exc()
        
        # Return what we have so far
        with data_lock:
            last_updated = datetime.now()
            is_refreshing = False
            refresh_progress = 100
            
        return sp500_data

def calculate_options_data(stock):
    """Calculate options data for 90% OTM probability within 4 weeks (matching Robinhood's approach)"""
    try:
        # Get expirations
        expirations = stock.options
        
        if not expirations or len(expirations) == 0:
            return {'otmCall': 0, 'otmPut': 0, 'callExpiry': 'N/A', 'putExpiry': 'N/A'}
            
        # Filter to expirations within the next 4 weeks
        now = datetime.now()
        valid_expirations = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            days_to_expiry = (exp_date - now).days
            if days_to_expiry > 0 and days_to_expiry <= 28:  # 4 weeks
                valid_expirations.append((exp, days_to_expiry))
        
        if not valid_expirations:
            exp_date = datetime.strptime(expirations[0], "%Y-%m-%d")
            days_to_expiry = (exp_date - now).days
            valid_expirations = [(expirations[0], days_to_expiry)]
        
        # Sort by days to expiry, ascending
        valid_expirations.sort(key=lambda x: x[1])
        
        # Get current price
        current_price = stock.history(period="1d").iloc[-1]['Close']
        
        best_call = {'price': 0, 'strike': 0, 'probability': 0, 'expiry': '', 'days': 0}
        best_put = {'price': 0, 'strike': 0, 'probability': 0, 'expiry': '', 'days': 0}
        
        print(f"Finding options with ~90% OTM probability within next 4 weeks")
        print(f"Current price: ${current_price:.2f}")
        
        # Find best options for each expiration
        for expiry, days in valid_expirations:
            print(f"Checking expiration {expiry} ({days} days)")
            
            try:
                options = stock.option_chain(expiry)
                calls = options.calls
                puts = options.puts
                
                # Process calls - We want the strike that gives ~90% probability OTM
                # For calls, higher strikes = higher probability of remaining OTM
                calls = calls[calls['strike'] > current_price * 1.05]  # At least 5% OTM
                
                for _, call in calls.iterrows():
                    strike = call['strike']
                    # Calculate approximate probability of staying OTM based on strike distance
                    # This is a simpler formula that better approximates Robinhood's probabilities
                    strike_ratio = strike / current_price
                    
                    # Adjust based on time to expiry (longer time = more uncertainty)
                    time_factor = min(1.0, days / 30)  # 30 days as reference point
                    
                    # For calls: Higher strike = higher OTM probability
                    # The formula below approximates option probabilities
                    if strike_ratio <= 1.15:  # Close to current price
                        prob_otm = 0.5 + (strike_ratio - 1.0) * 3.0  # Steeper curve near the money
                    else:  # Far OTM
                        prob_otm = 0.8 + (strike_ratio - 1.15) * 1.0  # Less steep curve far OTM
                    
                    # Adjust for time - longer time reduces certainty
                    prob_otm = 0.5 + (prob_otm - 0.5) * (1.0 - time_factor * 0.3)
                    
                    prob_otm = min(0.99, max(0.5, prob_otm))  # Constrain to reasonable range
                    
                    # We're looking for options with approximately 90% probability
                    if 0.88 <= prob_otm <= 0.92:
                        if call['lastPrice'] > best_call['price'] or (
                            abs(prob_otm - 0.9) < abs(best_call['probability'] - 0.9) and call['lastPrice'] > 0
                        ):
                            best_call = {
                                'price': call['lastPrice'],
                                'strike': strike,
                                'probability': prob_otm,
                                'expiry': expiry,
                                'days': days
                            }
                            print(f"Found call: ${strike} strike, ${call['lastPrice']} premium, {prob_otm:.2%} OTM prob")
                
                # Process puts - We want the strike that gives ~90% probability OTM
                # For puts, lower strikes = higher probability of remaining OTM
                puts = puts[puts['strike'] < current_price * 0.95]  # At least 5% OTM
                
                for _, put in puts.iterrows():
                    strike = put['strike']
                    # Calculate approximate probability
                    strike_ratio = strike / current_price
                    
                    # Adjust based on time to expiry
                    time_factor = min(1.0, days / 30)  # 30 days as reference point
                    
                    # For puts: Lower strike = higher OTM probability
                    if strike_ratio >= 0.85:  # Close to current price
                        prob_otm = 0.5 + (1.0 - strike_ratio) * 3.0  # Steeper curve near the money
                    else:  # Far OTM
                        prob_otm = 0.8 + (0.85 - strike_ratio) * 1.0  # Less steep curve far OTM
                    
                    # Adjust for time - longer time reduces certainty
                    prob_otm = 0.5 + (prob_otm - 0.5) * (1.0 - time_factor * 0.3)
                    
                    prob_otm = min(0.99, max(0.5, prob_otm))  # Constrain to reasonable range
                    
                    # Target ~90% probability
                    if 0.88 <= prob_otm <= 0.92:
                        if put['lastPrice'] > best_put['price'] or (
                            abs(prob_otm - 0.9) < abs(best_put['probability'] - 0.9) and put['lastPrice'] > 0
                        ):
                            best_put = {
                                'price': put['lastPrice'],
                                'strike': strike,
                                'probability': prob_otm,
                                'expiry': expiry,
                                'days': days
                            }
                            print(f"Found put: ${strike} strike, ${put['lastPrice']} premium, {prob_otm:.2%} OTM prob")
            
            except Exception as e:
                print(f"Error processing expiration {expiry}: {e}")
                continue
        
        # If we didn't find good options within our target range, try with wider criteria
        if best_call['price'] == 0:
            print("No ideal call options found in target probability range, trying alternatives...")
            for expiry, days in valid_expirations:
                try:
                    options = stock.option_chain(expiry)
                    calls = options.calls[calls['strike'] > current_price]
                    
                    if len(calls) > 0:
                        # Try to find a strike that's roughly 30-40% above current price
                        # This typically approximates 90% probability for Tesla
                        target_strikes = calls[calls['strike'] > current_price * 1.3]
                        target_strikes = target_strikes[target_strikes['strike'] < current_price * 1.4]
                        
                        if len(target_strikes) > 0:
                            call = target_strikes.iloc[0]
                            best_call = {
                                'price': call['lastPrice'],
                                'strike': call['strike'],
                                'probability': 0.9,  # Approximate
                                'expiry': expiry,
                                'days': days
                            }
                            print(f"Using alternative call: ${call['strike']} strike, ${call['lastPrice']} premium")
                            break
                except Exception:
                    continue
        
        if best_put['price'] == 0:
            print("No ideal put options found in target probability range, trying alternatives...")
            for expiry, days in valid_expirations:
                try:
                    options = stock.option_chain(expiry)
                    puts = options.puts[options.puts['strike'] < current_price]
                    
                    if len(puts) > 0:
                        # Try to find a strike that's roughly 30-40% below current price
                        # This typically approximates 90% probability for Tesla
                        target_strikes = puts[puts['strike'] < current_price * 0.7]
                        target_strikes = target_strikes[target_strikes['strike'] > current_price * 0.6]
                        
                        if len(target_strikes) > 0:
                            put = target_strikes.iloc[0]
                            best_put = {
                                'price': put['lastPrice'],
                                'strike': put['strike'],
                                'probability': 0.9,  # Approximate
                                'expiry': expiry,
                                'days': days
                            }
                            print(f"Using alternative put: ${put['strike']} strike, ${put['lastPrice']} premium")
                            break
                except Exception:
                    continue
        
        # Last resort - if we still don't have options, find any OTM options
        if best_call['price'] == 0 and len(valid_expirations) > 0:
            try:
                expiry, days = valid_expirations[0]
                options = stock.option_chain(expiry)
                calls = options.calls[options.calls['strike'] > current_price]
                
                if len(calls) > 0:
                    call = calls.iloc[0]
                    best_call = {
                        'price': call['lastPrice'],
                        'strike': call['strike'],
                        'probability': 0.7,  # Conservative estimate
                        'expiry': expiry,
                        'days': days
                    }
            except Exception:
                pass
                
        if best_put['price'] == 0 and len(valid_expirations) > 0:
            try:
                expiry, days = valid_expirations[0]
                options = stock.option_chain(expiry)
                puts = options.puts[options.puts['strike'] < current_price]
                
                if len(puts) > 0:
                    put = puts.iloc[-1]
                    best_put = {
                        'price': put['lastPrice'],
                        'strike': put['strike'],
                        'probability': 0.7,  # Conservative estimate
                        'expiry': expiry,
                        'days': days
                    }
            except Exception:
                pass
        
        print(f"Best call: ${best_call['price']} premium, strike ${best_call['strike']}, expiring {best_call['expiry']}, {best_call['probability']:.1%} prob")
        print(f"Best put: ${best_put['price']} premium, strike ${best_put['strike']}, expiring {best_put['expiry']}, {best_put['probability']:.1%} prob")
        
        # Calculate option contract costs (strike x 100)
        call_contract_cost = best_call['strike'] * 100 if best_call['strike'] > 0 else 0
        put_contract_cost = best_put['strike'] * 100 if best_put['strike'] > 0 else 0
        
        # Calculate return percentages
        call_return_pct = (best_call['price'] / best_call['strike']) * 100 if best_call['strike'] > 0 else 0
        put_return_pct = (best_put['price'] / best_put['strike']) * 100 if best_put['strike'] > 0 else 0
            
        return {
            'otmCall': best_call['price'],
            'otmPut': best_put['price'],
            'callStrike': best_call['strike'],
            'putStrike': best_put['strike'],
            'callExpiry': best_call['expiry'],
            'putExpiry': best_put['expiry'],
            'callDays': best_call['days'],
            'putDays': best_put['days'],
            'callProb': round(best_call['probability'] * 100, 1),
            'putProb': round(best_put['probability'] * 100, 1),
            'callContractCost': call_contract_cost,
            'putContractCost': put_contract_cost,
            'callReturnPct': round(call_return_pct, 2),
            'putReturnPct': round(put_return_pct, 2)
        }
        
    except Exception as e:
        print(f"Error calculating options: {e}")
        traceback.print_exc()
        return {
            'otmCall': 0, 
            'otmPut': 0, 
            'callStrike': 0,
            'putStrike': 0,
            'callExpiry': 'N/A',
            'putExpiry': 'N/A',
            'callDays': 0,
            'putDays': 0,
            'callProb': 0,
            'putProb': 0,
            'callContractCost': 0,
            'putContractCost': 0,
            'callReturnPct': 0,
            'putReturnPct': 0
        }

def save_data_to_cache(data):
    """
    Save fetched and processed stock data to a local cache file.
    
    This function serializes the data to JSON format and saves it with a timestamp,
    which is used to determine freshness when loading from cache later.
    
    Args:
        data (list): List of processed stock data objects to cache
        
    Notes:
        - The cache file is named 'data_cache.json' in the current directory
        - The cache includes a timestamp in ISO format for freshness checking
        - Caching helps reduce API calls and improves application startup time
    """
    try:
        with open('data_cache.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f)
        print("Data saved to cache")
    except Exception as e:
        print(f"Error saving to cache: {e}")

def load_data_from_cache():
    """
    Load stock data from the local cache file if available and fresh.
    
    This function checks if cached data exists and is recent enough to use,
    which helps reduce API calls and improves application startup time.
    
    Returns:
        list or None: List of cached stock data objects if available and fresh,
                     otherwise None
                     
    Notes:
        - Cache is considered fresh if less than 30 minutes old
        - Returns None if:
          - Cache file doesn't exist
          - Cache is older than 30 minutes
          - Error occurs while reading the cache
    """
    try:
        if os.path.exists('data_cache.json'):
            with open('data_cache.json', 'r') as f:
                cache = json.load(f)
                # Check if cache is less than 30 minutes old
                cache_time = datetime.fromisoformat(cache['timestamp'])
                if (datetime.now() - cache_time).total_seconds() < 1800:  # 30 minutes
                    return cache['data']
                else:
                    print("Cache is older than 30 minutes, not using")
    except Exception as e:
        print(f"Error loading from cache: {e}")
    return None

@app.route('/')
def index():
    """
    Render the main application page.
    
    This route serves the main HTML template for the web interface.
    
    Returns:
        str: Rendered HTML template
    """
    return render_template('index.html')

@app.route('/api/refresh-data', methods=['POST'])
def refresh_data():
    """
    Trigger a background refresh of stock data.
    
    This route starts a new thread to fetch and process stock data without
    blocking the UI. It uses a lock to prevent concurrent refreshes.
    
    Returns:
        Response: JSON response indicating success or failure
        
    HTTP Status Codes:
        - 200: Refresh started successfully
        - 409: Another refresh is already in progress
    """
    global is_refreshing
    
    with data_lock:
        if is_refreshing:
            return jsonify({
                'success': False,
                'message': 'Data refresh already in progress'
            }), 409
    
    # Start data refresh in background thread
    thread = threading.Thread(target=calculate_ma_ratios)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Data refresh started'
    })

@app.route('/api/data', methods=['POST'])
@app.route('/api/refresh', methods=['POST'])  # Add an alias for the new frontend
def get_data_post():
    """
    Fetch and process data for selected stocks based on criteria.
    
    This is the main API endpoint for retrieving filtered option data.
    It accepts JSON parameters specifying stocks and filtering criteria,
    then returns matching option opportunities.
    
    Request Parameters (JSON):
        - stocks (list): List of stock objects with 'ticker' and 'name'
        - minOtmProb (float): Minimum probability (%) of option expiring OTM
        - maxBudget (float): Maximum budget in dollars
        - timeframe (int): Maximum days to expiration
        
    Returns:
        Response: JSON response containing:
            - success (bool): Operation success indicator
            - data (list): List of matching option opportunities
            - stocks (list): Same as data (for compatibility)
            - summary (dict): Counts and statistics
            
    Notes:
        - If no stocks are selected, NVDA is used as a default for testing
        - Results are sorted by premium_per_day in descending order
    """
    try:
        data = request.get_json()
        selected_stocks = data.get('stocks', [])
        min_otm_prob = float(data.get('minOtmProb', data.get('min_prob', 90)))/100  # Convert to decimal
        max_budget = float(data.get('maxBudget', data.get('max_budget', 10000)))
        timeframe_days = int(data.get('timeframe', 30))
        
        logger.info(f"Processing request for {len(selected_stocks)} stocks with {min_otm_prob*100}% min probability, ${max_budget} max budget, {timeframe_days} days timeframe")
        
        results = []
        summary = {
            'totalStocks': len(selected_stocks),
            'greedCount': 0,
            'fearCount': 0,
            'bestCount': 0,
            'betterCount': 0,
            'goodCount': 0,
        }
        
        # For testing/debugging - if no stocks selected, add NVDA
        if not selected_stocks:
            logger.info("No stocks selected, adding NVDA for testing")
            selected_stocks = [{"ticker": "NVDA", "name": "NVIDIA Corporation"}]
            
        for stock in selected_stocks:
            try:
                if not isinstance(stock, dict) or 'ticker' not in stock:
                    logger.warning(f"Invalid stock data: {stock}")
                    continue
                    
                ticker = stock['ticker']
                logger.info(f"Fetching data for {ticker}")
                
                stock_data = fetch_stock_data(ticker)
                if not stock_data:
                    logger.warning(f"Failed to get data for {ticker}")
                    continue
                    
                logger.info(f"Processing options for {ticker} (current price: ${stock_data['price']:.2f})")
                
                # Count fear/greed only once per stock
                if stock_data['rsi'] > 50:
                    summary['greedCount'] += 1
                else:
                    summary['fearCount'] += 1
                
                # Process options
                options_data = stock_data.get('options', {})
                if not options_data or not options_data.get('puts') or len(options_data.get('puts', [])) == 0:
                    logger.warning(f"No valid options data for {ticker}")
                    continue
                    
                # Log the number of options before filtering
                logger.info(f"Found {len(options_data.get('puts', []))} put options for {ticker} before filtering")
                
                # Process puts one by one to find good CSP opportunities
                for put in options_data.get('puts', []):
                    try:
                        if not isinstance(put, dict):
                            continue
                            
                        # Validate required fields
                        if 'strike' not in put or 'lastPrice' not in put:
                            continue
                            
                        # Get or convert necessary fields
                        strike = float(put['strike'])
                        premium = float(put['lastPrice'])
                        
                        # Skip invalid premiums
                        if premium <= 0:
                            continue
                            
                        # Get expiration date - try different field names
                        exp_date_str = None
                        for field in ['expiration', 'expiry', 'lastTradeDate']:
                            if field in put and put[field]:
                                exp_date_str = put[field]
                                break
                                
                        if not exp_date_str:
                            continue
                            
                        # Parse expiration date
                        try:
                            exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
                        except ValueError:
                            try:
                                # Try alternative format
                                exp_date = parser.parse(exp_date_str).replace(hour=0, minute=0, second=0, microsecond=0)
                            except Exception as e:
                                continue
                        
                        # Get current date
                        now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                        
                        # Calculate days to expiry
                        days_to_expiry = (exp_date - now).days
                        
                        # Skip if outside desired range
                        if days_to_expiry < 0 or days_to_expiry > timeframe_days:
                            continue
                            
                        # Get implied volatility or use a default
                        iv = float(put.get('impliedVolatility', 0.4))
                        if iv <= 0:
                            iv = 0.4  # Default implied volatility
                            
                        # Calculate probability (uses normal distribution approximation)
                        time_to_expiry = days_to_expiry / 365.0
                        log_moneyness = math.log(strike / stock_data['price'])
                        
                        # Standard deviation for the move
                        std_dev = iv * math.sqrt(time_to_expiry)
                        
                        # Z-score for the probability
                        z_score = log_moneyness / std_dev if std_dev > 0 else 0
                        
                        # Probability of remaining OTM (below strike for puts)
                        otm_prob = 1 - norm.cdf(z_score)
                        
                        # Skip if probability doesn't meet minimum threshold
                        if otm_prob < min_otm_prob:
                            continue
                            
                        # Skip if premium requires more cash than budget
                        cash_required = strike * 100  # One contract is for 100 shares
                        if cash_required > max_budget:
                            continue
                            
                        # Calculate financial metrics
                        premium_percent = premium / strike * 100
                        premium_per_day = premium / days_to_expiry if days_to_expiry > 0 else 0
                        
                        # Determine category based on hybrid CSP criteria
                        category = 'None'
                        percent_below = ((stock_data['price'] - strike) / stock_data['price']) * 100
                        
                        # Calculate support quality
                        if stock_data['ma50'] > 0:
                            strike_to_ma_ratio = strike / stock_data['ma50']
                            # Define support quality levels
                            if strike_to_ma_ratio <= 0.90:  # Strike is 10% or more below 50MA (strong support)
                                support_quality = "excellent"
                            elif strike_to_ma_ratio <= 0.95:
                                support_quality = "good"
                            elif strike_to_ma_ratio <= 1.0:
                                support_quality = "moderate"
                            else:
                                support_quality = "weak"
                        else:
                            support_quality = "unknown"
                        
                        # Use 85% probability threshold across all categories
                        if otm_prob >= 0.85:
                            # Best: Premium ≥ $0.20/day AND excellent support
                            if premium_per_day >= 0.20 and support_quality == "excellent":
                                category = 'Best'
                                summary['bestCount'] += 1
                            # Better: Premium ≥ $0.15/day AND good or excellent support
                            elif premium_per_day >= 0.15 and (support_quality == "good" or support_quality == "excellent"):
                                category = 'Better'
                                summary['betterCount'] += 1
                            # Good: Premium ≥ $0.10/day (regardless of support quality)
                            elif premium_per_day >= 0.10:
                                category = 'Good'
                                summary['goodCount'] += 1
                        
                        # Store support quality for display
                        support_quality_display = support_quality.capitalize()
                        
                        # Create the option result
                        option_result = {
                            'strike': strike,
                            'premium': premium,
                            'expiry': exp_date_str,
                            'days_to_expiry': days_to_expiry,
                            'otm_probability': otm_prob * 100,  # Convert to percentage
                            'return_percent': premium_percent,
                            'premium_per_day': premium_per_day,
                            'implied_volatility': iv,
                            'percent_below': percent_below,
                            'category': category,
                            'support_quality': support_quality_display
                        }
                        
                        # Add stock data to the option
                        option_result.update({
                            'ticker': ticker,
                            'company': stock.get('name', ''),
                            'current_price': stock_data['price'],
                            'rsi': stock_data['rsi'],
                            'fear_greed': stock_data['rsi'] > 50 and 'Greed' or 'Fear',
                            'ma50': stock_data['ma50'],
                            'ma50_ratio': stock_data['price'] / stock_data['ma50'] if stock_data['ma50'] > 0 else 1.0
                        })
                        
                        # Add max movement data if available
                        try:
                            # Calculate max daily move if not already present
                            if 'max_daily_move' not in stock_data:
                                max_daily_move, max_daily_date = calculate_max_daily_move(ticker)
                                stock_data['max_daily_move'] = max_daily_move
                                stock_data['max_daily_move_date'] = max_daily_date
                            
                            # Calculate max weekly move if not already present
                            if 'max_weekly_move' not in stock_data:
                                max_weekly_move, start_date, end_date = calculate_max_weekly_move(ticker)
                                stock_data['max_weekly_move'] = max_weekly_move
                                stock_data['max_weekly_move_start'] = start_date
                                stock_data['max_weekly_move_end'] = end_date
                            
                            # Add movement data to option result
                            option_result.update({
                                'max_daily_move': stock_data.get('max_daily_move'),
                                'max_daily_move_date': stock_data.get('max_daily_move_date'),
                                'max_weekly_move': stock_data.get('max_weekly_move'),
                                'max_weekly_move_start': stock_data.get('max_weekly_move_start'),
                                'max_weekly_move_end': stock_data.get('max_weekly_move_end')
                            })
                        except Exception as e:
                            logger.warning(f"Error adding max movement data: {str(e)}")
                        
                        # Calculate CSP Quality Score
                        try:
                            # Get historical volatility or use a default
                            hv = stock_data.get('historical_volatility', 0.3)
                            if hv <= 0:
                                hv = 0.3  # Default historical volatility
                                
                            # Calculate the CSP quality score
                            csp_score, score_components = calculate_csp_quality_score(
                                iv=iv,
                                hv=hv,
                                rsi=stock_data['rsi'],
                                premium_per_day=premium_per_day,
                                otm_probability=otm_prob,
                                strike_price=strike,
                                current_price=stock_data['price'],
                                ma50=stock_data['ma50']
                            )
                            
                            # Add the score and components to the option result
                            option_result['csp_quality_score'] = csp_score
                            option_result['score_components'] = score_components
                            
                            # Add detailed debug logging
                            logger.debug(f"CSP Quality Score components for {ticker} at ${strike}:")
                            logger.debug(f"  IV/HV Score: {score_components.get('iv_hv_score', 0):.1f}/10 (weight: 30%)")
                            logger.debug(f"  RSI Score: {score_components.get('rsi_score', 0):.1f}/10 (weight: 25%)")
                            logger.debug(f"  Premium Score: {score_components.get('premium_score', 0):.1f}/10 (weight: 25%)")
                            logger.debug(f"  OTM Prob Score: {score_components.get('otm_score', 0):.1f}/10 (weight: 10%)")
                            logger.debug(f"  Support Score: {score_components.get('support_score', 0):.1f}/10 (weight: 10%)")
                            
                            # Add a signal based on the score
                            if csp_score >= 8:
                                option_result['csp_signal'] = 'EXCELLENT'
                            elif csp_score >= 6:
                                option_result['csp_signal'] = 'GOOD'
                            elif csp_score >= 4:
                                option_result['csp_signal'] = 'FAIR'
                            else:
                                option_result['csp_signal'] = 'POOR'
                                
                            logger.debug(f"CSP Quality Score for {ticker} at ${strike}: {csp_score:.1f} ({option_result['csp_signal']})")
                            
                        except Exception as e:
                            logger.error(f"Error calculating CSP quality score: {str(e)}")
                            # Default values if calculation fails
                            option_result['csp_quality_score'] = 0
                            option_result['score_components'] = {}
                            option_result['csp_signal'] = 'N/A'
                        
                        results.append(option_result)
                        
                    except Exception as e:
                        logger.debug(f"Skipping option due to error: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue
                
        logger.info(f"Total results: {len(results)} options across all stocks")
        
        # Sort by premium_per_day in descending order
        results.sort(key=lambda x: x.get('premium_per_day', 0), reverse=True)
            
        return jsonify({
            'success': True,
            'data': results,
            'stocks': results,  # Add this for new frontend
            'summary': summary  # Add this for new frontend
        })
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_data', methods=['GET'])
def get_data():
    """API endpoint to get stock data."""
    try:
        # Get parameters
        ticker = request.args.get('ticker', 'TSLA')
        prob = float(request.args.get('probability', 90))/100  # Convert to decimal
        budget = float(request.args.get('budget', 1000))
        max_days = int(request.args.get('timeframe', 60))
        min_days = int(request.args.get('min_days', 7))
        
        # Log request
        logger.info(f"Data requested for {ticker} with probability {prob*100}%, budget ${budget}, timeframe {min_days}-{max_days} days")
        
        # Get stock
        stock = get_stock_by_ticker(ticker)
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found"}), 404
            
        # Fetch stock data
        stock_data = fetch_stock_data(ticker)
        if not stock_data:
            return jsonify({"error": f"Failed to retrieve data for {ticker}"}), 500
            
        current_price = stock_data.get('current_price', 0)
        if current_price <= 0:
            return jsonify({"error": f"Invalid current price for {ticker}"}), 500
            
        # Process options data
        options_data = stock_data.get('options', {})
        otm_options = process_options_data(
            options_data, 
            current_price, 
            min_otm_prob=prob, 
            max_budget=budget,
            min_days=min_days,
            max_days=max_days
        )
        
        # Build response
        response = {
            "ticker": ticker,
            "name": stock.get('name', ''),
            "current_price": current_price,
            "ma50": stock_data.get('ma50', 0),
            "rsi": stock_data.get('rsi', 0),
            "otm_call": otm_options.get('best_otm_call'),
            "otm_put": otm_options.get('best_otm_put')
        }
        
        logger.info(f"Returning data for {ticker}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in get_data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def get_status():
    """
    Get the current status of data refresh operations.
    
    This endpoint provides information about the refresh process,
    allowing the frontend to display progress indicators and timestamps.
    
    Returns:
        Response: JSON response containing:
            - isRefreshing (bool): Whether a refresh is in progress
            - progress (int): Refresh progress percentage (0-100)
            - lastUpdated (str): ISO timestamp of last update or null
            - dataAvailable (bool): Whether any data is available
            - stockCount (int): Number of stocks in the dataset
    """
    with data_lock:
        return jsonify({
            'isRefreshing': is_refreshing,
            'progress': refresh_progress,
            'lastUpdated': last_updated.isoformat() if last_updated else None,
            'dataAvailable': len(sp500_data) > 0,
            'stockCount': len(sp500_data)
        })

@app.route('/api/options_data')
def get_options_data():
    """
    Get options data for default stocks (currently only Tesla).
    
    This is a simplified endpoint that returns pre-calculated options data,
    primarily used for quick testing and demonstration.
    
    Returns:
        Response: JSON response containing:
            - options_data: Pre-calculated options data
            - companies: List of company information
    """
    try:
        options_data = calculate_options_data()
        companies = [
            {"symbol": "TSLA", "name": "Tesla, Inc."}
        ]
        return jsonify({
            "options_data": options_data,
            "companies": companies
        })
    except Exception as e:
        app.logger.error(f"Error fetching options data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stocks')
def get_stocks():
    """Return list of available stocks for selection."""
    stocks = get_stock_list()
    return jsonify({"stocks": stocks})

@app.route('/api/ai-recommendation', methods=['POST'])
def ai_recommendation():
    """
    Get AI-powered recommendation for CSP opportunities.
    
    This endpoint takes the current stocks data and user preferences,
    then asks the AI model to analyze and recommend the best CSP opportunity.
    
    Request JSON parameters:
    - risk_tolerance (str, optional): User's risk preference ('conservative', 'moderate', or 'aggressive')
    
    Returns:
        JSON with AI recommendation details
    """
    try:
        # Get request data
        data = request.json or {}
        risk_tolerance = data.get('risk_tolerance', 'moderate')
        
        # Validate risk tolerance
        valid_tolerances = ['conservative', 'moderate', 'aggressive']
        if risk_tolerance not in valid_tolerances:
            risk_tolerance = 'moderate'
        
        # Check if we have data
        if not sp500_data:
            return jsonify({
                "success": False,
                "error": "No stock data available. Please refresh data first."
            }), 400
        
        # Get AI recommendation
        result = get_ai_recommendation(sp500_data, risk_tolerance)
        
        if 'error' in result and result.get('recommendation') is None:
            return jsonify({
                "success": False,
                "error": result['error']
            }), 500
            
        return jsonify({
            "success": True,
            "recommendation": result['recommendation']
        })
        
    except Exception as e:
        logger.error(f"Error in AI recommendation endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

def open_browser(port=8086):
    """
    Open the default web browser to the application URL.
    
    This function is called when the application starts, providing
    a seamless experience by automatically loading the UI.
    
    Args:
        port (int): The port number where the Flask app is running
    """
    time.sleep(1.5)  # Short delay to ensure server has started
    webbrowser.open(f'http://localhost:{port}')

def calculate_max_daily_move(ticker, period='3mo'):
    """Calculate the maximum single-day movement for a stock within a period."""
    try:
        # Retrieve historical data for the stock
        stock_data = get_ticker_with_headers(ticker).history(period=period)
        
        if len(stock_data) == 0:
            logging.warning(f"No historical data found for {ticker}.")
            return None, None
        
        # Calculate daily high-to-low percentage movement
        stock_data['daily_move'] = ((stock_data['High'] - stock_data['Low']) / stock_data['Low']) * 100
        
        # Find the maximum daily movement and its date
        max_idx = stock_data['daily_move'].idxmax()
        max_move = stock_data.loc[max_idx, 'daily_move']
        max_date = max_idx.strftime('%Y-%m-%d')
        
        logging.info(f"Max daily move for {ticker}: {max_move:.2f}% on {max_date}")
        return max_move, max_date
    except Exception as e:
        logging.error(f"Error calculating max daily move for {ticker}: {str(e)}")
        return None, None

def calculate_max_weekly_move(ticker, period='3mo'):
    """Calculate the maximum 5-day (weekly) movement for a stock within a period."""
    try:
        # Retrieve historical data for the stock
        stock_data = get_ticker_with_headers(ticker).history(period=period)
        
        if len(stock_data) < 5:
            logging.warning(f"Insufficient historical data for {ticker} to calculate weekly movement.")
            return None, None, None
        
        # Calculate 5-day rolling high and low
        stock_data['5d_high'] = stock_data['High'].rolling(window=5).max()
        stock_data['5d_low'] = stock_data['Low'].rolling(window=5).min()
        
        # Calculate the 5-day price range as a percentage
        stock_data['weekly_move'] = ((stock_data['5d_high'] - stock_data['5d_low']) / stock_data['5d_low']) * 100
        
        # Skip the first 4 days since they don't have complete 5-day windows
        valid_data = stock_data.iloc[4:]
        
        if len(valid_data) == 0:
            return None, None, None
        
        # Find the maximum weekly movement and its end date
        max_idx = valid_data['weekly_move'].idxmax()
        max_move = valid_data.loc[max_idx, 'weekly_move']
        end_date = max_idx.strftime('%Y-%m-%d')
        
        # Calculate the start date (5 trading days before)
        start_idx = stock_data.index.get_loc(max_idx) - 4
        if start_idx >= 0:
            start_date = stock_data.index[start_idx].strftime('%Y-%m-%d')
        else:
            start_date = "N/A"
        
        logging.info(f"Max weekly move for {ticker}: {max_move:.2f}% from {start_date} to {end_date}")
        return max_move, start_date, end_date
    except Exception as e:
        logging.error(f"Error calculating max weekly move for {ticker}: {str(e)}")
        return None, None, None

def calculate_csp_quality_score(iv, hv, rsi, premium_per_day, otm_probability, strike_price, current_price, ma50):
    """
    Calculate a CSP Quality Score (0-10) based on multiple factors.
    
    Parameters:
    - iv: Implied volatility (decimal, not percentage)
    - hv: Historical volatility (decimal, not percentage)
    - rsi: Relative Strength Index (0-100)
    - premium_per_day: Premium earned per day
    - otm_probability: Probability of option expiring out-of-the-money (decimal)
    - strike_price: Option strike price
    - current_price: Current stock price
    - ma50: 50-day moving average price
    
    Returns:
    - score: Quality score from 0-10
    - components: Dict with individual component scores
    """
    try:
        components = {}
        
        # IV/HV Ratio Component (30% weight)
        # Lower values are better (options priced below historical volatility)
        if hv > 0:
            iv_hv_ratio = iv / hv
            if iv_hv_ratio <= 0.8:
                components['iv_hv_score'] = 10  # Significantly underpriced options
            elif iv_hv_ratio <= 0.9:
                components['iv_hv_score'] = 8
            elif iv_hv_ratio <= 1.0:
                components['iv_hv_score'] = 6
            elif iv_hv_ratio <= 1.1:
                components['iv_hv_score'] = 4
            elif iv_hv_ratio <= 1.2:
                components['iv_hv_score'] = 2
            else:
                components['iv_hv_score'] = 0  # Significantly overpriced options
        else:
            components['iv_hv_score'] = 5  # Neutral if HV data is not available
        
        # RSI Component (25% weight)
        # Lower RSI values are better for CSPs (oversold conditions)
        if rsi <= 30:
            components['rsi_score'] = 10  # Oversold conditions, good for CSP
        elif rsi <= 40:
            components['rsi_score'] = 8
        elif rsi <= 50:
            components['rsi_score'] = 6
        elif rsi <= 60:
            components['rsi_score'] = 4
        elif rsi <= 70:
            components['rsi_score'] = 2
        else:
            components['rsi_score'] = 0  # Overbought conditions, poor for CSP
        
        # Premium Quality Component (25% weight)
        # Higher premium per day relative to strike price is better
        annualized_return = (premium_per_day * 365) / strike_price
        if annualized_return >= 0.40:  # 40% annualized return
            components['premium_score'] = 10
        elif annualized_return >= 0.30:
            components['premium_score'] = 8
        elif annualized_return >= 0.20:
            components['premium_score'] = 6
        elif annualized_return >= 0.10:
            components['premium_score'] = 4
        elif annualized_return >= 0.05:
            components['premium_score'] = 2
        else:
            components['premium_score'] = 0
        
        # OTM Probability Component (10% weight)
        # Higher probability of expiring OTM is better
        if otm_probability >= 0.95:
            components['otm_score'] = 10  # Very safe
        elif otm_probability >= 0.90:
            components['otm_score'] = 8
        elif otm_probability >= 0.80:
            components['otm_score'] = 6
        elif otm_probability >= 0.70:
            components['otm_score'] = 4
        elif otm_probability >= 0.60:
            components['otm_score'] = 2
        else:
            components['otm_score'] = 0  # High risk of assignment
        
        # Technical Support Component (10% weight)
        # Strike price near or below moving average support levels is better
        if ma50 > 0:
            strike_to_ma_ratio = strike_price / ma50
            if strike_to_ma_ratio <= 0.90:  # Strike is 10% or more below 50MA (strong support)
                components['support_score'] = 10
            elif strike_to_ma_ratio <= 0.95:
                components['support_score'] = 8
            elif strike_to_ma_ratio <= 1.0:
                components['support_score'] = 6
            elif strike_to_ma_ratio <= 1.05:
                components['support_score'] = 4
            elif strike_to_ma_ratio <= 1.10:
                components['support_score'] = 2
            else:
                components['support_score'] = 0  # Strike far above support levels
        else:
            components['support_score'] = 5  # Neutral if MA data is not available
        
        # Calculate weighted score
        final_score = (
            components['iv_hv_score'] * 0.30 + 
            components['rsi_score'] * 0.25 + 
            components['premium_score'] * 0.25 + 
            components['otm_score'] * 0.10 + 
            components['support_score'] * 0.10
        )
        
        return round(final_score, 1), components
        
    except Exception as e:
        logging.error(f"Error calculating CSP quality score: {str(e)}")
        return 0, {}

def get_ai_recommendation(stocks_data, risk_tolerance='moderate'):
    """
    Get AI-powered recommendation based on current stock data and market conditions.
    
    This function sends the current stock data, technical indicators, and market sentiment
    to the Google Generative AI model (Gemini) and requests a stock recommendation.
    
    Args:
        stocks_data (list): List of processed stock data with options chains and technical indicators
        risk_tolerance (str): User's risk tolerance level ('conservative', 'moderate', or 'aggressive')
        
    Returns:
        dict: AI recommendation including:
            - recommended_stock (dict): The recommended stock data
            - reasoning (str): Explanation for the recommendation
            - confidence (float): Confidence score (0-1)
            - market_sentiment (str): Current market sentiment assessment
    """
    if not genai_api_key:
        logger.error("Cannot get AI recommendation: GENAI_API_KEY not configured")
        return {
            "error": "API key not configured",
            "recommendation": None
        }
    
    try:
        # Select top 5 stocks by CSP quality score for processing efficiency
        if stocks_data:
            # Filter to only stocks with options data
            valid_stocks = [s for s in stocks_data if 'options' in s and s['options']]
            # Sort by CSP quality score (if available)
            sorted_stocks = sorted(valid_stocks, 
                                key=lambda x: x.get('csp_quality_score', 0), 
                                reverse=True)
            # Get top 5 for processing
            top_stocks = sorted_stocks[:5] if len(sorted_stocks) > 5 else sorted_stocks
        else:
            return {
                "error": "No stock data available",
                "recommendation": None
            }
        
        # Get current market data and sentiment indicators
        now = datetime.now().strftime("%Y-%m-%d")
        rsi_values = [s.get('rsi', 50) for s in stocks_data if 'rsi' in s]
        avg_market_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 50
        
        # Format data for the AI prompt
        stock_summaries = []
        for stock in top_stocks:
            ticker = stock.get('ticker', 'Unknown')
            company_name = stock.get('company_name', 'Unknown Company')
            current_price = stock.get('current_price', 0)
            rsi = stock.get('rsi', 0)
            ma50 = stock.get('ma50', 0)
            ma200 = stock.get('ma200', 0)
            csp_score = stock.get('csp_quality_score', 0)
            category = stock.get('category', 'N/A')
            
            # Get best option for this stock
            best_option = None
            if 'options' in stock and stock['options']:
                sorted_options = sorted(stock['options'], 
                                     key=lambda x: x.get('premium_per_day', 0),
                                     reverse=True)
                if sorted_options:
                    best_option = sorted_options[0]
            
            option_details = ""
            if best_option:
                option_details = f"""
                Best CSP Option:
                - Strike: ${best_option.get('strike', 0)}
                - Premium: ${best_option.get('premium', 0)}
                - Premium/day: ${best_option.get('premium_per_day', 0):.2f}
                - OTM Probability: {best_option.get('otm_probability', 0):.1f}%
                - Expiration: {best_option.get('expiration_date', 'Unknown')}
                """
            
            summary = f"""
            {ticker} ({company_name}):
            - Current Price: ${current_price}
            - RSI: {rsi:.1f}
            - MA50: ${ma50:.2f}
            - MA200: ${ma200:.2f}
            - CSP Quality Score: {csp_score:.1f}/10
            - Category: {category}
            {option_details}
            """
            stock_summaries.append(summary)
        
        # Create the prompt for Gemini
        prompt = f"""
        You are a financial advisor specializing in options trading, particularly Cash-Secured Puts (CSP) strategies.
        
        Current date: {now}
        Average market RSI: {avg_market_rsi:.1f}
        User's risk tolerance: {risk_tolerance}
        
        Analyze these top stocks for CSP opportunities:
        
        {"".join(stock_summaries)}
        
        Based on this data, recommend ONE specific stock and options contract for a Cash-Secured Put (CSP) strategy.
        Consider the user's risk tolerance of '{risk_tolerance}' in your recommendation.
        
        Your recommendation should include:
        1. The single best stock ticker to sell a CSP on
        2. The specific strike price and expiration to recommend
        3. 3-5 bullet points explaining your reasoning
        4. A confidence score (1-5) for your recommendation
        5. Brief assessment of current market conditions
        
        Format your response as JSON with these fields:
        ticker, strike_price, expiration_date, reasoning_bullets (array), confidence_score, market_assessment
        """
        
        # Set up the model and parameters
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Get response from Gemini
        model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash",
            generation_config=generation_config
        )
        response = model.generate_content(prompt)
        
        # Process the response (assuming JSON format)
        try:
            import json
            result = json.loads(response.text)
            
            # Find the stock data for the recommended ticker
            recommended_stock = next((s for s in stocks_data if s['ticker'] == result['ticker']), None)
            
            # Construct the final recommendation
            recommendation = {
                "recommended_stock": recommended_stock,
                "ticker": result['ticker'],
                "strike_price": result['strike_price'],
                "expiration_date": result['expiration_date'],
                "reasoning": result['reasoning_bullets'],
                "confidence": result['confidence_score'],
                "market_sentiment": result['market_assessment']
            }
            
            return {
                "success": True,
                "recommendation": recommendation
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing AI response: {e}")
            logger.error(f"Raw response: {response.text}")
            return {
                "error": f"Failed to parse AI response: {str(e)}",
                "raw_response": response.text,
                "recommendation": None
            }
            
    except Exception as e:
        logger.error(f"Error getting AI recommendation: {e}")
        return {
            "error": f"Error getting AI recommendation: {str(e)}",
            "recommendation": None
        }

if __name__ == '__main__':
    """
    Application entry point.
    
    This code runs when the script is executed directly (not imported).
    It performs the following tasks:
    1. Announces startup
    2. Attempts to load cached data
    3. Opens a browser to the application URL
    4. Starts the Flask web server
    
    The application can be accessed at http://localhost:8086 by default,
    or a custom port can be specified as a command-line argument.
    """
    print("Starting S&P 500 Moving Average & Options Analysis Tool...")
    print("Loading page in browser...")
    
    # Try to load cache in advance
    cached_data = load_data_from_cache()
    if cached_data:
        with data_lock:
            sp500_data = cached_data
            last_updated = datetime.now()
            print(f"Loaded {len(cached_data)} stocks from cache")
    else:
        print("No cached data found. Use the 'Refresh Data' button to fetch data.")
    
    # Get port from command line argument if provided
    port = 8086  # Default port
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port 8086.")
    
    # Start browser in a new thread
    Thread(target=lambda: open_browser(port)).start()
    
    # Start Flask app with the specified port
    try:
        # debug=True enables auto-reloading when code changes
        # use_reloader=False prevents duplicate execution with Flask's reloader
        app.run(debug=True, use_reloader=False, port=port)
    except Exception as e:
        print(f"Error starting server on port {port}: {e}")
        print("Trying alternative port...")
        try:
            app.run(debug=True, use_reloader=False, port=port+1)
        except Exception as e:
            print(f"Error starting server on alternative port {port+1}: {e}")
            print("Please kill any other Flask applications and try again.") 