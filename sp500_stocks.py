"""
S&P 500 Stocks Module
=====================

This module provides a static list of S&P 500 stock tickers and company names,
along with helper functions to access this data.

The static list serves as a fallback when the application can't fetch the
current S&P 500 constituents from the web. It contains a selection of major
S&P 500 companies for demonstration purposes.

Functions:
    get_stock_list(): Returns the complete list of stocks
    get_stock_by_ticker(ticker): Retrieves information for a specific stock
"""

# Static list of major S&P 500 stocks for fallback/demo use
# This is not the complete S&P 500 list, just a subset of major companies
SP500_STOCKS = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "AMZN", "name": "Amazon.com Inc."},
    {"ticker": "NVDA", "name": "NVIDIA Corporation"},
    {"ticker": "GOOGL", "name": "Alphabet Inc. Class A"},
    {"ticker": "META", "name": "Meta Platforms Inc."},
    {"ticker": "BRK.B", "name": "Berkshire Hathaway Inc. Class B"},
    {"ticker": "TSLA", "name": "Tesla Inc."},
    {"ticker": "UNH", "name": "UnitedHealth Group Incorporated"},
    {"ticker": "JNJ", "name": "Johnson & Johnson"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co."},
    {"ticker": "V", "name": "Visa Inc."},
    {"ticker": "PG", "name": "Procter & Gamble Co."},
    {"ticker": "MA", "name": "Mastercard Incorporated"},
    {"ticker": "HD", "name": "Home Depot Inc."},
    {"ticker": "CVX", "name": "Chevron Corporation"},
    {"ticker": "AVGO", "name": "Broadcom Inc."},
    {"ticker": "ABBV", "name": "AbbVie Inc."},
    {"ticker": "LLY", "name": "Eli Lilly and Company"},
    {"ticker": "PFE", "name": "Pfizer Inc."},
    {"ticker": "PLTR", "name": "Palantir Technologies Inc."},
    # Add more stocks as needed...
]

def get_stock_list():
    """
    Return the complete list of S&P 500 stocks.
    
    Returns:
        list: List of dictionaries, each containing:
            - ticker (str): Stock ticker symbol
            - name (str): Company name
    """
    return SP500_STOCKS

def get_stock_by_ticker(ticker):
    """
    Return stock information for a given ticker symbol.
    
    Args:
        ticker (str): Stock ticker symbol to look up
        
    Returns:
        dict or None: Dictionary containing stock information if found,
                     otherwise None
                     
    Example:
        >>> get_stock_by_ticker("AAPL")
        {'ticker': 'AAPL', 'name': 'Apple Inc.'}
    """
    for stock in SP500_STOCKS:
        if stock["ticker"] == ticker:
            return stock
    return None 