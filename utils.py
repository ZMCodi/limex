import requests
import json
import time
import pandas as pd
import dotenv, os

dotenv.load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')
ACCOUNT_NUMBER = os.getenv('ACCOUNT_NUMBER')

# Endpoints
AUTH_URL = 'https://auth.lime.co/connect/token'
BALANCE_URL = 'https://api.lime.co/accounts'
ORDER_URL = 'https://api.lime.co/orders/place'
PRICE_HIST_URL = 'https://api.lime.co/marketdata/history'

# Get Access Token
def get_access_token():
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'grant_type': 'password',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'username': USERNAME,
        'password': PASSWORD
    }

    response = requests.post(AUTH_URL, headers=headers, data=data)

    if response.status_code == 200:
        token_data = response.json()
        return token_data['access_token']
    else:
        raise Exception(f"Failed to get token: {response.status_code}, {response.text}")

# Fetch OHLC Data by ticker symbol
def fetch_price_data(symbol, access_token, days_back=3, period='minute_5'):
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    # Calculate UNIX timestamps for the last n days (needed for 'from' and 'to')
    now = int(time.time())
    from_time = now - (days_back * 24 * 60 * 60)

    params = {
        'symbol': symbol,
        'period': period,  # The supported periods are: minute, minute_5, minute_15, minute_30, hour, day, week, month, quarter, year
        'from': from_time,
        'to': now
    }

    response = requests.get(PRICE_HIST_URL, headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch price data: {response.status_code}, {response.text}")

    candles = response.json()

    # Convert candle data to a DataFrame for easy processing
    df = pd.DataFrame(candles)
    df.drop(columns=['period'], inplace=True) # clean up and show only OHLC data

    # Convert Unix timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    return df

# Place an Order
def place_order(access_token, order_payload):
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.post(ORDER_URL, headers=headers, json=order_payload)

    if response.status_code == 200:
        print("Order placed successfully:")
        print(response.json())
    else:
        raise Exception(f"Failed to place order: {response.status_code}, {response.text}")

# Get Account Balances
def get_account_balances(access_token):
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(BALANCE_URL, headers=headers)

    if response.status_code == 200:
        accounts = response.json()
        print("Accounts and Balances:")
        print(json.dumps(accounts, indent=2))
        return accounts
    else:
        raise Exception(f"Failed to fetch balances: {response.status_code}, {response.text}")

def execute_trade(access_token, symbol, side, quantity):
    order_payload = {
        "account_number": ACCOUNT_NUMBER,
        "symbol": symbol,
        "quantity": quantity,
        #"price": 250.00, #price only for limit order
        "time_in_force": "day",
        "order_type": "market",
        "side": side,
        "exchange": "auto"
    }
    place_order(access_token, order_payload)
