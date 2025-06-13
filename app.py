from flask import Flask, render_template, jsonify, request
import yfinance as yf

app = Flask(__name__)

# Fibonacci retracement calculator
def calculate_fibonacci_retracements(high, low):
    diff = high - low
    return {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100.0%": low
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    symbol = request.args.get('symbol', 'AAPL').upper()
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d')
        if hist.empty:
            raise Exception("No data found")

        high = hist['High'].max()
        low = hist['Low'].min()
        current_price = hist['Close'][-1]

        fib_levels = calculate_fibonacci_retracements(high, low)

        # For ticker bar, a small set of popular stocks (simulate live data)
        popular_stocks = ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN', 'NFLX', 'NVDA']
        tickers_data = {}
        for s in popular_stocks:
            t = yf.Ticker(s)
            d = t.history(period='1d')
            price = d['Close'][-1] if not d.empty else None
            tickers_data[s] = price

        return jsonify({
            "symbol": symbol,
            "current_price": current_price,
            "high": high,
            "low": low,
            "fib_levels": fib_levels,
            "tickers_data": tickers_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
