
# Fibonacci Retracement Levels BarPlot

A simple and interactive **Python Flask web app** to calculate and visualize Fibonacci retracement levels.  
Features a **Bloomberg-inspired dark theme**, auto-refreshing live stock price display, and clean, readable bar plots.

---

## 🚀 Overview
This tool helps you:
- **Calculate key Fibonacci retracement levels** from any two price points (High/Low).
- **View interactive bar charts** showing these levels.
- **Select stock tickers from a dropdown** to get real-time price updates (powered by Yahoo Finance API).
- **Experience Bloomberg-like dark styling** for a professional trading feel.
- **Enjoy auto-refreshing data** without reloading the page.

---

## 📦 Features

✅ User-friendly Fibonacci retracement calculator  
✅ Interactive **bar charts**  
✅ Live **stock price ticker and dropdown selection**  
✅ **Dark/Bloomberg terminal-inspired theme**  
✅ Auto-refreshing **real-time stock data**  
✅ Easy-to-run **Flask backend with HTML/JS frontend**

---

## 🛠️ Getting Started

### Prerequisites

✔️ Python 3.7+  
✔️ pip package manager  

### Installation

1. **Clone this repository:**

```bash
git clone https://github.com/GarrettOlitsky/Fibonacci_Retracement_Levels_BarPlot.git
cd Fibonacci_Retracement_Levels_BarPlot
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app:**

```bash
python app.py
```

4. Open your browser and go to:
   👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 📂 Project Structure

```
Fibonacci_Retracement_Levels_BarPlot/
│
├── app.py                # Flask backend
├── requirements.txt      # Python dependencies
├── README.md             # This file
│
├── templates/
│   └── index.html        # Main UI (HTML)
│
├── static/
│   ├── style.css         # Custom CSS (dark theme)
│   └── script.js         # JS (Chart.js, live stock data fetching)
```

---

## 🖥️ Technologies Used

* Python (Flask)
* HTML5
* CSS3 (Dark Theme)
* JavaScript (Chart.js, Fetch API)
* Yahoo Finance (yfinance)

---

## ❗ Notes

* Live stock data is fetched automatically and displayed.
* Data auto-refreshes every 60 seconds.
* For additional features or improvements, feel free to fork and contribute!

---

## 💡 Future Plans

* Candlestick charts
* User account system
* Enhanced technical indicators (RSI, MACD)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.

---

## 📜 License

[MIT](LICENSE)


