let fibChart;
let currentSymbol = 'AAPL';

async function fetchData(symbol = currentSymbol) {
    try {
        const response = await fetch(`/data?symbol=${symbol}`);
        const data = await response.json();

        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }

        currentSymbol = data.symbol;

        // Update price display
        document.getElementById('symbol').textContent = data.symbol;
        document.getElementById('price').textContent = data.current_price.toFixed(2);

        // Update ticker bar
        updateTickerBar(data.tickers_data);

        // Update Fibonacci Chart
        updateFibChart(data.fib_levels);

    } catch (e) {
        alert('Failed to fetch data.');
        console.error(e);
    }
}

function updateTickerBar(tickers) {
    const tickerBar = document.getElementById('tickerBar');
    tickerBar.innerHTML = '';

    for (const [symbol, price] of Object.entries(tickers)) {
        if (price === null) continue;
        const item = document.createElement('span');
        item.classList.add('ticker-item');
        item.textContent = `${symbol}: $${price.toFixed(2)}`;
        tickerBar.appendChild(item);
    }
}

function updateFibChart(levels) {
    const ctx = document.getElementById('fibChart').getContext('2d');

    // If chart exists, destroy before recreating
    if (fibChart) fibChart.destroy();

    const labels = Object.keys(levels);
    const values = Object.values(levels);

    fibChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Fibonacci Levels',
                data: values,
                backgroundColor: 'rgba(0, 255, 144, 0.7)',
                borderColor: 'rgba(0, 255, 144, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: false,
                    reverse: true,
                    ticks: {
                        color: '#00ff90',
                        font: { size: 14 }
                    },
                    grid: {
                        color: '#004d22'
                    }
                },
                x: {
                    ticks: {
                        color: '#00ff90',
                        font: { size: 14 }
                    },
                    grid: {
                        color: '#004d22'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#00ff90',
                        font: { size: 16 }
                    }
                }
            }
        }
    });
}

document.getElementById('tickerInput').addEventListener('keyup', (e) => {
    if (e.key === 'Enter') {
        fetchData(e.target.value.trim().toUpperCase());
    }
});

document.querySelector('button').addEventListener('click', () => {
    const inputVal = document.getElementById('tickerInput').value.trim().toUpperCase();
    if (inputVal) fetchData(inputVal);
});

// Initial fetch & auto refresh every 10 seconds
fetchData();
setInterval(() => fetchData(currentSymbol), 10000);
