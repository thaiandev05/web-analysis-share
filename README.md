# Vietnamese Stock Prediction Dashboard

A full-stack application for predicting Vietnamese stock prices using machine learning with FastAPI backend and Streamlit frontend.

├── 📁 backend/
│   ├── 📁 __pycache__/ 🚫 (auto-hidden)
│   ├── 🔒 .env 🚫 (auto-hidden)
│   ├── 📄 .env.example
│   ├── 🚫 .gitignore
│   └── 🐍 main.py
├── 📁 frontend/
│   └── 🐍 dashboard.py
└── 📖 README.md

## 🚀 Features

- **Stock Price Prediction**: AI-powered predictions for Vietnamese stocks (ACB, FPT, VNM)
- **Technical Analysis**: Moving averages, RSI, MACD indicators
- **Portfolio Analytics**: Risk metrics, performance analysis, and optimization
- **Real-time Data**: Live stock data fetching with Redis caching
- **Interactive Dashboard**: User-friendly Streamlit interface
- **RESTful API**: FastAPI backend with automatic documentation

## 📊 Supported Stocks

- **ACB** - Asia Commercial Bank
- **FPT** - FPT Corporation  
- **VNM** - Vietnam Dairy Products

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern web framework for APIs
- **Redis** - In-memory caching for performance
- **scikit-learn** - Machine learning models
- **vnstock** - Vietnamese stock data provider
- **Pandas/NumPy** - Data manipulation and analysis

### Frontend
- **Streamlit** - Interactive web dashboard
- **Plotly** - Advanced data visualization
- **Requests** - API communication

## 📋 Prerequisites

- Python 3.8+
- Redis Server
- Virtual Environment (recommended)

## 🔧 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd learning_numpy/Project
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn streamlit pandas numpy vnstock scikit-learn redis plotly requests python-dotenv pydantic
```

### 4. Install and start Redis
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server

# macOS
brew install redis
brew services start redis

# Windows (using WSL or Docker)
docker run -d -p 6379:6379 redis:alpine
```

### 5. Configure environment
```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
```

## 🚀 Running the Application

### Start the Backend API
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Frontend Dashboard
```bash
# In a new terminal
cd frontend
streamlit run dashboard.py
```

## 🌐 Access Points

- **Frontend Dashboard**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **API Redoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### Stock Predictions
- `GET /prediction/{symbol}` - Get stock predictions
  - Symbols: `ACB`, `FPT`, `VNM`
  - Returns: Price predictions, technical indicators, confidence scores

### Portfolio Analysis
- `GET /portfolio` - Get portfolio metrics
  - Returns: Risk analysis, performance metrics, allocation suggestions

### Cache Management
- `GET /cache/status` - Check Redis cache status
- `DELETE /cache/clear` - Clear all cached data
- `GET /cache/keys` - List all cache keys

## 💼 Usage Examples

### API Usage
```python
import requests

# Get stock prediction
response = requests.get("http://localhost:8000/prediction/ACB")
data = response.json()

# Get portfolio analysis
portfolio = requests.get("http://localhost:8000/portfolio")
metrics = portfolio.json()
```

### Dashboard Features
1. **Stock Selection**: Choose from ACB, FPT, or VNM
2. **Prediction View**: See AI-generated price forecasts
3. **Technical Analysis**: View charts with indicators
4. **Portfolio Metrics**: Analyze risk and performance
5. **Cache Management**: Monitor and clear cache

## 🔍 Technical Details

### Machine Learning Models
- **Linear Regression**: Base prediction model
- **Feature Engineering**: Technical indicators, moving averages
- **Data Processing**: Normalization, trend analysis

### Caching Strategy
- **Redis TTL**: 1 hour for stock data
- **Key Structure**: `stock:{symbol}`, `portfolio:metrics`
- **Performance**: Reduces API calls and improves response time

### Data Sources
- **vnstock**: Primary Vietnamese stock data provider
- **Real-time Updates**: Automatic data refresh
- **Historical Data**: Training data for ML models

## 🛡️ Error Handling

The application includes comprehensive error handling:
- Redis connection failures
- Stock data unavailability
- API rate limiting
- Model prediction errors

## 📈 Performance Optimization

- **Redis Caching**: Reduces data fetching time
- **Async Operations**: Non-blocking API calls
- **Data Compression**: Efficient storage of historical data
- **Connection Pooling**: Optimized database connections

## 🐛 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check if Redis is running
   redis-cli ping
   # Should return: PONG
   ```

2. **Port Already in Use**
   ```bash
   # Find and kill process using port 8000
   lsof -ti:8000 | xargs kill -9
   ```

3. **Module Not Found**
   ```bash
   # Ensure virtual environment is activated
   which python
   pip list
   ```

4. **Stock Data Not Available**
   - Check internet connection
   - Verify stock symbol is supported
   - Check vnstock service status

## 📝 Development

### Project Structure
```
Project/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── .env.example         # Environment template
│   └── .gitignore          # Git ignore rules
├── frontend/
│   └── dashboard.py         # Streamlit dashboard
└── README.md               # This file
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please respect data provider terms of service.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock predictions are not financial advice. Always consult with financial professionals before making investment decisions.

## 📞 Support

For questions or issues, please create an issue in the repository or contact the development team.

---

**Happy Trading! 📊🚀**
