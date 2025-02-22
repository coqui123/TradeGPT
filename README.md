# TradeGPT 🚀 

TradeGPT is a full-stack cryptocurrency trading application that combines a modern Fresh (Deno) frontend with a Python (FASTAPI) backend for Coinbase integration and Azure AI Services for intelligent trading analysis. 💹

## Project Structure 📁

```
tradegpt/              # Frontend (Fresh/Deno)
├── islands/           # Interactive components
├── routes/            # Page routes
├── components/        # Shared components
├── static/           # Static assets
└── ...

tradegpt-backend/     # Backend (Python/FASTAPI)
├── coinbase-tradegpt-server.py  # Main server
└── .env              # Environment configuration
```

## Prerequisites ✅

- [Deno](https://deno.land/manual/getting_started/installation) for the frontend 🦕
- [Python 3.x](https://www.python.org/downloads/) for the backend 🐍
- Coinbase API credentials 🔑
- Local LLM or API KEY for AI Services (GPT-4o-mini, GPT-3.0-mini, sonar-reasoning, sonar-pro, sonar-reasoning-pro, Phi-3.5-MoE) 🤖

## Frontend Setup 🎨

1. Navigate to the frontend directory:
```bash
cd tradegpt
```

2. Start the development server:
```bash
deno task start
```

The frontend will be available at `http://localhost:8000` 🌐

## Backend Setup ⚙️

1. Navigate to the backend directory:
```bash
cd tradegpt-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in `.env`:
```
# COINBASE API
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret

# LLM API
LLM_API_ENDPOINT=your_llm_endpoint
LLM_API_TOKEN=your_llm_api_key
LLM_API_MODEL_NAME=your_llm_model_name
```

4. Run the backend server:
```bash
python coinbase-tradegpt-server.py
```

## Features ✨

- Real-time cryptocurrency price tracking 📊
- Interactive trading charts 📈
- Coinbase integration for live trading 💱
- Modern UI with Tailwind CSS 🎯

## Development 🛠️

The project uses:
- Fresh framework for the frontend 🌟
- Tailwind CSS for styling 💅
- Python for backend services 🐍
- Coinbase API for trading functionality 💰

## Contributing 🤝

1. Fork the repository 🍴
2. Create your feature branch 🌿
3. Commit your changes 💾
4. Push to the branch 🚀
5. Create a new Pull Request ✅

## Images 📸
![image](https://github.com/user-attachments/assets/348fdca7-29f2-46d7-b86d-4f49342f0ccd)

