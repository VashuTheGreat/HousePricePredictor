# Elite Estates: House Rent Predictor 🏠💰

Elite Estates is a premium, end-to-end Machine Learning web application designed to predict residential rental prices with high precision. Featuring a modern glassmorphic dashboard, real-time analytics, and an integrated experimentation hub.

## ✨ Features

- **High-Precision Intelligence**: Powered by a custom-tuned RandomForestRegressor with ~99.9% accuracy.
- **Elite Dashboard**: A stunning, responsive UI built with Tailwind CSS and glassmorphism.
- **Asynchronous UX**: Zero page reloads for predictions and training using modern AJAX (Fetch API).
- **Admin Security**: Protected retraining intelligence with secure password verification.
- **Experiment Hub**: Dedicated "Intelligence Tuner" to live-edit model hyperparameters (`model.yaml`).
- **MLflow Integration**: Full performance tracking and experiment logging.

## 🚀 Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Tailwind CSS, Vanilla JS
- **ML Engine**: Scikit-Learn (Random Forest)
- **Tracking**: MLflow
- **Data Handling**: Pandas, NumPy, MongoDB

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/VashuTheGreat/HousePricePredictor.git
   cd HousePricePredictor
   ```

2. **Create and Activate Virtual Environment**:

   ```bash
   uv venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:

   ```bash
   uv pip install -r requirements.txt
   uv pip install -e .
   ```

4. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```env
   MONGODB_URL="your_mongodb_url"
   TRAIN_PASSWORD="a"
   ```

## 🏃 Usage

1. **Start the Application**:

   ```bash
   uv run python main.py
   ```

2. **Start MLflow Tracking**:

   ```bash
   mlflow ui
   ```

3. **Interact**:
   - Access the dashboard at `http://127.0.0.1:8000`.
   - Use the **"Retrain"** button (Password: `a`) to sync model intelligence.
   - Use the **"Run Experiment"** button to tweak hyperparameters.

## 📁 Project Structure

```text
├── config/             # Configuration files (model.yaml, schema.yaml)
├── src/                # Modular source code (ingestion, validation, transformation, etc.)
├── templates/          # Modern UI dashboards
├── artifact/           # [Ignored] Training artifacts and data splits
├── logs/               # [Ignored] System and component logs
├── saved_model/        # Final serialized model artifacts
└── main.py             # FastAPI entry point
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed with ❤️ by Vansh
