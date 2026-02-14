# Short-Term Electricity Load Forecasting

This project reproduces and evaluates deep learning models (LSTM, CNN, and Ensemble) for short-term electricity load forecasting based on the referenced research paper.

## 📁 Project Structure

electricity_forecasting/
│
├── main.py
├── config.py
│
├── data/
│ ├── dayton.csv
│ └── houston.csv
│
├── preprocessing/
├── models/
├── training/
├── evaluation/
│
├── saved_models/
├── results/
└── plots/


## ⚙️ Requirements

- Python 3.10 or 3.11
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib

Install dependencies:

pip install tensorflow scikit-learn pandas numpy matplotlib


## 🚀 How to Run
1️⃣ Train LSTM Model
python main.py --model lstm --dataset dayton

2️⃣ Train CNN Model
python main.py --model cnn --dataset dayton

3️⃣ Run Ensemble Model
⚠️ Make sure LSTM and CNN are trained first.
python main.py --model ensemble --dataset dayton

## 🧠 Workflow Summary
Load → Split → Scale → Sequence → Build Model → Train → Predict → Inverse Scale → Evaluate → Save


## 📊 Outputs
After running, the project generates:

📁 saved_models/
lstm_dayton.keras
cnn_dayton.keras

📁 results/
JSON files containing:
RMSE
MAE
MAPE
N-RMSE
N-MAE

📁 plots/
Prediction vs Actual graph
Training loss curves

📈 Evaluation Metrics
The following metrics are computed:
RMSE
MAE
MAPE
N-RMSE
N-MAE
These match the evaluation methodology used in the paper.

## 🔁 Reproducibility

Train/Validation/Test split follows paper specification.

Scaling is applied using training data only.

Sliding window: 168-hour input → 24-hour forecast.

Models are saved for reproducible evaluation.