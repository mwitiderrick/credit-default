# 🏦 Credit Risk Prediction (UCI Dataset)

This project predicts the likelihood of credit default using the UCI credit card dataset. It demonstrates a full machine learning pipeline including:

- Automated data ingestion and preprocessing
- Model training using XGBoost (with GPU support when available)
- Prediction with risk level assessment

## 🚀 How to Run

### 1. Build the Docker Image
```bash
docker build -t credit-risk .
```

### 2. Run the Pipeline

#### Step 1: Train the Model
First, train the model to generate the model file:

```bash
docker run --rm -v $(pwd)/output:/output credit-risk
```

#### Step 2: Run Inference
After the model is trained and saved, you can run inference:

```bash
# First, create a sample input if you haven't already
echo '[
    [20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 
     3913, 3102, 689, 0, 0, 0, 
     0, 0, 0, 0, 0, 0]
]' > sample_input.json

# Then run inference
docker run --rm \
    -v $(pwd)/output:/output \
    -v $(pwd)/sample_input.json:/app/sample_input.json \
    credit-risk python inference.py --input sample_input.json
```

The container will:
1. Download and preprocess the UCI credit card dataset
2. Train an XGBoost model with early stopping
3. Save artifacts to the mounted /output directory:
   - best_model.json: Trained XGBoost model
   - metrics.json: Model performance metrics
   - params.json: Training parameters and settings

### 🔍 Model Output Format

Example prediction output:
```json
{
  "predictions": [
    {
      "probability": 0.1224,
      "prediction": "NO DEFAULT",
      "risk_level": "LOW"
    }
  ]
}
```

Risk Levels:
- LOW: probability < 0.3
- MEDIUM: probability between 0.3 and 0.7
- HIGH: probability >= 0.7     

### 💻 GPU Support
The pipeline can run in both CPU and GPU modes:

#### Prerequisites for GPU Support:
1. NVIDIA GPU with CUDA support
2. NVIDIA drivers installed on host machine
3. Docker configured with NVIDIA runtime

To check if your system is ready for GPU support:
```bash
nvidia-smi  # Should show GPU information
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi  # Should show GPU in container
```

The model will automatically:
- Use GPU acceleration if available (`tree_method='gpu_hist'`)
- Fall back to CPU if no GPU is detected (`tree_method='hist'`)
- Log which mode it's using during training

### 📦 Output Files
All outputs are written to the mounted `/output` directory:
- `best_model.json`: Trained XGBoost model
- `metrics.json`: Model performance metrics including AUC, precision, recall
- `params.json`: Training parameters including early stopping details

### 📚 Dataset Source
UCI Credit Card Default Dataset:
https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip

