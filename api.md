# Contoh Request API Heart Disease Prediction

## 1. Health Check

```bash
curl -X GET http://127.0.0.1:5000/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-06-02T10:30:00.123456",
  "uptime_seconds": 3600.5
}
```

## 2. Get Available Models

```bash
curl -X GET http://127.0.0.1:5000/models
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "available_models": ["RandomForest", "LogisticRegression", "SVM"],
    "feature_count": 15,
    "feature_names": ["Age", "Sex", "ChestPain_ATA", "ChestPain_ASY", ...]
  },
  "timestamp": "2025-06-02T10:30:00.123456"
}
```

## 3. Single Prediction - High Risk Patient

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RandomForest",
    "data": {
      "Age": 65,
      "Sex": "M",
      "ChestPainType": "ASY",
      "RestingBP": 140,
      "Cholesterol": 250,
      "FastingBS": 1,
      "RestingECG": "ST",
      "MaxHR": 120,
      "ExerciseAngina": "Y",
      "Oldpeak": 2.0,
      "ST_Slope": "Flat"
    }
  }'
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "prediction": 1,
    "prediction_label": "Heart Disease",
    "probability": {
      "No Heart Disease": 0.25,
      "Heart Disease": 0.75
    },
    "model_used": "RandomForest",
    "inference_time_seconds": 0.0234,
    "timestamp": "2025-06-02T10:30:00.123456"
  },
  "timestamp": "2025-06-02T10:30:00.123456"
}
```

## 4. Single Prediction - Low Risk Patient

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LogisticRegression",
    "data": {
      "Age": 35,
      "Sex": "F",
      "ChestPainType": "ATA",
      "RestingBP": 120,
      "Cholesterol": 200,
      "FastingBS": 0,
      "RestingECG": "Normal",
      "MaxHR": 180,
      "ExerciseAngina": "N",
      "Oldpeak": 0.0,
      "ST_Slope": "Up"
    }
  }'
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "prediction": 0,
    "prediction_label": "No Heart Disease",
    "probability": {
      "No Heart Disease": 0.85,
      "Heart Disease": 0.15
    },
    "model_used": "LogisticRegression",
    "inference_time_seconds": 0.0187,
    "timestamp": "2025-06-02T10:30:00.123456"
  },
  "timestamp": "2025-06-02T10:30:00.123456"
}
```

## 5. Batch Prediction

```bash
curl -X POST http://127.0.0.1:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SVM",
    "data": [
      {
        "Age": 65,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 140,
        "Cholesterol": 250,
        "FastingBS": 1,
        "RestingECG": "ST",
        "MaxHR": 120,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.0,
        "ST_Slope": "Flat"
      },
      {
        "Age": 35,
        "Sex": "F",
        "ChestPainType": "ATA",
        "RestingBP": 120,
        "Cholesterol": 200,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 180,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up"
      }
    ]
  }'
```

**Response:**

```json
{
  "status": "success",
  "data": [
    {
      "sample_id": 0,
      "prediction": 1,
      "prediction_label": "Heart Disease",
      "probability": {
        "No Heart Disease": 0.3,
        "Heart Disease": 0.7
      },
      "model_used": "SVM",
      "inference_time_seconds": 0.0198,
      "timestamp": "2025-06-02T10:30:00.123456"
    },
    {
      "sample_id": 1,
      "prediction": 0,
      "prediction_label": "No Heart Disease",
      "probability": {
        "No Heart Disease": 0.88,
        "Heart Disease": 0.12
      },
      "model_used": "SVM",
      "inference_time_seconds": 0.0201,
      "timestamp": "2025-06-02T10:30:00.123456"
    }
  ],
  "batch_size": 2,
  "timestamp": "2025-06-02T10:30:00.123456"
}
```

## 6. Custom Metrics

```bash
curl -X GET http://127.0.0.1:5000/metrics/custom
```

**Response:**

```json
{
  "uptime_seconds": 3600.5,
  "total_requests": 150,
  "total_errors": 2,
  "error_rate_percent": 1.33,
  "models_loaded": 3,
  "timestamp": "2025-06-02T10:30:00.123456"
}
```

## 7. Prometheus Metrics

```bash
curl -X GET http://127.0.0.1:5000/metrics
```

**Response (Prometheus format):**

```
# HELP heart_disease_prediction_requests_total Total number of prediction requests
# TYPE heart_disease_prediction_requests_total counter
heart_disease_prediction_requests_total{model="RandomForest",endpoint="/predict",status="success"} 45.0

# HELP heart_disease_prediction_results_total Total number of predictions by result
# TYPE heart_disease_prediction_results_total counter
heart_disease_prediction_results_total{model="RandomForest",result="Heart Disease"} 23.0
heart_disease_prediction_results_total{model="RandomForest",result="No Heart Disease"} 22.0

# ... dan metrics lainnya
```

## Parameter Input yang Valid

### Categorical Fields:

- **Sex**: "M" atau "F"
- **ChestPainType**: "ATA", "NAP", "ASY", "TA"
- **RestingECG**: "Normal", "ST", "LVH"
- **ExerciseAngina**: "Y" atau "N"
- **ST_Slope**: "Up", "Flat", "Down"

### Numerical Fields:

- **Age**: 28-77
- **RestingBP**: 94-200
- **Cholesterol**: 85-603
- **MaxHR**: 60-202
- **Oldpeak**: -2.6 sampai 6.2
- **FastingBS**: 0 atau 1

### Models yang Tersedia:

- "RandomForest" (default)
- "LogisticRegression"
- "SVM"
