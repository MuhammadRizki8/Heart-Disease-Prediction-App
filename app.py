"""
Flask API Server untuk Heart Disease Prediction
==============================================
Server API yang menyediakan endpoint untuk prediksi penyakit jantung
dengan monitoring metrics untuk Prometheus
"""

from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from flask_cors import CORS
import time
import json
import os
import logging
from datetime import datetime
import sys

# Import inference system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import HeartDiseasePredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Heart Disease Prediction API', version='1.0.0')

# Custom Prometheus Metrics
# 1. Request counters
prediction_requests_total = Counter(
    'heart_disease_prediction_requests_total',
    'Total number of prediction requests',
    ['model', 'endpoint', 'status']
)

# 2. Prediction results counter
prediction_results_total = Counter(
    'heart_disease_prediction_results_total',
    'Total number of predictions by result',
    ['model', 'result']
)

# 3. Response time histogram
prediction_duration_seconds = Histogram(
    'heart_disease_prediction_duration_seconds',
    'Time spent on predictions',
    ['model', 'endpoint']
)

# 4. Model performance gauge
model_accuracy_gauge = Gauge(
    'heart_disease_model_accuracy',
    'Model accuracy from last evaluation',
    ['model']
)

# 5. Active connections gauge
active_connections = Gauge(
    'heart_disease_api_active_connections',
    'Number of active connections'
)

# 6. Memory usage gauge
memory_usage_bytes = Gauge(
    'heart_disease_api_memory_usage_bytes',
    'Memory usage in bytes'
)

# 7. CPU usage gauge
cpu_usage_percent = Gauge(
    'heart_disease_api_cpu_usage_percent',
    'CPU usage percentage'
)

# 8. Error rate gauge
error_rate_percent = Gauge(
    'heart_disease_api_error_rate_percent',
    'Error rate percentage in last 5 minutes'
)

# 9. Batch prediction size histogram
batch_size_histogram = Histogram(
    'heart_disease_batch_prediction_size',
    'Size of batch predictions',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500]
)

# 10. Model loading status
model_loading_status = Gauge(
    'heart_disease_model_loading_status',
    'Model loading status (1=loaded, 0=not loaded)',
    ['model']
)

# 11. Patient age distribution
patient_age_histogram = Histogram(
    'heart_disease_patient_age_distribution',
    'Distribution of patient ages',
    buckets=[20, 30, 40, 50, 60, 70, 80, 90]
)

# 12. High risk predictions gauge
high_risk_predictions_total = Counter(
    'heart_disease_high_risk_predictions_total',
    'Total number of high risk predictions (probability > 0.7)',
    ['model']
)

# Global variables
predictor = None
request_count = 0
error_count = 0
start_time = time.time()

def get_system_metrics():
    """Get system metrics for monitoring"""
    import psutil
    process = psutil.Process()
    
    # Update memory usage
    memory_info = process.memory_info()
    memory_usage_bytes.set(memory_info.rss)
    
    # Update CPU usage
    cpu_percent = process.cpu_percent()
    cpu_usage_percent.set(cpu_percent)
    
    # Calculate error rate (simplified)
    global request_count, error_count
    if request_count > 0:
        error_rate = (error_count / request_count) * 100
        error_rate_percent.set(error_rate)

def initialize_predictor():
    """Initialize the predictor and load models"""
    global predictor
    try:
        predictor = HeartDiseasePredictor()
        logger.info("Predictor initialized successfully")
        
        # Update model loading status
        for model_name in predictor.models.keys():
            model_loading_status.labels(model=model_name).set(1)
        
        # Load model performance metrics if available
        model_info = predictor.get_model_info()
        if 'model_performance' in model_info:
            for model_name, perf in model_info['model_performance'].items():
                if 'accuracy' in perf:
                    model_accuracy_gauge.labels(model=model_name).set(perf['accuracy'])
                    
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        # Set all models as not loaded
        for model_name in ['RandomForest', 'LogisticRegression', 'SVM']:
            model_loading_status.labels(model=model_name).set(0)

initialize_predictor()

@app.before_request
def before_request():
    """Track active connections"""
    active_connections.inc()
    get_system_metrics()

@app.after_request
def after_request(response):
    """Track active connections and request metrics"""
    active_connections.dec()
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global request_count
    request_count += 1
    
    prediction_requests_total.labels(
        model='none', 
        endpoint='/health', 
        status='success'
    ).inc()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': time.time() - start_time
    })

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models and their information"""
    global request_count
    request_count += 1
    
    try:
        if predictor is None:
            raise Exception("Predictor not initialized")
        
        model_info = predictor.get_model_info()
        
        prediction_requests_total.labels(
            model='none', 
            endpoint='/models', 
            status='success'
        ).inc()
        
        return jsonify({
            'status': 'success',
            'data': model_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        global error_count
        error_count += 1
        
        prediction_requests_total.labels(
            model='none', 
            endpoint='/models', 
            status='error'
        ).inc()
        
        logger.error(f"Error getting models: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    global request_count
    request_count += 1
    
    start_time_req = time.time()
    model_name = request.json.get('model', 'RandomForest')
    
    try:
        if predictor is None:
            raise Exception("Predictor not initialized")
        
        # Get input data
        input_data = request.json.get('data')
        if not input_data:
            raise ValueError("No input data provided")
        
        # Track patient age distribution
        if 'Age' in input_data:
            patient_age_histogram.observe(input_data['Age'])
        
        # Make prediction
        with prediction_duration_seconds.labels(
            model=model_name, 
            endpoint='/predict'
        ).time():
            result = predictor.predict(input_data, model_name)
        
        # Track metrics
        prediction_requests_total.labels(
            model=model_name, 
            endpoint='/predict', 
            status='success'
        ).inc()
        
        prediction_results_total.labels(
            model=model_name,
            result=result['prediction_label']
        ).inc()
        
        # Track high risk predictions
        if result.get('probability') and result['probability'].get('Heart Disease', 0) > 0.7:
            high_risk_predictions_total.labels(model=model_name).inc()
        
        return jsonify({
            'status': 'success',
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        global error_count
        error_count += 1
        
        prediction_requests_total.labels(
            model=model_name, 
            endpoint='/predict', 
            status='error'
        ).inc()
        
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    global request_count
    request_count += 1
    
    start_time_req = time.time()
    model_name = request.json.get('model', 'RandomForest')
    
    try:
        if predictor is None:
            raise Exception("Predictor not initialized")
        
        # Get batch data
        batch_data = request.json.get('data')
        if not batch_data or not isinstance(batch_data, list):
            raise ValueError("No batch data provided or data is not a list")
        
        # Track batch size
        batch_size_histogram.observe(len(batch_data))
        
        # Track patient ages
        for data in batch_data:
            if 'Age' in data:
                patient_age_histogram.observe(data['Age'])
        
        # Make batch prediction
        with prediction_duration_seconds.labels(
            model=model_name, 
            endpoint='/predict/batch'
        ).time():
            results = predictor.predict_batch(batch_data, model_name)
        
        # Track metrics
        prediction_requests_total.labels(
            model=model_name, 
            endpoint='/predict/batch', 
            status='success'
        ).inc()
        
        # Track individual results
        for result in results:
            if 'prediction_label' in result:
                prediction_results_total.labels(
                    model=model_name,
                    result=result['prediction_label']
                ).inc()
                
                # Track high risk predictions
                if result.get('probability') and result['probability'].get('Heart Disease', 0) > 0.7:
                    high_risk_predictions_total.labels(model=model_name).inc()
        
        return jsonify({
            'status': 'success',
            'data': results,
            'batch_size': len(batch_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        global error_count
        error_count += 1
        
        prediction_requests_total.labels(
            model=model_name, 
            endpoint='/predict/batch', 
            status='error'
        ).inc()
        
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    global request_count
    request_count += 1
    
    # Generate Prometheus format metrics
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/metrics/custom', methods=['GET'])
def custom_metrics():
    """Custom metrics endpoint for additional monitoring (JSON format)"""
    global request_count, error_count
    
    uptime = time.time() - start_time
    error_rate = (error_count / max(request_count, 1)) * 100
    
    return jsonify({
        'uptime_seconds': uptime,
        'total_requests': request_count,
        'total_errors': error_count,
        'error_rate_percent': error_rate,
        'models_loaded': len(predictor.models) if predictor else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)