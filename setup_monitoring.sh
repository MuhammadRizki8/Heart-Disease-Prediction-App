#!/bin/bash

# Setup script untuk Heart Disease Prediction API Monitoring
# Pastikan Docker dan Docker Compose sudah terinstall

echo "=== Heart Disease API Monitoring Setup ==="

# Buat direktori yang diperlukan
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards
mkdir -p logs

# Buat konfigurasi datasource Grafana
cat > grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Buat konfigurasi dashboard Grafana
cat > grafana/provisioning/dashboards/dashboard.yml << EOF
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Buat konfigurasi Alertmanager
cat > alertmanager.yml << EOF
global:
  smtp_smarthost: '127.0.0.1:587'
  smtp_from: 'alerts@heartdisease-api.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://127.0.0.1:5001/'
EOF

# Buat requirements.txt untuk dependencies
cat > requirements.txt << EOF
flask==2.3.3
prometheus-flask-exporter==0.22.4
prometheus-client==0.17.1
psutil==5.9.5
requests==2.31.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
flask-cors
EOF

# Buat Dockerfile
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
EOF

echo "Setup completed!"
echo ""
echo "Langkah selanjutnya:"
echo "1. Pastikan file inference.py dan model (di folder outputs) ada di direktori ini"
echo "2. Edit heart-disease-dashboard.json"
echo "3. Jalankan: docker-compose up -d"
echo "4. Akses Grafana di http://127.0.0.1:3000 (admin/admin123)"
echo "5. Akses Prometheus di http://127.0.0.1:9090"
echo "6. API tersedia di http://127.0.0.1:5000"
echo "7. aplikasi prediksi dengan UI tersedia di http://127.0.0.1:5000/"
echo ""
echo "Untuk monitoring tambahan, jalankan:"
echo "python prometheus_exporter.py"