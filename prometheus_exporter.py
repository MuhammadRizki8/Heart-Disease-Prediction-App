#!/usr/bin/env python3
"""
Prometheus Exporter untuk Heart Disease Prediction API
Mengumpulkan metrics tambahan untuk monitoring sistem
"""

import time
import psutil
import requests
import json
from prometheus_client import start_http_server, Gauge, Counter, Info
import logging
from datetime import datetime
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage')
SYSTEM_NETWORK_SENT = Counter('system_network_bytes_sent_total', 'Total bytes sent')
SYSTEM_NETWORK_RECV = Counter('system_network_bytes_recv_total', 'Total bytes received')
API_AVAILABILITY = Gauge('api_availability', 'API availability (1=up, 0=down)')
API_RESPONSE_TIME = Gauge('api_health_response_time_seconds', 'API health check response time')
SYSTEM_LOAD_AVERAGE = Gauge('system_load_average', 'System load average', ['period'])
SYSTEM_UPTIME = Gauge('system_uptime_seconds', 'System uptime in seconds')
PYTHON_PROCESSES = Gauge('python_processes_count', 'Number of Python processes running')

class SystemMetricsExporter:
    def __init__(self, api_url="http://127.0.0.1:5000", port=8000):
        self.api_url = api_url
        self.port = port
        self.previous_network_io = psutil.net_io_counters()
        
    def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            SYSTEM_DISK_USAGE.set(disk_percent)
            
            # Network I/O
            current_network_io = psutil.net_io_counters()
            if hasattr(self, 'previous_network_io'):
                bytes_sent = current_network_io.bytes_sent - self.previous_network_io.bytes_sent
                bytes_recv = current_network_io.bytes_recv - self.previous_network_io.bytes_recv
                SYSTEM_NETWORK_SENT.inc(max(0, bytes_sent))
                SYSTEM_NETWORK_RECV.inc(max(0, bytes_recv))
            self.previous_network_io = current_network_io
            
            # Load average
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
                SYSTEM_LOAD_AVERAGE.labels(period='1min').set(load_avg[0])
                SYSTEM_LOAD_AVERAGE.labels(period='5min').set(load_avg[1])
                SYSTEM_LOAD_AVERAGE.labels(period='15min').set(load_avg[2])
            
            # System uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            SYSTEM_UPTIME.set(uptime)
            
            # Python processes count
            python_processes = 0
            for proc in psutil.process_iter(['name']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            PYTHON_PROCESSES.set(python_processes)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def check_api_availability(self):
        """Check API availability and response time"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                API_AVAILABILITY.set(1)
                API_RESPONSE_TIME.set(response_time)
                logger.debug(f"API health check passed in {response_time:.3f}s")
            else:
                API_AVAILABILITY.set(0)
                logger.warning(f"API health check failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            API_AVAILABILITY.set(0)
            logger.error(f"API health check failed: {e}")
    
    def collect_metrics(self):
        """Main metrics collection function"""
        while True:
            try:
                self.collect_system_metrics()
                self.check_api_availability()
                time.sleep(10)  # Collect metrics every 10 seconds
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(10)
    
    def start(self):
        """Start the exporter"""
        logger.info(f"Starting Prometheus exporter on port {self.port}")
        start_http_server(self.port)
        
        # Start metrics collection in background thread
        metrics_thread = threading.Thread(target=self.collect_metrics, daemon=True)
        metrics_thread.start()
        
        logger.info(f"Metrics available at http://127.0.0.1:{self.port}/metrics")
        logger.info(f"Monitoring API at {self.api_url}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down exporter...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Heart Disease API Prometheus Exporter')
    parser.add_argument('--api-url', default='http://127.0.0.1:5000', 
                       help='Heart Disease API URL')
    parser.add_argument('--port', type=int, default=8000, 
                       help='Exporter port')
    
    args = parser.parse_args()
    
    exporter = SystemMetricsExporter(api_url=args.api_url, port=args.port)
    exporter.start()