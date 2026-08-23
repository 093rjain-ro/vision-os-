import paho.mqtt.client as mqtt
import json
import yaml

class AlertNotifier:
    def __init__(self, broker="mqtt.eclipseprojects.io", topic="vision_os/alerts", config_path="config/config.yaml"):
        """Feature 7 & 12: Cellular-Based Alerting with JSON payload and Approximate Location"""
        self.broker = broker
        self.topic = topic
        
        # Load mock location
        try:
            with open(config_path, 'r') as f:
                self.location = yaml.safe_load(f).get('alerting', {}).get('mock_lat_lon', 'unknown')
        except:
            self.location = 'unknown'
            
        self.client = mqtt.Client()
        self.connected = False
        
        try:
            self.client.connect(self.broker, 1883, 60)
            print(f"Connected to MQTT broker at {self.broker}")
            self.client.loop_start()
            self.connected = True
        except Exception as e:
            print(f"MQTT Connection failed: {e}. Running in console-only mode.")
            self.client = None

    def send_alert(self, message, event_type="Security Alert", logger=None):
        payload = {
            "event": event_type,
            "message": message,
            "location": self.location
        }
        payload_json = json.dumps(payload)
        print(f"=== SENDING CELLULAR ALERT: {payload_json} ===")
        
        if self.connected and self.client:
            try:
                self.client.publish(self.topic, payload_json)
            except Exception as e:
                print(f"Failed to publish MQTT message: {e}")
                if logger: logger.queue_alert(payload_json) # Feature 7: Store-and-forward
        else:
            if logger: logger.queue_alert(payload_json)
                
    def __del__(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
