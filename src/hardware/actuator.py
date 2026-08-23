import time
import os
import yaml

class Actuator:
    def __init__(self, config_path="../../config/config.yaml"):
        # Load config to check if mock or edge
        self.mode = "mock"
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.mode = config.get('system', {}).get('mode', 'mock')
                hardware_cfg = config.get('hardware', {})
                self.barrier_pin = hardware_cfg.get('barrier_pin', 17)
                self.led_pin = hardware_cfg.get('led_pin', 22)
        except Exception as e:
            print(f"Failed to load config, defaulting to mock mode. Error: {e}")
            
        print(f"Actuator initialized in {self.mode} mode.")
        
        if self.mode == "edge":
            try:
                from gpiozero import LED, Servo
                self.led = LED(self.led_pin)
                self.barrier = Servo(self.barrier_pin)
                print("GPIO hardware mapped successfully.")
            except ImportError:
                print("gpiozero not found. Falling back to mock mode.")
                self.mode = "mock"
                
        # Feature 10: Latching Siren
        self.siren_latched = False
                
    def open_barrier(self):
        print(">>> ACTION: Opening Barrier")
        if self.mode == "edge":
            self.barrier.max() # Open
            time.sleep(5)
            self.barrier.min() # Close
            
    def trigger_alert(self):
        if self.siren_latched:
            return # Already latched
            
        print("!!! ALERT !!! SIREN LATCHED ON")
        self.siren_latched = True
        if self.mode == "edge":
            self.led.on() # Simulating siren with LED staying ON

    def disarm_siren(self):
        """Feature 11: Remote-First Disarm"""
        print(">>> ACTION: Disarming Siren")
        self.siren_latched = False
        if self.mode == "edge":
            self.led.off()
