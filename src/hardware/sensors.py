import threading
import time
import random

class HardwareSensors:
    def __init__(self, actuator, notifier, logger):
        """
        Features 8 & 9: Tamper/Vibration, Power-Loss, and Backup Battery.
        Runs in a separate thread polling continuously.
        """
        self.actuator = actuator
        self.notifier = notifier
        self.logger = logger
        
        self.battery_level = 100
        self.main_power_active = True
        self.is_tampered = False
        
        self.running = True
        self.sensor_thread = threading.Thread(target=self.poll_sensors, daemon=True)
        self.sensor_thread.start()

    def poll_sensors(self):
        while self.running:
            # Simulate random power loss (very rare)
            if random.random() < 0.001 and self.main_power_active:
                self.main_power_active = False
                self.logger.log_event("Hardware", "Main Power Lost", 1.0, "Switched to Battery")
                self.notifier.send_alert("MAIN POWER LOST - ON BACKUP BATTERY", "Hardware Alert", self.logger)
                
            # Simulate battery drain if power is lost
            if not self.main_power_active:
                self.battery_level -= 0.1
                
            # Simulate tamper/vibration (ADXL345 mock)
            if random.random() < 0.0005 and not self.is_tampered:
                self.is_tampered = True
                self.actuator.trigger_alert() # Latching Siren
                self.logger.log_event("Hardware", "Vibration/Tamper Detected", 1.0, "Siren Latched")
                self.notifier.send_alert("TAMPER DETECTED - ALARM TRIGGERED", "Security Alert", self.logger)
                
            time.sleep(1)

    def stop(self):
        self.running = False
