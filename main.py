import sys
import os

# Add src to path so absolute imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import VisionOSPipeline

if __name__ == "__main__":
    print("Initializing Vision OS...")
    pipeline = VisionOSPipeline(config_path="config/config.yaml")
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
        pipeline.cleanup()
