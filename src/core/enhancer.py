import cv2
import numpy as np

class ImageEnhancer:
    def __init__(self, method="fast"):
        """
        Enhancer for degraded CCTV footage.
        method: "fast" (basic denoising/sharpening) or "super_res" (OpenCV DNN super resolution - requires weights).
        """
        self.method = method
        self.kernel_sharpen = np.array([[0, -1, 0],
                                        [-1, 5,-1],
                                        [0, -1, 0]])

    def enhance(self, frame):
        """Enhances a single frame or cropped ROI."""
        if frame is None:
            return None
            
        if self.method == "fast":
            # 1. Denoise (Fast Non-Local Means Denoising)
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
            # 2. Sharpen
            sharpened = cv2.filter2D(denoised, -1, self.kernel_sharpen)
            
            # 3. Contrast / Brightness adjustment using Histogram Equalization on Y channel
            img_yuv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            
            return enhanced
        else:
            # DNN based super-resolution using cv2.dnn_superres
            try:
                import os
                import urllib.request
                
                model_path = "models/FSRCNN_x2.pb"
                if not hasattr(self, 'sr'):
                    if not os.path.exists("models"):
                        os.makedirs("models")
                        
                    # Auto-download lightweight FSRCNN model if missing
                    if not os.path.exists(model_path):
                        print(f"Downloading FSRCNN_x2.pb to {model_path}...")
                        url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
                        urllib.request.urlretrieve(url, model_path)
                        
                    self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    self.sr.readModel(model_path)
                    self.sr.setModel("fsrcnn", 2) # Algorithm=fsrcnn, Scale=2
                
                # Apply super resolution
                enhanced = self.sr.upsample(frame)
                return enhanced
            except Exception as e:
                print(f"Super Resolution failed: {e}. Falling back to fast method.")
                self.method = "fast"
                return self.enhance(frame)
