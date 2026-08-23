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
            # Placeholder for DNN based super-resolution
            # e.g., using cv2.dnn_superres.DnnSuperResImpl_create()
            return frame
