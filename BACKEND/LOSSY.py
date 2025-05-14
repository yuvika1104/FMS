import cv2
import numpy as np
import os
from PIL import Image

class LossyImageCompressor:
    def __init__(self, levels=16):
        self.levels = levels  # Number of quantization levels
    
    def _quantize_image(self, image):
        """Quantize the image to reduce pixel value levels."""
        max_value = 255
        interval = max_value // (self.levels - 1)
        quantized_image = np.round(image / interval) * interval
        return quantized_image.astype(np.uint8)
    def remove_metadata(self,image_path, output_path):
        """Remove metadata from the saved image."""
        with Image.open(image_path) as img:
            img.save(output_path, format="JPEG")
             
    def compress(self, image_path, output_folder):
        """Compress the image using lossy quantization."""
        filename, _ = os.path.splitext(os.path.basename(image_path))
        compressed_image_path = os.path.join(output_folder, f"{filename}_temp_compressed.jpg")
        final_path = os.path.join(output_folder, f"{filename}_compressed.jpeg")

        # Load the image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        # Quantize the image
        quantized_image = self._quantize_image(image)

        # Save the compressed image
        cv2.imwrite(compressed_image_path, quantized_image)

        self.remove_metadata(compressed_image_path, final_path)

    # Delete the temporary file
        os.remove(compressed_image_path)

        return final_path

