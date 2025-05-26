import cv2
import numpy as np
import os
from PIL import Image

class ColorLossyImageCompressor:
    def __init__(self, levels=16):
        self.levels = levels  # Number of quantization levels
        self.mapping = {}  # Stores mapping for quantization

    def _quantize_image(self, image):
        """Quantize the image to reduce pixel value levels."""
        max_value = 255
        interval = max_value // (self.levels - 1)
        quantized_image = np.round(image / interval) * interval # quantized value
        return quantized_image.astype(np.uint8) #8 bit integer
    
    def remove_metadata(self,image_path, output_path):
        """Remove metadata from the saved image."""
        with Image.open(image_path) as img:
            img.save(output_path, format="JPEG") 

    def compress(self, image_path, output_folder):
        """Compress the image using lossy quantization."""
        filename, _ = os.path.splitext(os.path.basename(image_path))
        compressed_image_path = os.path.join(output_folder, f"{filename}_temp_compressed.jpg")
        final_path = os.path.join(output_folder, f"{filename}_compressed.jpeg")

        # Load the image in color
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        # Quantize each channel independently
        quantized_image = np.zeros_like(image) #new array of zeros with same dimensions
        for channel in range(3):  # Iterate over RGB channels
            quantized_image[:, :, channel] = self._quantize_image(image[:, :, channel])
        cv2.imwrite(compressed_image_path, quantized_image)
        # Save the compressed image
        self.remove_metadata(compressed_image_path, final_path)

        # Delete the temporary file
        os.remove(compressed_image_path)

        return final_path