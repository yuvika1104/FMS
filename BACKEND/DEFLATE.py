import zlib
import os
import numpy as np
import cv2
import struct # For packing/unpacking number of channels

class DeflateCoding:
    def compress(self, image_path, output_folder):
        """
        Compresses an image using zlib (Deflate algorithm).
        Stores image dimensions (height, width, channels) and then the compressed data.
        Output file has a .zlib extension.
        """
        filename, _ = os.path.splitext(os.path.basename(image_path))
        # Ensure the output extension is .zlib as expected by the frontend/backend mapping
        compressed_path = os.path.join(output_folder, filename + ".zlib")

        # Load the image. cv2.IMREAD_COLOR loads as BGR.
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        image_shape = image.shape
        height = image_shape[0]
        width = image_shape[1]
        
        # Determine number of channels
        if len(image_shape) == 3:
            channels = image_shape[2]
        else: # Grayscale image (should not happen with IMREAD_COLOR, but good for robustness)
            channels = 1
            # If it was truly grayscale and read as 2D, we might need to reshape for consistency
            # However, IMREAD_COLOR usually makes it 3-channel even if grayscale.

        image_data = image.flatten()
        compressed_data = zlib.compress(image_data.tobytes(), level=9) # zlib.Z_BEST_COMPRESSION

        # Save the compressed data and metadata
        with open(compressed_path, "wb") as compressed_file:
            # Store height (4 bytes), width (4 bytes), channels (1 byte)
            compressed_file.write(struct.pack('!I', height))      # I for unsigned int
            compressed_file.write(struct.pack('!I', width))       # I for unsigned int
            compressed_file.write(struct.pack('!B', channels))    # B for unsigned char
            compressed_file.write(compressed_data)

        return compressed_path

    def decompress(self, compressed_path, output_folder):
        """
        Decompresses a .zlib file (created by this class's compress method)
        back into an image and saves it as a PNG.
        """
        filename, _ = os.path.splitext(os.path.basename(compressed_path))
        # Ensure the output is a standard, viewable image format like PNG
        decompressed_path = os.path.join(output_folder, filename + "_decompressed.png")

        # Load the compressed data
        with open(compressed_path, "rb") as compressed_file:
            height = struct.unpack('!I', compressed_file.read(4))[0]
            width = struct.unpack('!I', compressed_file.read(4))[0]
            channels = struct.unpack('!B', compressed_file.read(1))[0]
            compressed_data = compressed_file.read()

        # Decompress the image data
        image_data_bytes = zlib.decompress(compressed_data)
        
        # Determine the target shape for reconstruction
        if channels == 1:
            target_shape = (height, width)
        else:
            target_shape = (height, width, channels)
            
        image_array = np.frombuffer(image_data_bytes, dtype=np.uint8).reshape(target_shape)

        # Save the decompressed image
        # cv2.imwrite handles both grayscale and color images correctly based on array shape.
        if not cv2.imwrite(decompressed_path, image_array):
            raise RuntimeError(f"Failed to save decompressed image to {decompressed_path}")

        return decompressed_path
