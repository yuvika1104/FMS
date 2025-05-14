import zlib
import os
import numpy as np
import cv2


class DeflateCoding:
    def compress(self, image_path, output_folder):
        filename, _ = os.path.splitext(os.path.basename(image_path))
        compressed_path = os.path.join(output_folder, filename + ".zlib")

        # Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        # Flatten the image data and compress it using zlib
        image_shape = image.shape
        image_data = image.flatten()
        compressed_data = zlib.compress(image_data.tobytes(), level=9)

        # Save the compressed data and metadata
        with open(compressed_path, "wb") as compressed_file:
            compressed_file.write(image_shape[0].to_bytes(4, byteorder="big"))
            compressed_file.write(image_shape[1].to_bytes(4, byteorder="big"))
            compressed_file.write(compressed_data)

        return compressed_path

    def decompress(self, compressed_path, output_folder):
        filename, _ = os.path.splitext(os.path.basename(compressed_path))
        decompressed_path = os.path.join(output_folder, filename + "_decompressed.png")

        # Load the compressed data
        with open(compressed_path, "rb") as compressed_file:
            height = int.from_bytes(compressed_file.read(4), byteorder="big")
            width = int.from_bytes(compressed_file.read(4), byteorder="big")
            compressed_data = compressed_file.read()

        # Decompress the image data
        image_data = zlib.decompress(compressed_data)
        image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width,3))

        # Save the decompressed image
        print("Image shape after decompression:", image_array.shape)
        print(decompressed_path)
        cv2.imwrite(decompressed_path, image_array)

        return decompressed_path
