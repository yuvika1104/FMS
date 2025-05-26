# DCTCompression.py
import os
import cv2
import numpy as np
import struct
import zlib

class DCTCompression:
    """
    Revised DCT-based image compression (lossy).
    - Converts image to grayscale.
    - Processes in 8x8 blocks.
    - Applies 2D DCT to each block.
    - Quantizes DCT coefficients using a standard quality-based matrix.
    - Implements Zigzag scanning and Run-Length Encoding (RLE) of quantized coefficients.
    - Compresses the RLE-encoded data using zlib.

    Compressed file format (.dctz):
    1. Original Extension Length (USHORT - 2 bytes)
    2. Original Extension (UTF-8 string)
    3. Image Height (UINT - 4 bytes)
    4. Image Width (UINT - 4 bytes)
    5. Block Size (UBYTE - 1 byte) - Currently fixed at 8
    6. Quantization Quality (UBYTE - 1 byte)
    7. Compressed RLE-encoded Quantized Coefficients (using zlib)
       RLE format: Each token is (run_length_of_zeros, value).
       An End Of Block (EOB) is signaled by a (0, 0) pair if not all 64 coefficients
       in the block are explicitly encoded.
    """
    DEFAULT_BLOCK_SIZE = 8

    def __init__(self):
        pass

    def _get_quantization_matrix(self, quality, block_size):
        """
        Generates a quantization matrix based on JPEG standard luminance matrix, scaled by quality.
        """
        if quality < 1: quality = 1
        if quality > 100: quality = 100

        # Standard JPEG Luminance Quantization Matrix
        std_luminance_matrix = np.array([
            [16,  11,  10,  16,  24,  40,  51,  61],
            [12,  12,  14,  19,  26,  58,  60,  55],
            [14,  13,  16,  24,  40,  57,  69,  56],
            [14,  17,  22,  29,  51,  87,  80,  62],
            [18,  22,  37,  56,  68, 109, 103,  77],
            [24,  35,  55,  64,  81, 104, 113,  92],
            [49,  64,  78,  87, 103, 121, 120, 101],
            [72,  92,  95,  98, 112, 100, 103,  99]
        ], dtype=np.float32)

        if block_size != 8:
            # For non-8x8 blocks, use a simple scaling (less optimal)
            # This part is mostly a placeholder as 8x8 is standard for DCT in JPEG
            q_matrix = np.ones((block_size, block_size), dtype=np.float32)
            for i in range(block_size):
                for j in range(block_size):
                    q_matrix[i, j] = 1 + (i + j) * ((100 - quality) / 10.0 + 1) # Basic scaling
            q_matrix[q_matrix < 1] = 1
            q_matrix[q_matrix > 255] = 255
            return q_matrix.astype(np.uint8)

        # Scale the standard matrix based on quality
        if quality < 50:
            scale_factor = 5000.0 / quality
        else:
            scale_factor = 200.0 - quality * 2.0
        
        q_matrix = (std_luminance_matrix * scale_factor + 50.0) / 100.0 # Adding 50 for rounding before int conversion
        q_matrix[q_matrix < 1.0] = 1.0 # Ensure no zero values, min quantization step is 1
        q_matrix[q_matrix > 255.0] = 255.0 # Cap at 255
        
        return q_matrix.astype(np.uint8)

    def _zigzag_scan(self, matrix):
        """
        Performs zigzag scan on a square matrix (optimized for 8x8).
        """
        rows, cols = matrix.shape
        if rows != 8 or cols != 8: # Fallback for non-8x8, though not typical for this DCT use
            return matrix.flatten()

        result = np.empty(rows * cols, dtype=matrix.dtype)
        r, c = 0, 0
        idx = 0
        direction = 1  # 1 for up-right, -1 for down-left

        for _ in range(rows * cols):
            result[idx] = matrix[r, c]
            idx += 1

            if direction == 1:  # Moving up-right
                if c == cols - 1:  # Hit right edge
                    r += 1
                    direction = -1
                elif r == 0:  # Hit top edge
                    c += 1
                    direction = -1
                else:
                    r -= 1
                    c += 1
            else:  # Moving down-left (direction == -1)
                if r == rows - 1:  # Hit bottom edge
                    c += 1
                    direction = 1
                elif c == 0:  # Hit left edge
                    r += 1
                    direction = 1
                else:
                    r += 1
                    c -= 1
        return result

    def _inverse_zigzag_scan(self, zigzag_list, block_shape=(8, 8)):
        """
        Performs inverse zigzag scan to reconstruct a square matrix (optimized for 8x8).
        """
        rows, cols = block_shape
        if rows != 8 or cols != 8 or len(zigzag_list) != rows * cols: # Fallback
            return np.array(zigzag_list).reshape(block_shape)

        matrix = np.empty(block_shape, dtype=zigzag_list.dtype)
        r, c = 0, 0
        idx = 0
        direction = 1  # 1 for up-right, -1 for down-left

        for _ in range(rows * cols):
            matrix[r, c] = zigzag_list[idx]
            idx += 1

            if direction == 1:  # Moving up-right
                if c == cols - 1:
                    r += 1
                    direction = -1
                elif r == 0:
                    c += 1
                    direction = -1
                else:
                    r -= 1
                    c += 1
            else:  # Moving down-left (direction == -1)
                if r == rows - 1:
                    c += 1
                    direction = 1
                elif c == 0:
                    r += 1
                    direction = 1
                else:
                    r += 1
                    c -= 1
        return matrix

    def _rle_encode(self, zigzag_coeffs):
        """
        Encodes zigzag scanned coefficients using RLE.
        Format: list of (run_of_zeros, value) pairs.
        EOB is (0,0) if the block ends early (i.e., trailing zeros).
        """
        encoded_data = []
        zero_run_length = 0
        
        for coeff in zigzag_coeffs:
            if coeff == 0:
                zero_run_length += 1
            else:
                encoded_data.append((zero_run_length, int(coeff))) # Ensure coeff is standard int
                zero_run_length = 0
        
        # If the last coefficients were zeros, the zero_run_length might not have been written.
        # JPEG's EOB marker (0,0) is used if the last non-zero AC coefficient is not the last coefficient in the block.
        # If all coefficients after the last non-zero one are zero, we add EOB.
        # If the entire block is zeros, RLE will be empty, so add EOB.
        # If the last element itself is non-zero, no EOB is needed from RLE perspective
        # as the block is full of significant data up to the end.
        
        # Check if the last non-zero coeff was before the end of the block
        # or if the block was all zeros.
        if zero_run_length > 0 or not encoded_data : # If trailing zeros, or block was all zeros
             encoded_data.append((0, 0)) # Add EOB marker
        
        return encoded_data

    def _rle_decode(self, rle_data, num_coeffs_in_block):
        """Decodes RLE data back to zigzag coefficients."""
        zigzag_coeffs = []
        for run_length, value in rle_data:
            if run_length == 0 and value == 0:  # EOB marker
                break  # Stop decoding for this block
            zigzag_coeffs.extend([0] * run_length)
            zigzag_coeffs.append(value)
        
        # Fill remaining coefficients with zeros if EOB was encountered early
        # or if RLE ended before filling the block.
        while len(zigzag_coeffs) < num_coeffs_in_block:
            zigzag_coeffs.append(0)
            
        # Ensure the list is not longer than num_coeffs_in_block (shouldn't happen with break on EOB)
        return zigzag_coeffs[:num_coeffs_in_block]

    def compress(self, input_file_path, output_folder, quality=50):
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input image file not found: {input_file_path}")

        block_size = self.DEFAULT_BLOCK_SIZE 

        try:
            img = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {input_file_path}")
        except Exception as e:
            raise IOError(f"Error reading image file {input_file_path}: {e}")

        base_name = os.path.basename(input_file_path)
        file_name_no_ext, file_extension = os.path.splitext(base_name)
        output_file_path = os.path.join(output_folder, file_name_no_ext + ".dctz")
        os.makedirs(output_folder, exist_ok=True)

        img_height, img_width = img.shape
        quant_matrix_uint8 = self._get_quantization_matrix(quality, block_size)
        quant_matrix = quant_matrix_uint8.astype(np.float32) # For division

        all_packed_rle_for_zlib = bytearray()

        # Pad image to be multiple of block_size
        padded_height = ((img_height + block_size - 1) // block_size) * block_size
        padded_width = ((img_width + block_size - 1) // block_size) * block_size
        
        padded_img = np.zeros((padded_height, padded_width), dtype=np.float32)
        padded_img[0:img_height, 0:img_width] = img.astype(np.float32)
        padded_img -= 128.0 # Level shift (center around zero) ranges changes to -128 to 127

        for r in range(0, padded_height, block_size):
            for c in range(0, padded_width, block_size):
                block = padded_img[r:r+block_size, c:c+block_size] #creates block
                
                dct_block = cv2.dct(block)
                # Quantization: divide by quant_matrix and round.
                # Values can be negative.
                quantized_block = np.round(dct_block / quant_matrix).astype(np.int16) 
                
                zigzag_coeffs = self._zigzag_scan(quantized_block)
                rle_encoded_block = self._rle_encode(zigzag_coeffs)
                
                # Pack RLE data for this block
                # Each (run, value) pair: run (UBYTE), value (SHORT for int16)
                for run, val in rle_encoded_block:
                    # 'B' for unsigned char (run_length), 'h' for signed short (value)
                    all_packed_rle_for_zlib.extend(struct.pack('!Bh', run, val))

        compressed_payload = zlib.compress(bytes(all_packed_rle_for_zlib), level=9)

        try:
            with open(output_file_path, 'wb') as f_out:
                ext_bytes = file_extension.encode('utf-8')
                f_out.write(struct.pack('!H', len(ext_bytes)))
                f_out.write(ext_bytes)
                f_out.write(struct.pack('!I', img_height))
                f_out.write(struct.pack('!I', img_width))
                f_out.write(struct.pack('!B', block_size))
                f_out.write(struct.pack('!B', quality))
                f_out.write(compressed_payload)
        except IOError as e:
            raise IOError(f"Error writing DCT compressed file {output_file_path}: {e}")

        return output_file_path

    def decompress(self, input_file_path, output_folder):
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Compressed DCT file not found: {input_file_path}")

        os.makedirs(output_folder, exist_ok=True)
        output_file_path = "" # Initialize

        try:
            # reading the magic numbers
            with open(input_file_path, 'rb') as f_in:
                ext_len = struct.unpack('!H', f_in.read(2))[0]
                original_extension = f_in.read(ext_len).decode('utf-8')
                img_height = struct.unpack('!I', f_in.read(4))[0]
                img_width = struct.unpack('!I', f_in.read(4))[0]
                block_size = struct.unpack('!B', f_in.read(1))[0]
                quality = struct.unpack('!B', f_in.read(1))[0]
                
                compressed_payload = f_in.read()

            if block_size != self.DEFAULT_BLOCK_SIZE:
                 print(f"Warning: Decompressing with block size {block_size} which might differ from default {self.DEFAULT_BLOCK_SIZE}")

            packed_rle_bytes = zlib.decompress(compressed_payload)
            
            quant_matrix_uint8 = self._get_quantization_matrix(quality, block_size)
            quant_matrix = quant_matrix_uint8.astype(np.float32) # For multiplication
            
            num_coeffs_in_block = block_size * block_size
            padded_height = ((img_height + block_size - 1) // block_size) * block_size
            padded_width = ((img_width + block_size - 1) // block_size) * block_size
            
            reconstructed_img_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
            
            bytes_offset = 0 # Current position in packed_rle_bytes
            for r_idx in range(padded_height // block_size):
                for c_idx in range(padded_width // block_size):
                    current_block_rle_pairs = []
                    # Unpack RLE pairs for one block from the stream
                    while bytes_offset < len(packed_rle_bytes):
                        # Need 1 byte for run, 2 bytes for value ('h')
                        if bytes_offset + 3 > len(packed_rle_bytes): 
                            # This indicates a potentially truncated stream or an error.
                            print(f"Warning: Not enough bytes to read RLE pair at offset {bytes_offset} for block ({r_idx},{c_idx}). Stream might be corrupt.")
                            break 
                        
                        run = struct.unpack_from('!B', packed_rle_bytes, bytes_offset)[0]
                        bytes_offset += 1
                        val = struct.unpack_from('!h', packed_rle_bytes, bytes_offset)[0]
                        bytes_offset += 2
                        
                        current_block_rle_pairs.append((run, val))
                        if run == 0 and val == 0: # EOB for this block
                            break
                    
                    if not current_block_rle_pairs and num_coeffs_in_block > 0:
                        # This case can happen if a block was all zeros and only EOB was written,
                        # or if the stream ended prematurely.
                        # _rle_decode will handle filling with zeros if current_block_rle_pairs is empty
                        # or just contains EOB.
                        print(f"Info: Block ({r_idx},{c_idx}) RLE data seems empty or only EOB. Assuming zeros.")
                        # Pass an EOB if it's truly empty to ensure _rle_decode fills correctly
                        if not current_block_rle_pairs:
                            current_block_rle_pairs.append((0,0))


                    zigzag_coeffs = self._rle_decode(current_block_rle_pairs, num_coeffs_in_block)

                    if len(zigzag_coeffs) != num_coeffs_in_block:
                        # This is a safeguard; _rle_decode should handle padding.
                        print(f"Critical Error: Decoded RLE for block ({r_idx},{c_idx}) resulted in {len(zigzag_coeffs)} coeffs, expected {num_coeffs_in_block}. This indicates a flaw in RLE logic or corrupt data.")
                        # Attempt to recover by padding/truncating, but result will be flawed.
                        zigzag_coeffs = (zigzag_coeffs + [0]*num_coeffs_in_block)[:num_coeffs_in_block]

                    reconstructed_block_quantized = self._inverse_zigzag_scan(np.array(zigzag_coeffs, dtype=np.int16), (block_size, block_size))
                    
                    # Dequantize: multiply by quant_matrix
                    dequantized_block = reconstructed_block_quantized.astype(np.float32) * quant_matrix
                    
                    # Inverse DCT
                    reconstructed_block = cv2.idct(dequantized_block)
                    
                    r_start, c_start = r_idx * block_size, c_idx * block_size
                    reconstructed_img_padded[r_start:r_start+block_size, c_start:c_start+block_size] = reconstructed_block

            reconstructed_img_padded += 128.0 # Level shift back
            reconstructed_img_padded = np.clip(reconstructed_img_padded, 0, 255) # Clip to valid pixel range [0, 255]
            
            final_image = reconstructed_img_padded[0:img_height, 0:img_width].astype(np.uint8)

            base_name = os.path.basename(input_file_path)
            file_name_no_ext, _ = os.path.splitext(base_name)
            
            valid_img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
            output_extension = original_extension if original_extension.lower() in valid_img_extensions else ".png"
            
            output_file_path = os.path.join(output_folder, file_name_no_ext + "_decompressed" + output_extension)
            if not cv2.imwrite(output_file_path, final_image):
                raise RuntimeError(f"Failed to write decompressed image to {output_file_path}")

        except (IOError, struct.error, zlib.error, ValueError, RuntimeError) as e:
            # Clean up potentially partially written file if error occurs
            if output_file_path and os.path.exists(output_file_path) and os.path.getsize(output_file_path) == 0:
                try: os.remove(output_file_path)
                except OSError: pass # Ignore if file couldn't be removed
            raise RuntimeError(f"Error during DCT decompression of {input_file_path}: {e}")

        return output_file_path

# Example Usage (for testing directly)
if __name__ == '__main__':
    # Create dummy folders for testing
    test_img_folder = "test_images_dct"
    compressed_folder = "compressed_output_dct"
    decompressed_folder = "decompressed_output_dct"
    os.makedirs(test_img_folder, exist_ok=True)
    os.makedirs(compressed_folder, exist_ok=True)
    os.makedirs(decompressed_folder, exist_ok=True)

    # Create a dummy grayscale image (e.g., 64x64)
    dummy_image_path = os.path.join(test_img_folder, "dummy_gray_dct.png")
    if not os.path.exists(dummy_image_path):
        # Create a simple gradient image for better visual assessment
        gradient_img = np.zeros((64, 64), dtype=np.uint8)
        for i in range(64):
            gradient_img[i, :] = int(i / 63.0 * 255)
        cv2.imwrite(dummy_image_path, gradient_img)
        print(f"Created dummy image: {dummy_image_path}")


    dct_compressor = DCTCompression()
    test_quality = 50
    
    try:
        print(f"Compressing {dummy_image_path} with quality {test_quality}...")
        compressed_file = dct_compressor.compress(dummy_image_path, compressed_folder, quality=test_quality)
        print(f"Compressed to: {compressed_file}")
        print(f"Compressed file size: {os.path.getsize(compressed_file) / 1024:.2f} KB")


        print(f"Decompressing {compressed_file}...")
        decompressed_file = dct_compressor.decompress(compressed_file, decompressed_folder)
        print(f"Decompressed to: {decompressed_file}")

        # Basic check: compare original and decompressed
        original_img = cv2.imread(dummy_image_path, cv2.IMREAD_GRAYSCALE)
        final_reconstructed_img = cv2.imread(decompressed_file, cv2.IMREAD_GRAYSCALE)
        
        if original_img is not None and final_reconstructed_img is not None:
            if original_img.shape != final_reconstructed_img.shape:
                print(f"Error: Shape mismatch! Original: {original_img.shape}, Decompressed: {final_reconstructed_img.shape}")
            else:
                mse = np.mean((original_img.astype("float") - final_reconstructed_img.astype("float")) ** 2)
                print(f"Mean Squared Error between original and decompressed: {mse:.2f}")
                # A lower MSE generally means better quality for lossy compression.
                # The acceptable threshold depends on the image and quality setting.
                if mse < 500: # Adjusted threshold, highly dependent on image and quality
                     print("DCT Compression and Decompression test seems OK (check images visually).")
                else:
                     print("DCT Test: MSE is somewhat high. Review results or quality setting. This is expected for lossy compression.")
        else:
            print("Error: Could not read back original or decompressed image for comparison.")

    except Exception as e:
        print(f"An error occurred during DCT testing: {e}")
        import traceback
        traceback.print_exc()
