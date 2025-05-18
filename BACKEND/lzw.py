import os
import struct

class LZWCompression:
    """
    Implementation of the LZW (Lempel-Ziv-Welch) compression and decompression algorithm.

    The compressed file format:
    1. Original Extension Length (USHORT - 2 bytes)
    2. Original Extension (UTF-8 string)
    3. Series of Codes (USHORT - 2 bytes each, assuming max_dict_size <= 65535)
       - The dictionary size can grow. We need to handle the bit width of codes
         or use a fixed-width code that's large enough.
       - A common approach is to start with 9-bit codes and increase to 10, 11, 12 bits etc.,
         up to a max (e.g., 12 or 16 bits). A special code can signal dictionary reset.
       - For simplicity and compatibility with the previous structure that used `CODE_BYTES = 3`,
         let's use a fixed number of bytes per code. If `CODE_BYTES = 2` (USHORT), max dict size is 65535.
         If `CODE_BYTES = 3`, max dict size is 2^24, which is very large.
         The previous `lzw.py` used `CODE_BYTES = 3`. Let's stick to that for now,
         but note that 2 bytes (max 65535 entries) is often sufficient for many files.
         Using 3 bytes per code is simple but might not be the most space-efficient if codes are small.
    
    Let's use 2 bytes (unsigned short) for codes, allowing for a dictionary size up to 65,535.
    This is a common choice for LZW implementations (e.g., GIF).
    We'll need a way to handle dictionary full situations if we fix max size.
    Standard LZW initializes dict with 256 single-byte sequences.
    Max code value will be `max_dict_size - 1`.
    """

    MAX_DICT_SIZE_BITS = 12 # Max bits for a code, e.g., 12 bits for 4096 entries. GIF uses up to 12.
                            # If we use 16 bits (2 bytes), then 65536 entries.
                            # Let's use a fixed 2 bytes per code for simplicity in this version.
    CODE_SIZE_BYTES = 2 # Each code will be written as 2 bytes (unsigned short)
    INITIAL_DICT_SIZE = 256

    def __init__(self):
        pass # No specific state needed in init for this stateless approach

    def compress(self, input_file_path, output_folder):
        """
        Compresses a file using the LZW algorithm.
        :param input_file_path: Path to the input file.
        :param output_folder: Folder to save the compressed file.
        :return: Path to the compressed file.
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        try:
            with open(input_file_path, 'rb') as f_in:
                data = f_in.read()
        except IOError as e:
            raise IOError(f"Error reading input file {input_file_path}: {e}")

        base_name = os.path.basename(input_file_path)
        file_name_no_ext, file_extension = os.path.splitext(base_name)
        output_file_path = os.path.join(output_folder, file_name_no_ext + ".lzw")
        
        os.makedirs(output_folder, exist_ok=True)

        if not data: # Handle empty file
            with open(output_file_path, 'wb') as f_out:
                ext_bytes = file_extension.encode('utf-8')
                f_out.write(struct.pack('!H', len(ext_bytes)))
                f_out.write(ext_bytes)
            return output_file_path

        # Initialize dictionary with all single-byte sequences
        dictionary_size = self.INITIAL_DICT_SIZE
        dictionary = {bytes([i]): i for i in range(dictionary_size)}
        
        # Max possible code value with CODE_SIZE_BYTES
        max_code_value = (1 << (self.CODE_SIZE_BYTES * 8)) -1

        current_sequence = b""
        compressed_codes = []

        for byte_val in data:
            byte_as_bytes = bytes([byte_val])
            new_sequence = current_sequence + byte_as_bytes
            if new_sequence in dictionary:
                current_sequence = new_sequence
            else:
                compressed_codes.append(dictionary[current_sequence])
                # Add new_sequence to dictionary if space allows
                if dictionary_size <= max_code_value: # Check against max value a code can represent
                    dictionary[new_sequence] = dictionary_size
                    dictionary_size += 1
                current_sequence = byte_as_bytes
        
        # Add the last sequence's code
        if current_sequence:
            compressed_codes.append(dictionary[current_sequence])

        try:
            with open(output_file_path, 'wb') as f_out:
                ext_bytes = file_extension.encode('utf-8')
                f_out.write(struct.pack('!H', len(ext_bytes)))
                f_out.write(ext_bytes)

                for code in compressed_codes:
                    f_out.write(struct.pack('!H', code)) # Write each code as 2 bytes (USHORT)
        except IOError as e:
            raise IOError(f"Error writing compressed file {output_file_path}: {e}")
        
        return output_file_path

    def decompress(self, input_file_path, output_folder):
        """
        Decompresses a file compressed by this LZW implementation.
        :param input_file_path: Path to the compressed file.
        :param output_folder: Folder to save the decompressed file.
        :return: Path to the decompressed file.
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Compressed file not found: {input_file_path}")

        os.makedirs(output_folder, exist_ok=True)
        decompressed_data = bytearray()
        
        output_file_path = "" # Initialize to prevent reference before assignment in finally

        try:
            with open(input_file_path, 'rb') as f_in:
                ext_len_bytes = f_in.read(2)
                if not ext_len_bytes: raise ValueError("Invalid LZW file: missing ext length.")
                ext_len = struct.unpack('!H', ext_len_bytes)[0]
                
                ext_bytes = f_in.read(ext_len)
                if len(ext_bytes) != ext_len: raise ValueError("Invalid LZW file: incomplete ext string.")
                original_extension = ext_bytes.decode('utf-8')

                base_name = os.path.basename(input_file_path)
                file_name_no_ext, _ = os.path.splitext(base_name)
                output_file_path = os.path.join(output_folder, file_name_no_ext + "_decompressed" + original_extension)

                # Initialize dictionary
                dictionary_size = self.INITIAL_DICT_SIZE
                # Store index -> bytes for decompression
                dictionary = {i: bytes([i]) for i in range(dictionary_size)}
                max_code_value = (1 << (self.CODE_SIZE_BYTES * 8)) - 1


                # Read the first code
                first_code_bytes = f_in.read(self.CODE_SIZE_BYTES)
                if not first_code_bytes: # Empty compressed data after header
                    if ext_len == os.path.getsize(input_file_path) - 2: # Check if only header was present
                         with open(output_file_path, 'wb') as f_out: # Create empty file
                            pass
                         return output_file_path
                    raise ValueError("Invalid LZW file: no data after header.")

                if len(first_code_bytes) != self.CODE_SIZE_BYTES:
                    raise ValueError("Invalid LZW file: truncated first code.")
                
                previous_code = struct.unpack('!H', first_code_bytes)[0]
                if previous_code >= self.INITIAL_DICT_SIZE : # First code must be in initial dict
                    raise ValueError(f"Invalid first code {previous_code} in LZW stream.")

                current_sequence = dictionary[previous_code]
                decompressed_data.extend(current_sequence)

                while True:
                    code_bytes = f_in.read(self.CODE_SIZE_BYTES)
                    if not code_bytes:
                        break # End of stream
                    if len(code_bytes) != self.CODE_SIZE_BYTES:
                        raise ValueError("Invalid LZW file: truncated code.")
                    
                    current_code = struct.unpack('!H', code_bytes)[0]

                    entry = b""
                    if current_code in dictionary:
                        entry = dictionary[current_code]
                    elif current_code == dictionary_size: # Special case: sequence is previous_sequence + first_char_of_previous_sequence
                        entry = dictionary[previous_code] + dictionary[previous_code][:1]
                    else:
                        raise ValueError(f"Bad compressed code: {current_code} (dict_size: {dictionary_size})")
                    
                    decompressed_data.extend(entry)

                    # Add new sequence to dictionary
                    if dictionary_size <= max_code_value:
                         # The new entry is previous_sequence + first_char_of_current_entry
                        new_dict_entry = dictionary[previous_code] + entry[:1]
                        dictionary[dictionary_size] = new_dict_entry
                        dictionary_size += 1
                    
                    previous_code = current_code
            
            with open(output_file_path, 'wb') as f_out:
                f_out.write(decompressed_data)
        
        except (IOError, struct.error, ValueError) as e:
            if output_file_path and os.path.exists(output_file_path):
                # os.remove(output_file_path) # Optional
                pass
            raise RuntimeError(f"Error during LZW decompression of {input_file_path}: {e}")
            
        return output_file_path

