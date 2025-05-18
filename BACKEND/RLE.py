import os
import struct

class RLECompression:
    """
    Implementation of Run-Length Encoding (RLE) for text and binary data.

    Compressed file format:
    1. Original Extension Length (USHORT - 2 bytes)
    2. Original Extension (UTF-8 string)
    3. Series of runs:
       - Run Type (BYTE - 1 byte):
         - 0x00: Non-repeating sequence
         - 0x01: Repeating sequence
       - Length (BYTE - 1 byte): Length of the sequence (1-255).
       - Data:
         - If non-repeating: 'Length' bytes of literal data.
         - If repeating: 1 byte of the repeated character.
    This allows for encoding both runs of identical bytes and sequences of non-repeating bytes.
    A count byte can represent up to 255. If a run is longer, it will be split.
    """
    MAX_RUN_LENGTH = 255

    def __init__(self):
        pass

    def compress(self, input_file_path, output_folder):
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        try:
            with open(input_file_path, 'rb') as f_in:
                data = f_in.read()
        except IOError as e:
            raise IOError(f"Error reading input file {input_file_path}: {e}")

        base_name = os.path.basename(input_file_path)
        file_name_no_ext, file_extension = os.path.splitext(base_name)
        output_file_path = os.path.join(output_folder, file_name_no_ext + ".rle")
        
        os.makedirs(output_folder, exist_ok=True)

        if not data: # Handle empty file
            with open(output_file_path, 'wb') as f_out:
                ext_bytes = file_extension.encode('utf-8')
                f_out.write(struct.pack('!H', len(ext_bytes)))
                f_out.write(ext_bytes)
            return output_file_path

        compressed_data = bytearray()
        i = 0
        n = len(data)
        while i < n:
            # Check for repeating sequence
            current_byte = data[i]
            count = 1
            j = i + 1
            while j < n and data[j] == current_byte and count < self.MAX_RUN_LENGTH:
                count += 1
                j += 1
            
            if count > 1: # Found a run of repeating characters (min length 2 for it to be a "run")
                compressed_data.append(0x01) # Repeating sequence marker
                compressed_data.append(count)    # Length of run
                compressed_data.append(current_byte) # The repeated byte
                i += count
            else:
                # Start of a non-repeating sequence or single byte
                # Find how long the non-repeating sequence is
                k = i
                literal_run = bytearray()
                while k < n and len(literal_run) < self.MAX_RUN_LENGTH:
                    # Look ahead one byte to see if a run starts
                    if k + 1 < n and data[k+1] == data[k]: # A run is about to start
                        # If a run of at least 2 starts at data[k], we stop the literal run here.
                        # Or if a run of 3 starts at data[k+1]
                        if k + 2 < n and data[k+1] == data[k+2]: # A run of at least 2 starts at k+1
                             break # Stop literal run before this new repeating run
                    literal_run.append(data[k])
                    # Check if current byte is part of a run of 2
                    if k + 1 < n and data[k] == data[k+1]: 
                        # If this literal run ends and the next two bytes are same,
                        # we might be better off ending literal run here.
                        # This logic is tricky to optimize perfectly.
                        # For now, just check if the *next* byte starts a run.
                        if k + 2 < n and data[k+1] == data[k+2]: # if data[k+1] starts a run of 2 or more
                            # e.g. A B C C C. If current is A, next is B.
                            # If current is B, next is C C C. So B should be literal.
                            pass # continue adding to literal_run
                        else: # if data[k+1] does not start a run of 2 or more, but data[k] == data[k+1]
                              # This means data[k], data[k+1] is a run of 2.
                              # If literal_run already has content, break.
                              # If literal_run is empty, this means data[i], data[i+1] is a run of 2.
                              # This case should have been caught by the `count > 1` block above.
                              # This part of logic handles sequences like A B C D D E F
                              # It will try to group A B C D as literal.
                              pass


                    k += 1
                
                compressed_data.append(0x00) # Non-repeating sequence marker
                compressed_data.append(len(literal_run))
                compressed_data.extend(literal_run)
                i += len(literal_run)


        try:
            with open(output_file_path, 'wb') as f_out:
                ext_bytes = file_extension.encode('utf-8')
                f_out.write(struct.pack('!H', len(ext_bytes)))
                f_out.write(ext_bytes)
                f_out.write(compressed_data)
        except IOError as e:
            raise IOError(f"Error writing compressed RLE file {output_file_path}: {e}")
        
        return output_file_path

    def decompress(self, input_file_path, output_folder):
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Compressed RLE file not found: {input_file_path}")

        os.makedirs(output_folder, exist_ok=True)
        decompressed_data = bytearray()
        output_file_path = ""

        try:
            with open(input_file_path, 'rb') as f_in:
                ext_len_bytes = f_in.read(2)
                if not ext_len_bytes: raise ValueError("Invalid RLE file: missing ext length.")
                ext_len = struct.unpack('!H', ext_len_bytes)[0]
                
                ext_bytes = f_in.read(ext_len)
                if len(ext_bytes) != ext_len: raise ValueError("Invalid RLE file: incomplete ext string.")
                original_extension = ext_bytes.decode('utf-8')

                base_name = os.path.basename(input_file_path)
                file_name_no_ext, _ = os.path.splitext(base_name)
                output_file_path = os.path.join(output_folder, file_name_no_ext + "_decompressed" + original_extension)

                while True:
                    run_type_byte = f_in.read(1)
                    if not run_type_byte: break # End of data

                    run_type = run_type_byte[0]
                    
                    length_byte = f_in.read(1)
                    if not length_byte: raise ValueError("Invalid RLE file: truncated run length.")
                    length = length_byte[0]

                    if length == 0 : # Should not happen with this encoding if MAX_RUN_LENGTH > 0
                        raise ValueError("Invalid RLE file: zero length run.")

                    if run_type == 0x01: # Repeating sequence
                        char_byte = f_in.read(1)
                        if not char_byte: raise ValueError("Invalid RLE file: truncated repeating char.")
                        decompressed_data.extend([char_byte[0]] * length)
                    elif run_type == 0x00: # Non-repeating sequence
                        literal_bytes = f_in.read(length)
                        if len(literal_bytes) != length:
                            raise ValueError("Invalid RLE file: truncated literal sequence.")
                        decompressed_data.extend(literal_bytes)
                    else:
                        raise ValueError(f"Invalid RLE run type marker: {run_type}")
            
            with open(output_file_path, 'wb') as f_out:
                f_out.write(decompressed_data)
        
        except (IOError, struct.error, ValueError) as e:
            if output_file_path and os.path.exists(output_file_path):
                pass # os.remove(output_file_path) # Optional
            raise RuntimeError(f"Error during RLE decompression of {input_file_path}: {e}")
            
        return output_file_path
