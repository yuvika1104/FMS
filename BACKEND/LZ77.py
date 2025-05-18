import os
import struct

class LZ77:
    """
    Implementation of the LZ77 compression and decompression algorithm.

    The compressed file format will be:
    1. Original Extension Length (USHORT - 2 bytes)
    2. Original Extension (UTF-8 string)
    3. Series of tokens:
        - Offset (USHORT - 2 bytes)
        - Length (USHORT - 2 bytes)
        - Next Byte (BYTE - 1 byte)
          If length is 0, this means (0, 0, literal_byte).
          If length > 0, this means (offset, length, next_byte_after_match).
          If at the end of the file and there's a match without a next byte,
          a special token might be needed or the logic adjusted.
          For simplicity, we'll ensure the last token is either a literal
          or a match that is fully contained within the input data,
          meaning there's always a "next byte" concept, even if it's
          just the last byte of the match itself if the match ends the file.
          A simpler way: if a match takes us to the end of the file,
          we can store a zero-length "next byte" or handle it by not reading one.
          Let's stick to (offset, length, next_byte) where next_byte is always present
          for a literal, and for a match, it's the byte *after* the match.
          If a match extends to the very end of the file, there is no next byte.
          We can use a flag or a special value.

    Revised Token Strategy:
    - Literal: (0, 0, literal_byte) - 5 bytes
    - Match: (offset, length) - 4 bytes. The decompressor implicitly knows the next byte is from the input stream.

    Let's stick to the common (offset, length, char) for matches, and (0,0,char) for literals.
    If a match takes us to the end, the 'char' part of the token is not strictly needed
    but for fixed token structure, we might write a dummy or the last char of the match.
    The provided original code's `compress` method for LZ77:
    if match_len > 0:
        nxt_idx = i + match_len
        nxt = data[nxt_idx] if nxt_idx < len(data) else None
        o.write(match_off.to_bytes(2,'big'))
        o.write(match_len.to_bytes(2,'big'))
        if nxt is not None:
            o.write(bytes([nxt]))
        i += match_len + (1 if nxt is not None else 0)
    else: # literal
        o.write((0).to_bytes(2,'big')) # offset = 0
        o.write((0).to_bytes(2,'big')) # length = 0
        o.write(bytes([data[i]]))
        i += 1
    This means a match token is (offset, length, next_actual_byte_from_input)
    and a literal token is (0, 0, literal_byte).
    The match token consumes `match_len + 1` bytes from input.
    If `nxt is None` (match goes to end of file), then the token is just (offset, length)
    and consumes `match_len` bytes. This makes token parsing tricky.

    Let's use a simpler, more standard token:
    - (0, length, literal_bytes_array) -> if length > 0, it's a sequence of literals.
    - (offset, length) -> if offset > 0, it's a match.

    Or, even simpler and common:
    Each token is (offset, length, next_symbol).
    - If offset=0, length=0, then next_symbol is a literal.
    - If offset>0, length>0, then (offset,length) is a match, and next_symbol is the symbol *following* the match.
    This is what the original code seems to be doing, but with a conditional write for next_symbol.

    Let's refine the token structure for LZ77 to ensure robust decompression:
    Each token will be one of two types:
    1. Literal Token:
       - Type_Byte (e.g., 0x00)
       - Literal_Byte (the actual byte)
    2. Match_Token:
       - Type_Byte (e.g., 0x01)
       - Offset (USHORT, 2 bytes, distance backwards)
       - Length (USHORT, 2 bytes, length of match)

    This is clear but might be less efficient.
    The original approach of (offset, length, next_char_or_literal) is common.
    Let's try to make that robust.
    A token will be (USHORT offset, USHORT length).
    If offset=0 and length=0, it's followed by 1 literal byte.
    If offset>0 and length>0, it's a match. The next byte in the input stream is implicitly consumed.

    Let's re-evaluate the original LZ77.py's structure:
    `o.write(match_off.to_bytes(2,'big'))`
    `o.write(match_len.to_bytes(2,'big'))`
    `if nxt is not None: o.write(bytes([nxt]))` (for match)
    OR
    `o.write((0).to_bytes(2,'big'))`
    `o.write((0).to_bytes(2,'big'))`
    `o.write(bytes([data[i]]))` (for literal)

    This means a literal token is 5 bytes: (0, 0, byte).
    A match token is 4 or 5 bytes: (off, len) if it's the end, or (off, len, next_byte) if not.
    This variable length token is problematic for the decompressor.

    A better standard:
    Compressed data: (Original Extension Info) [ (Offset, Length, Literal/NextChar) ... ]
    - Offset (2 bytes): If 0, 'Length' is 0, and 'Literal/NextChar' is the literal.
    - Length (2 bytes): If 0 (and Offset is 0), it's a literal. If >0, it's match length.
    - Literal/NextChar (1 byte): The literal character, or the character immediately following the match.

    This means every token is 5 bytes.
    (0, 0, literal_char)
    (offset, length, char_after_match)

    What if a match goes to the end of the file? There is no char_after_match.
    In this case, we can write a dummy byte for char_after_match, or the decompressor
    needs to know when to stop based on total decompressed size (if stored, but it's not).

    Let's use the structure from a widely cited LZ77 example:
    Output consists of triples (offset, length, next_symbol).
    - `offset`: distance to start of match in window. 0 if no match.
    - `length`: length of match. 0 if no match.
    - `next_symbol`: symbol following match, or the literal if no match.

    This is 5 bytes per token.
    (0, 0, literal_byte)
    (match_offset, match_length, first_byte_after_match_from_input)
    The decompressor copies `match_length` bytes, then appends `first_byte_after_match_from_input`.
    Total bytes processed by this token: `match_length + 1`.
    """

    # Default values for window size and lookahead buffer size
    DEFAULT_WINDOW_SIZE = 4096  # Max size of the sliding window (search buffer)
    DEFAULT_LOOKAHEAD_BUFFER_SIZE = 256  # Max size of the lookahead buffer

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE, lookahead_buffer_size=DEFAULT_LOOKAHEAD_BUFFER_SIZE):
        """
        Initializes the LZ77 compressor/decompressor.
        :param window_size: The size of the sliding window.
        :param lookahead_buffer_size: The size of the lookahead buffer.
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive.")
        if lookahead_buffer_size <= 0:
            raise ValueError("Lookahead buffer size must be positive.")
        
        self.window_size = window_size
        # Max match length is constrained by lookahead buffer size
        self.max_match_length = lookahead_buffer_size 
        # Max offset is constrained by window size
        self.max_offset = window_size


    def find_longest_match(self, data, current_pos):
        """
        Finds the longest match for the lookahead buffer in the sliding window.
        :param data: The input data (bytes).
        :param current_pos: The current position in the data, marking the start of the lookahead buffer.
        :return: A tuple (match_offset, match_length).
                 match_offset is the distance backwards from current_pos.
                 match_length is the length of the match.
                 Returns (0, 0) if no match is found.
        """
        best_match_length = 0
        best_match_offset = 0

        # Define the end of the lookahead buffer (cannot exceed data length)
        lookahead_end = min(current_pos + self.max_match_length, len(data))
        
        # Define the start of the search window (cannot be less than 0)
        search_window_start = max(0, current_pos - self.window_size)

        # Iterate through all possible starting positions in the search window
        for i in range(search_window_start, current_pos):
            match_length = 0
            # Compare bytes from search window (data[i:]) with lookahead buffer (data[current_pos:])
            while (match_length < self.max_match_length and # Ensure match length doesn't exceed buffer
                   i + match_length < current_pos and      # Ensure search window part is before lookahead
                   current_pos + match_length < len(data) and # Ensure lookahead part is within data
                   data[i + match_length] == data[current_pos + match_length]):
                match_length += 1
            
            if match_length > best_match_length:
                best_match_length = match_length
                best_match_offset = current_pos - i # Offset is distance from current_pos back to start of match

        return best_match_offset, best_match_length

    def compress(self, input_file_path, output_folder):
        """
        Compresses a file using the LZ77 algorithm.
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

        if not data: # Handle empty file
            base_name = os.path.basename(input_file_path)
            file_name_no_ext, file_extension = os.path.splitext(base_name)
            output_file_path = os.path.join(output_folder, file_name_no_ext + ".lz77")
            os.makedirs(output_folder, exist_ok=True)
            with open(output_file_path, 'wb') as f_out:
                ext_bytes = file_extension.encode('utf-8')
                f_out.write(struct.pack('!H', len(ext_bytes))) # Length of extension
                f_out.write(ext_bytes) # Extension itself
            return output_file_path


        base_name = os.path.basename(input_file_path)
        file_name_no_ext, file_extension = os.path.splitext(base_name)
        output_file_path = os.path.join(output_folder, file_name_no_ext + ".lz77")
        
        os.makedirs(output_folder, exist_ok=True)

        compressed_tokens = []
        current_pos = 0
        while current_pos < len(data):
            match_offset, match_length = self.find_longest_match(data, current_pos)

            # A match must be at least, say, 3 bytes to be worthwhile, 
            # as a token (O,L,C) is 5 bytes. (O,L) for match is 4 bytes.
            # If we use (O,L,C) for everything:
            # Literal: (0,0,C) = 5 bytes for 1 byte of data.
            # Match: (O,L,C_after_match) = 5 bytes for L+1 bytes of data.
            # To be beneficial, L+1 should be > 5, so L > 4.
            # Or, if we consider the alternative of L literals, that's 5*L bytes.
            # So, a match (O,L,C_after_match) is better if 5 < 5*L (for L=1) or 5 < 5*(L+1) (for L data + C_after_match).
            # Let's use a minimum match length (e.g., 3) for it to be encoded as a match.
            min_encodeable_match_length = 3 # Smallest match length to encode as (offset, length)

            if match_length >= min_encodeable_match_length:
                # Encode as a match
                # Token: (offset, length, next_char_after_match)
                next_char_pos = current_pos + match_length
                if next_char_pos < len(data):
                    next_char = data[next_char_pos]
                    compressed_tokens.append((match_offset, match_length, next_char))
                    current_pos += match_length + 1
                else: # Match extends to the end of the file
                    # We need a way to signify this. Let's use a special token or adjust.
                    # For now, let's make sure our token structure is (O,L) for match, and (0,Literal) for literal.
                    # This requires a flag bit or different token types.

                    # Sticking to (Offset, Length, NextSymbol)
                    # If match goes to end, what is NextSymbol?
                    # Option: Last token is just (Offset, Length), no NextSymbol. Decompressor needs to know.
                    # Option: (Offset, Length, DUMMY_VALUE) if match is at end.
                    # Option: Encode the last match as literals if it's too complex.

                    # Let's try: if a match takes us to the end, we simply don't have a "next_char".
                    # The token will be (match_offset, match_length), and we advance by match_length.
                    # This means variable token structure.
                    # To keep fixed token structure (O,L,C), if match is at end, C can be a dummy
                    # or the last char of the match itself (less useful).

                    # Simpler: if match_length > 0, always emit (offset, length).
                    # Then, if current_pos + match_length < len(data), emit a literal for that next char.
                    # This is not standard LZ77.

                    # Standard LZ77 emits (offset, length, char_following_match) or (0, 0, literal_char)
                    # Let's assume the problem of "char_following_match" for end-of-file matches
                    # will be handled by the decompressor knowing the original file size, or by
                    # ensuring the last N bytes are literals if they can't form a full (O,L,C) token.

                    # For now, if match goes to end, we'll output (offset, length) and the decompressor
                    # will handle it. This means the last token can be shorter.
                    # To simplify, we'll always write 5 bytes. If match is at end, next_char can be 0.
                    # This needs careful handling in decompressor.

                    # Let's use the (0,0,Literal) and (Offset,Length,NextActualChar) model.
                    # If match goes to end, then there is no NextActualChar.
                    # The original code did: `if nxt is not None: o.write(bytes([nxt]))`
                    # This implies the decompressor must handle variable length tokens or know when to expect the 3rd part.

                    # Let's use a fixed 5-byte token: (Offset, Length, Symbol)
                    # If Offset=0, Length=0, Symbol is literal. current_pos += 1.
                    # If Offset>0, Length>0, Symbol is the character *after* the match. current_pos += Length + 1.
                    # If a match (Offset, Length) occurs, and current_pos + Length == len(data) (i.e., match is at the end)
                    # then we cannot provide a "Symbol after match".
                    # In this specific case, we can encode it as (Offset, Length, DUMMY_SYMBOL)
                    # and the decompressor, upon seeing this, copies Length and stops.
                    # Or, we can ensure the last few bytes are always literals.

                    # Reverting to a simpler token for this implementation:
                    # Token type 1: Literal (1 byte prefix '0', 1 byte literal)
                    # Token type 2: Match (1 byte prefix '1', 2 bytes offset, 2 bytes length)
                    # This is clearer.
                    compressed_tokens.append({'type': 'match', 'offset': match_offset, 'length': match_length})
                    current_pos += match_length
            else:
                # Encode as a literal
                compressed_tokens.append({'type': 'literal', 'char': data[current_pos]})
                current_pos += 1
        
        try:
            with open(output_file_path, 'wb') as f_out:
                # Write original file extension
                ext_bytes = file_extension.encode('utf-8')
                f_out.write(struct.pack('!H', len(ext_bytes))) # Length of extension
                f_out.write(ext_bytes) # Extension itself

                # Write tokens
                for token in compressed_tokens:
                    if token['type'] == 'literal':
                        f_out.write(b'\x00') # Literal marker
                        f_out.write(bytes([token['char']]))
                    elif token['type'] == 'match':
                        f_out.write(b'\x01') # Match marker
                        # Ensure offset and length fit in 2 bytes.
                        # Max offset is self.window_size, max length is self.max_match_length.
                        # These should be < 65535.
                        f_out.write(struct.pack("!H", token['offset'])) # USHORT for offset
                        f_out.write(struct.pack("!H", token['length'])) # USHORT for length
        except IOError as e:
            raise IOError(f"Error writing compressed file {output_file_path}: {e}")
        
        return output_file_path

    def decompress(self, input_file_path, output_folder):
        """
        Decompresses a file compressed by this LZ77 implementation.
        :param input_file_path: Path to the compressed file.
        :param output_folder: Folder to save the decompressed file.
        :return: Path to the decompressed file.
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Compressed file not found: {input_file_path}")

        os.makedirs(output_folder, exist_ok=True)
        decompressed_data = bytearray()

        try:
            with open(input_file_path, 'rb') as f_in:
                # Read original file extension
                ext_len_bytes = f_in.read(2)
                if not ext_len_bytes: raise ValueError("Invalid compressed file: missing ext length.")
                ext_len = struct.unpack('!H', ext_len_bytes)[0]
                
                ext_bytes = f_in.read(ext_len)
                if len(ext_bytes) != ext_len: raise ValueError("Invalid compressed file: incomplete ext string.")
                original_extension = ext_bytes.decode('utf-8')

                base_name = os.path.basename(input_file_path)
                file_name_no_ext, _ = os.path.splitext(base_name) # Get name part from compressed file
                # Construct output path
                output_file_path = os.path.join(output_folder, file_name_no_ext + "_decompressed" + original_extension)

                # Read tokens
                while True:
                    marker_byte = f_in.read(1)
                    if not marker_byte:
                        break # End of file

                    if marker_byte == b'\x00': # Literal
                        literal_char_byte = f_in.read(1)
                        if not literal_char_byte: raise ValueError("Invalid compressed file: truncated literal token.")
                        decompressed_data.append(literal_char_byte[0])
                    elif marker_byte == b'\x01': # Match
                        offset_bytes = f_in.read(2)
                        length_bytes = f_in.read(2)
                        if not offset_bytes or not length_bytes or len(offset_bytes) != 2 or len(length_bytes) != 2:
                            raise ValueError("Invalid compressed file: truncated match token.")
                        
                        offset = struct.unpack("!H", offset_bytes)[0]
                        length = struct.unpack("!H", length_bytes)[0]

                        if offset == 0 or length == 0: # Should not happen for match token with this marker
                             raise ValueError(f"Invalid match token: offset={offset}, length={length}")
                        if offset > len(decompressed_data):
                            raise ValueError(f"Invalid offset {offset} > decompressed_data_len {len(decompressed_data)}")

                        # Copy bytes for match
                        start_copy_pos = len(decompressed_data) - offset
                        for _ in range(length):
                            decompressed_data.append(decompressed_data[start_copy_pos])
                            start_copy_pos += 1
                    else:
                        raise ValueError(f"Invalid token marker: {marker_byte}")
            
            with open(output_file_path, 'wb') as f_out:
                f_out.write(decompressed_data)
        
        except (IOError, struct.error, ValueError) as e:
            # Clean up potentially partially written file if error occurs
            if os.path.exists(output_file_path):
                 # os.remove(output_file_path) # Optional: remove partial file on error
                 pass
            raise RuntimeError(f"Error during decompression of {input_file_path}: {e}")
            
        return output_file_path

