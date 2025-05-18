import os
import heapq
import json # For serializing the frequency table
import struct

class HuffmanCoding:
    """
    Implementation of Huffman Coding for compression and decompression.

    The compressed file format:
    1. Original Extension Length (USHORT - 2 bytes)
    2. Original Extension (UTF-8 string)
    3. Padding Bits Count (UBYTE - 1 byte): Number of padding bits at the end of compressed data (0-7).
    4. Frequency Table Length (UINT - 4 bytes): Length of the JSON serialized frequency table.
    5. Frequency Table (JSON string): Maps byte values (as strings) to their frequencies.
       e.g., {"0": 10, "97": 5, ...}
    6. Compressed Data (bytes)
    """

    class HeapNode:
        def __init__(self, char_code, freq):
            self.char_code = char_code  # Integer byte value or None for internal nodes
            self.freq = freq
            self.left = None
            self.right = None

        # Comparator for the min-heap
        def __lt__(self, other):
            if other is None:
                return -1
            if not isinstance(other, HuffmanCoding.HeapNode):
                return -1
            return self.freq < other.freq

    def __init__(self):
        self.codes = {}
        self.reverse_mapping = {} # For decompression: code_string -> char_code

    def _make_frequency_dict(self, data):
        """Calculates frequency of each byte in the data."""
        frequency = {}
        for byte_val in data:
            frequency[byte_val] = frequency.get(byte_val, 0) + 1
        return frequency

    def _build_heap(self, frequency_dict):
        """Builds a min-heap from the frequency dictionary."""
        heap = []
        for char_code, freq in frequency_dict.items():
            node = self.HeapNode(char_code, freq)
            heapq.heappush(heap, node)
        return heap

    def _merge_nodes(self, heap):
        """Merges nodes in the heap to build the Huffman tree."""
        if not heap: return None
        if len(heap) == 1: # Handle case of single unique character
            node = heapq.heappop(heap)
            # Create a dummy parent if only one node exists to form a tree structure
            merged = self.HeapNode(None, node.freq)
            merged.left = node 
            # merged.right could be None or another dummy node if strict binary tree needed for traversal
            # For code generation, having one branch is fine.
            heapq.heappush(heap, merged)


        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)

            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(heap, merged)
        return heap[0] if heap else None # The root of the Huffman tree

    def _make_codes_helper(self, root_node, current_code):
        """Recursively builds Huffman codes from the tree."""
        if root_node is None:
            return

        if root_node.char_code is not None: # Leaf node
            self.codes[root_node.char_code] = current_code
            self.reverse_mapping[current_code] = root_node.char_code
            return

        self._make_codes_helper(root_node.left, current_code + "0")
        self._make_codes_helper(root_node.right, current_code + "1")

    def _build_codes_from_tree(self, root_node):
        """Initiates Huffman code generation."""
        self.codes = {}
        self.reverse_mapping = {}
        if root_node:
             # Handle special case: if root is a leaf (only one symbol in input)
            if root_node.char_code is not None:
                self._make_codes_helper(root_node, "0") # Assign '0' or any default code
            else:
                self._make_codes_helper(root_node, "")


    def _get_encoded_data(self, data):
        """Encodes the input data using the generated Huffman codes."""
        encoded_text = ""
        for byte_val in data:
            encoded_text += self.codes[byte_val]
        return encoded_text

    def _pad_encoded_data(self, encoded_text):
        """Pads the encoded bit string to make its length a multiple of 8."""
        extra_padding = 8 - (len(encoded_text) % 8)
        if extra_padding == 8: # No padding needed if already multiple of 8
            extra_padding = 0
        
        padded_encoded_text = encoded_text + ('0' * extra_padding)
        padded_info = extra_padding # Number of padding bits added
        return padded_encoded_text, padded_info

    def _get_byte_array(self, padded_encoded_text):
        """Converts the padded bit string to a byte array."""
        if len(padded_encoded_text) % 8 != 0:
            raise ValueError("Encoded text not padded correctly.")
        
        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2))
        return b

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
        output_file_path = os.path.join(output_folder, file_name_no_ext + ".huff")
        
        os.makedirs(output_folder, exist_ok=True)

        if not data: # Handle empty file
            with open(output_file_path, 'wb') as f_out:
                ext_bytes = file_extension.encode('utf-8')
                f_out.write(struct.pack('!H', len(ext_bytes)))
                f_out.write(ext_bytes)
                f_out.write(struct.pack('!B', 0)) # Padding bits
                f_out.write(struct.pack('!I', 0)) # Freq table length
                # No frequency table, no data
            return output_file_path

        frequency_dict = self._make_frequency_dict(data)
        # Convert integer keys to strings for JSON serialization
        serializable_freq_dict = {str(k): v for k, v in frequency_dict.items()}
        freq_table_json = json.dumps(serializable_freq_dict).encode('utf-8')

        heap = self._build_heap(frequency_dict)
        huffman_tree_root = self._merge_nodes(heap)
        self._build_codes_from_tree(huffman_tree_root)
        
        if not self.codes and data: # Should not happen if data is not empty and tree built
            # This can happen if only one unique character exists and tree building is minimal
            # For example, if data is "AAAAA", freq_dict is {65:5}.
            # Tree root might be a leaf. _build_codes_from_tree assigns "0".
            if len(frequency_dict) == 1: # Only one unique character
                char_code = list(frequency_dict.keys())[0]
                self.codes = {char_code: "0"} # Assign a default code, e.g., "0"
            else: # Should not happen if data is present
                 raise RuntimeError("Huffman codes could not be generated.")


        encoded_data_str = self._get_encoded_data(data)
        padded_encoded_data, padding_info = self._pad_encoded_data(encoded_data_str)
        output_bytes = self._get_byte_array(padded_encoded_data)

        try:
            with open(output_file_path, 'wb') as f_out:
                # 1. Original Extension
                ext_bytes = file_extension.encode('utf-8')
                f_out.write(struct.pack('!H', len(ext_bytes)))
                f_out.write(ext_bytes)
                # 2. Padding Bits Count
                f_out.write(struct.pack('!B', padding_info))
                # 3. Frequency Table Length
                f_out.write(struct.pack('!I', len(freq_table_json)))
                # 4. Frequency Table
                f_out.write(freq_table_json)
                # 5. Compressed Data
                f_out.write(output_bytes)
        except IOError as e:
            raise IOError(f"Error writing compressed file {output_file_path}: {e}")
            
        return output_file_path

    def _rebuild_huffman_tree_from_freq(self, freq_dict_str_keys):
        """Rebuilds Huffman tree from a frequency dictionary with string keys."""
        # Convert string keys back to int byte values
        freq_dict = {int(k): v for k, v in freq_dict_str_keys.items()}
        if not freq_dict: return None # Handle empty frequency dict (e.g. for empty original file)

        heap = self._build_heap(freq_dict)
        return self._merge_nodes(heap)


    def decompress(self, input_file_path, output_folder):
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Compressed file not found: {input_file_path}")

        os.makedirs(output_folder, exist_ok=True)
        output_file_path = ""

        try:
            with open(input_file_path, 'rb') as f_in:
                # 1. Read Original Extension
                ext_len_bytes = f_in.read(2)
                if not ext_len_bytes: raise ValueError("Invalid Huffman file: missing ext length.")
                ext_len = struct.unpack('!H', ext_len_bytes)[0]
                
                ext_bytes = f_in.read(ext_len)
                if len(ext_bytes) != ext_len: raise ValueError("Invalid Huffman file: incomplete ext string.")
                original_extension = ext_bytes.decode('utf-8')

                base_name = os.path.basename(input_file_path)
                file_name_no_ext, _ = os.path.splitext(base_name)
                output_file_path = os.path.join(output_folder, file_name_no_ext + "_decompressed" + original_extension)

                # 2. Read Padding Bits Count
                padding_info_byte = f_in.read(1)
                if not padding_info_byte: raise ValueError("Invalid Huffman file: missing padding info.")
                padding_info = struct.unpack('!B', padding_info_byte)[0]

                # 3. Read Frequency Table Length
                freq_table_len_bytes = f_in.read(4)
                if not freq_table_len_bytes: raise ValueError("Invalid Huffman file: missing freq table length.")
                freq_table_len = struct.unpack('!I', freq_table_len_bytes)[0]

                # 4. Read Frequency Table
                if freq_table_len > 0:
                    freq_table_json = f_in.read(freq_table_len)
                    if len(freq_table_json) != freq_table_len:
                        raise ValueError("Invalid Huffman file: incomplete freq table.")
                    serializable_freq_dict = json.loads(freq_table_json.decode('utf-8'))
                    # Convert string keys from JSON back to int for freq_dict
                    freq_dict_for_tree = {int(k): v for k,v in serializable_freq_dict.items()}
                else: # Empty file was compressed
                    freq_dict_for_tree = {}


                # 5. Read Compressed Data
                compressed_byte_data = f_in.read()

                if not freq_dict_for_tree and compressed_byte_data:
                    raise ValueError("Data present but no frequency table for non-empty file.")
                if not freq_dict_for_tree and not compressed_byte_data: # Empty file correctly handled
                    with open(output_file_path, 'wb') as f_out: # Create empty file
                        pass
                    return output_file_path


                # Rebuild Huffman tree and codes for decompression
                huffman_tree_root = self._rebuild_huffman_tree_from_freq(freq_dict_for_tree)
                # We need the reverse mapping (code_str -> char_code) which is built by _build_codes_from_tree
                self._build_codes_from_tree(huffman_tree_root) # This populates self.reverse_mapping

                if not self.reverse_mapping and compressed_byte_data:
                    # This can happen if only one unique character exists
                    if len(freq_dict_for_tree) == 1:
                         char_code = list(freq_dict_for_tree.keys())[0]
                         self.reverse_mapping = {"0": char_code} # Code was "0"
                    else:
                        raise RuntimeError("Huffman reverse codes could not be generated for decompression.")


                encoded_bit_string = ""
                for byte_val in compressed_byte_data:
                    encoded_bit_string += bin(byte_val)[2:].rjust(8, '0')
                
                if padding_info > 0:
                    encoded_bit_string = encoded_bit_string[:-padding_info]

                decompressed_data = bytearray()
                current_code = ""
                
                # Handle case of single unique character file (e.g. "AAAAA")
                # Its code might be "0". If encoded_bit_string is "00000"
                if len(self.reverse_mapping) == 1 and list(self.reverse_mapping.keys())[0] * len(encoded_bit_string) == encoded_bit_string:
                    single_char_code = list(self.reverse_mapping.keys())[0]
                    single_char_val = self.reverse_mapping[single_char_code]
                    num_chars = len(encoded_bit_string) // len(single_char_code)
                    for _ in range(num_chars):
                        decompressed_data.append(single_char_val)
                else: # General case
                    for bit in encoded_bit_string:
                        current_code += bit
                        if current_code in self.reverse_mapping:
                            char_code = self.reverse_mapping[current_code]
                            decompressed_data.append(char_code)
                            current_code = ""
                
                if current_code: # Should be empty if all codes were valid
                    # This might indicate an error or an incomplete final code.
                    # However, for valid Huffman codes, this shouldn't happen if data is not corrupt.
                    pass


            with open(output_file_path, 'wb') as f_out:
                f_out.write(decompressed_data)
        
        except (IOError, struct.error, json.JSONDecodeError, ValueError) as e:
            if output_file_path and os.path.exists(output_file_path):
                # os.remove(output_file_path) # Optional
                pass
            raise RuntimeError(f"Error during Huffman decompression of {input_file_path}: {e}")
            
        return output_file_path

