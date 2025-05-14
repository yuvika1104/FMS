import heapq
import os
import json

# Define COMPRESSED_FOLDER (can be overridden by Flask app)
# Ensure this matches the path used in app.py
COMPRESSED_FOLDER = 'compressed'

class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        # Define comparison for heapq
        def __lt__(self, other):
            return self.freq < other.freq

    def make_frequency_dict(self, text):
        """Calculates the frequency of each character in the text."""
        frequency = {}
        for character in text:
            if character not in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self, frequency):
        """Creates a min-heap of HeapNode objects based on character frequencies."""
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        """Merges nodes in the heap to build the Huffman tree."""
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            # Create a new internal node with combined frequency
            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        """Recursive helper function to generate Huffman codes."""
        if root is None:
            return
        # If it's a leaf node, assign the code and reverse mapping
        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return
        # Traverse left (0) and right (1)
        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        """Generates Huffman codes from the built tree."""
        # The root of the tree is the only node left in the heap
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        """Encodes the input text using the generated Huffman codes."""
        encoded_text = ""
        for character in text:
            # Look up the code for each character
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self, encoded_text):
        """Pads the encoded text to ensure its length is a multiple of 8."""
        extra_padding = 8 - len(encoded_text) % 8
        # Add padding bits (all '0's)
        for i in range(extra_padding):
            encoded_text += "0"
        # Prepend an 8-bit binary string indicating the amount of padding
        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        """Converts the padded encoded text (binary string) into a byte array."""
        if len(padded_encoded_text) % 8 != 0:
            # This should not happen if padding is done correctly
            print("Encoded text not padded properly")
            # Consider raising an error here instead of exiting
            exit(1)

        b = bytearray()
        # Process 8 bits at a time
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2)) # Convert binary string to integer and append as byte
        return b

    def compress(self, file_path, output_folder):
        """Compresses a text file using Huffman coding."""
        filename, file_extension = os.path.splitext(os.path.basename(file_path))
        # Output file for compressed data (binary)
        output_path = os.path.join(output_folder, filename + ".bin")
        # Output file for the character-to-code mapping (JSON)
        # This mapping is needed for decompression
        mapping_path = os.path.join(output_folder, filename + "_mapping.json") # Save mapping in output_folder

        # Open the input file with explicit UTF-8 encoding
        # 'errors='ignore'' will skip characters that cannot be decoded
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()

        # Perform Huffman encoding steps
        frequency = self.make_frequency_dict(text)
        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()

        # Ensure the output directory exists before saving files
        os.makedirs(output_folder, exist_ok=True)

        # Save the reverse mapping (code to character) to a JSON file
        # This mapping is crucial for the decompressor
        with open(mapping_path, 'w', encoding='utf-8') as mapping_file: # Save mapping with UTF-8 encoding
            json.dump(self.reverse_mapping, mapping_file)

        # Get the encoded binary string
        encoded_text = self.get_encoded_text(text)
        # Pad the encoded string
        padded_encoded_text = self.pad_encoded_text(encoded_text)
        # Convert the padded string to a byte array
        byte_array = self.get_byte_array(padded_encoded_text)

        # Write the byte array to the output file
        with open(output_path, 'wb') as output:
            output.write(bytes(byte_array))

        # Return the path to the compressed file
        return output_path

    def decompress(self, file_path, output_folder):
        """Decompresses a Huffman-encoded binary file."""
        filename, _ = os.path.splitext(os.path.basename(file_path))
        # Output file for decompressed text
        output_path = os.path.join(output_folder, filename + "_decompressed.txt")
        # Path to the mapping file
        mapping_path = os.path.join(output_folder, filename + "_mapping.json") # Load mapping from output_folder

        print(f"Decompressing file: {file_path}")  # Debugging
        print(f"Output path: {output_path}")  # Debugging
        print(f"Mapping path: {mapping_path}")  # Debugging

        # Load the reverse mapping from the JSON file
        try:
            with open(mapping_path, 'r', encoding='utf-8') as mapping_file: # Load mapping with UTF-8 encoding
                self.reverse_mapping = json.load(mapping_file)
                print(f"Reverse mapping loaded: {self.reverse_mapping}")  # Debugging
        except FileNotFoundError:
             print(f"Error: Mapping file not found at {mapping_path}")
             raise FileNotFoundError(f"Mapping file not found at {mapping_path}. Cannot decompress.")
        except json.JSONDecodeError:
             print(f"Error: Could not decode JSON from mapping file {mapping_path}")
             raise json.JSONDecodeError(f"Invalid JSON in mapping file {mapping_path}. Cannot decompress.", mapping_file.read(), 0)
        except Exception as e:
            print(f"Error loading reverse mapping: {e}")  # Debugging
            raise e

        # Read the binary compressed file
        with open(file_path, 'rb') as file:
            bit_string = ""
            byte = file.read(1)
            while byte:
                # Convert each byte to its 8-bit binary string representation
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

        # Extract padding information from the first 8 bits
        if len(bit_string) < 8:
             print("Error: Compressed file is too short.")
             raise ValueError("Invalid compressed file format: too short.")

        padded_info = bit_string[:8]
        extra_padding = int(padded_info, 2)

        # Remove the padding info and the extra padding bits
        bit_string = bit_string[8:]
        if extra_padding > 0:
             bit_string = bit_string[:-extra_padding]

        current_code = ""
        decompressed_text = ""

        # Decode the bit string using the reverse mapping
        for bit in bit_string:
            current_code += bit
            if current_code in self.reverse_mapping:
                character = self.reverse_mapping[current_code]
                decompressed_text += character
                current_code = "" # Reset for the next character

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)
        # Write the decompressed text to the output file with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as output:
            output.write(decompressed_text)

        return output_path
