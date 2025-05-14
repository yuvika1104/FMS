import os

class LZ77:
    def __init__(self, window_size=2048, lookahead_buffer_size=256): # Increased default sizes for better performance
        self.window_size = window_size
        self.lookahead_buffer_size = lookahead_buffer_size

    def compress(self, file_path, output_folder):
        """Compresses a text file using the LZ77 algorithm."""
        # Open the input file with explicit UTF-8 encoding
        # 'errors='ignore'' will skip characters that cannot be decoded
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            data = file.read()

        compressed_data = []
        i = 0 # Current position in the data

        while i < len(data):
            # Define the search window and lookahead buffer
            search_window_start = max(0, i - self.window_size)
            search_window = data[search_window_start:i]
            lookahead_buffer = data[i:i + self.lookahead_buffer_size]

            # Find the longest match in the search window
            match_length, match_offset = self.find_longest_match(search_window, lookahead_buffer)

            if match_length > 0:
                # If a match is found, output (offset, length, next_character)
                # The next_character is the first character *after* the match
                next_char_index = i + match_length
                # Handle case where match goes to the very end of the data
                next_character = data[next_char_index] if next_char_index < len(data) else ''
                compressed_data.append((match_offset, match_length, next_character))
                i += match_length + 1 # Move index past the match and the next character
            else:
                # If no match is found, output (0, 0, current_character)
                compressed_data.append((0, 0, data[i]))
                i += 1 # Move index to the next character

        # Determine output file path
        filename, _ = os.path.splitext(os.path.basename(file_path))
        output_path = os.path.join(output_folder, filename + ".lz77")

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Write compressed data to the output file
        # Each item is written as offset,length,character on a new line
        with open(output_path, 'w', encoding='utf-8') as output_file: # Write with UTF-8 encoding
            for item in compressed_data:
                # Ensure each line has 3 values: offset, length, char
                # Handle the case where next_character is empty string for the last token
                output_file.write(f"{item[0]},{item[1]},{item[2]}\n")

        # Return the path to the compressed file
        return output_path

    def decompress(self, file_path, output_folder):
        """Decompresses an LZ77-encoded file."""
        print(f"Decompressing file: {file_path}")  # Debugging

        # Open the compressed file with explicit UTF-8 encoding for reading the tokens
        with open(file_path, 'r', encoding='utf-8') as file:
            compressed_data = file.readlines()

        decompressed_text = ""

        for line in compressed_data:
            try:
                # Split the line into offset, length, and char
                parts = line.strip().split(',')
                # Expect exactly 3 parts: offset, length, char
                if len(parts) != 3:
                    print(f"Skipping invalid line format: {line.strip()}")  # Debugging
                    continue

                offset_str, length_str, char = parts
                offset = int(offset_str)
                length = int(length_str)

                if offset == 0 and length == 0:
                    # No match, just append the character
                    decompressed_text += char
                else:
                    # Match found, copy from previously decompressed text
                    start = len(decompressed_text) - offset
                    # Ensure we don't go out of bounds
                    if start < 0:
                        print(f"Warning: Invalid offset {offset} at line: {line.strip()}")
                        # Attempt to recover by just appending the character
                        decompressed_text += char
                        continue

                    # Copy the matched sequence
                    matched_sequence = decompressed_text[start : start + length]
                    decompressed_text += matched_sequence + char # Append the sequence and the next character

            except ValueError:
                print(f"Skipping line due to invalid number format: {line.strip()}") # Debugging for int conversion errors
                continue
            except Exception as e:
                print(f"Error processing line: {line.strip()}. Error: {e}")  # General debugging
                continue

        # Determine output file path
        filename, _ = os.path.splitext(os.path.basename(file_path))
        output_path = os.path.join(output_folder, filename + "_decompressed.txt")

        print(f"Output path: {output_path}")  # Debugging

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        # Write the decompressed text to the output file with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(decompressed_text)

        return output_path

    def find_longest_match(self, search_window, lookahead_buffer):
        """Finds the longest match of the lookahead buffer's prefix in the search window."""
        match_length = 0
        match_offset = 0 # Offset relative to the end of the search window

        # Iterate through all possible prefixes of the lookahead buffer
        for length in range(1, len(lookahead_buffer) + 1):
            substring = lookahead_buffer[:length]
            # Find the last occurrence of the substring in the search window
            offset = search_window.rfind(substring)

            if offset != -1:
                # If found, update match_length and match_offset
                match_length = length
                match_offset = len(search_window) - offset # Calculate offset from the end
            else:
                # If the current prefix is not found, longer prefixes won't be either
                break # Optimization: stop searching for longer matches

        return match_length, match_offset
