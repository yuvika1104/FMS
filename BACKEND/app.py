from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import mimetypes  # For better file type detection
import json  # For creating JSON responses
from werkzeug.utils import secure_filename  # For secure filenames
import collections  # For keyword density
import subprocess # To run external commands like ffprobe
import re # For parsing ffprobe output

# Assuming your compression modules are in the same directory or accessible via PYTHONPATH
# These should be the REWRITTEN versions
from LZ77 import LZ77
from HUFFMANN import HuffmanCoding
from lzw import LZWCompression # Assuming filename is lzw.py

# Other existing compression modules
from COLORLOSSY import ColorLossyImageCompressor
from DEFLATE import DeflateCoding
from LOSSLESS import LosslessImageCompressor
from LOSSY import LossyImageCompressor
import HEVC_AND_AVC as VideoCompressor

# Import the FileArchiver class
from archive import FileArchiver

# Import the new compression classes
from DCTCompression import DCTCompression
from DitheringCompression import DitheringCompression
from RLE import RLECompression


# --- Pillow (PIL) for image analysis, pdfminer for PDF ---
try:
    from PIL import Image, ImageStat
except ImportError:
    Image = None  # PIL not installed
    ImageStat = None

try:
    from pdfminer.high_level import extract_text  # For PDF text extraction
except ImportError:
    extract_text = None  # pdfminer.six not installed


app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Define folders for uploads, compressed files, decompressed files, and archives
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
COMPRESSED_FOLDER = os.path.join(BASE_DIR, 'compressed')
DECOMPRESSED_FOLDER = os.path.join(BASE_DIR, 'decompressed') # New folder for decompressed files
ARCHIVE_FOLDER = os.path.join(BASE_DIR, 'archives')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)
os.makedirs(DECOMPRESSED_FOLDER, exist_ok=True) # Create the decompressed folder
os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

# Initialize the FileArchiver
file_archiver = FileArchiver(UPLOAD_FOLDER, ARCHIVE_FOLDER)


# --- Compression Algorithm Mapping ---
# Ensure these use the rewritten classes for LZ77, Huffman, LZW
# Updated file_types for text compressors to include 'pdf'
# Added DCT, Dithering, and RLE
ALGORITHM_MAP = {
    'LZ77': {'class': LZ77, 'method': 'compress', 'file_types': ['text', 'pdf']},
    'Huffman': {'class': HuffmanCoding, 'method': 'compress', 'file_types': ['text', 'pdf']},
    'LZW': {'class': LZWCompression, 'method': 'compress', 'file_types': ['text', 'pdf']},
    'RLE': {'class': RLECompression, 'method': 'compress', 'file_types': ['text', 'pdf']}, # RLE for text/pdf
    'ColorLossy': {'class': ColorLossyImageCompressor, 'method': 'compress', 'file_types': ['image']},
    'Deflate': {'class': DeflateCoding, 'method': 'compress', 'file_types': ['image']},
    'Lossless': {'class': LosslessImageCompressor, 'method': 'compress', 'file_types': ['image']},
    'Lossy': {'class': LossyImageCompressor, 'method': 'compress', 'file_types': ['image']},
    'DCT': {'class': DCTCompression, 'method': 'compress', 'file_types': ['image']}, # DCT for images
    'Dithering': {'class': DitheringCompression, 'method': 'compress', 'file_types': ['image']}, # Dithering for images
    'HEVC': {'function': VideoCompressor.compress_mp4_hevc, 'file_types': ['video']},
    'AVC': {'function': VideoCompressor.compress_mp4_avc, 'file_types': ['video']},
}

# --- Decompression Algorithm Mapping ---
# Added DCT and RLE decompression. Dithering outputs a standard image, no separate decompress needed.
DECOMPRESSION_ALGORITHM_MAP = {
    'LZ77': {'class': LZ77, 'method': 'decompress', 'expected_extension': '.lz77'},
    'Huffman': {'class': HuffmanCoding, 'method': 'decompress', 'expected_extension': '.huff'},
    'LZW': {'class': LZWCompression, 'method': 'decompress', 'expected_extension': '.lzw'}, # Assuming new LZW saves as .lzw
    'RLE': {'class': RLECompression, 'method': 'decompress', 'expected_extension': '.rle'}, # RLE decompression
    'DCT': {'class': DCTCompression, 'method': 'decompress', 'expected_extension': '.dctz'}, # DCT decompression
    # Dithering does not have a specific decompression method as it outputs a standard image file
}

# --- Helper function to determine file category ---
def get_file_category(filepath):
    """Determines the general category of a file based on its MIME type or extension."""
    mime_type, _ = mimetypes.guess_type(filepath)
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    # Check for specific compressed extensions first for decompression purposes
    # Added .rle and .dctz extensions
    if ext in ['.lz77', '.lzw', '.huff', '.cmp', '.zlib', '.rle', '.dctz']:
        return 'compressed'

    if mime_type:
        if mime_type.startswith('text/'):
            return 'text'
        if mime_type.startswith('application/pdf'):
            return 'pdf' # Explicitly pdf
        if mime_type.startswith('image/'):
            return 'image'
        if mime_type.startswith('video/'):
            return 'video'
        if mime_type.startswith('audio/'):
            return 'audio'

    # Fallback for unknown or common extensions if MIME type is not specific enough
    if ext in ['.txt', '.csv', '.log', '.py', '.js', '.html', '.css', '.md', '.json', '.xml']:
        return 'text'
    if ext == '.pdf':
        return 'pdf'
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp']:
        return 'image'
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    if ext in ['.mp3', '.wav', '.ogg', '.aac', '.flac']:
        return 'audio'
    return 'other'

# --- Helper function to convert RGB to Hex ---
def rgb_to_hex(rgb):
    """Converts an RGB tuple to a hex color string."""
    if isinstance(rgb, int):  # Grayscale
        return '#{0:02x}{0:02x}{0:02x}'.format(rgb)
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# --- Helper function to get video info using ffprobe ---
def get_video_info_ffprobe(filepath):
    """
    Extracts video information (duration, resolution, frame rate, codec)
    using the ffprobe command-line tool.
    """
    if not os.path.exists(filepath):
        return None, "File not found."
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=duration,width,height,avg_frame_rate,codec_name',
        '-of', 'json', filepath
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        ffprobe_output = json.loads(result.stdout)
        video_info = {}
        if 'streams' in ffprobe_output and len(ffprobe_output['streams']) > 0:
            stream = ffprobe_output['streams'][0]
            if 'duration' in stream:
                try: video_info['duration_seconds'] = round(float(stream['duration']), 2)
                except: video_info['duration_seconds'] = 'N/A'
            if 'width' in stream and 'height' in stream:
                video_info['resolution'] = f"{stream['width']}x{stream['height']}"
            else: video_info['resolution'] = 'N/A'
            if 'avg_frame_rate' in stream:
                 try:
                     num, den = map(int, stream['avg_frame_rate'].split('/'))
                     video_info['frame_rate'] = round(num / den, 2) if den != 0 else 'N/A'
                 except: video_info['frame_rate'] = 'N/A'
            else: video_info['frame_rate'] = 'N/A'
            if 'codec_name' in stream: video_info['codec_info'] = stream['codec_name']
            else: video_info['codec_info'] = 'N/A'
            return video_info, None
        else: return None, "No video stream found."
    except FileNotFoundError: return None, "ffprobe not found. Is FFmpeg installed and in PATH?"
    except subprocess.CalledProcessError as e: return None, f"ffprobe error: {e.stderr.strip()}"
    except json.JSONDecodeError: return None, "Failed to parse ffprobe output."
    except Exception as e: return None, f"Unexpected ffprobe error: {str(e)}"


# --- Compression Endpoint ---
@app.route('/compress', methods=['POST'])
def compress():
    try:
        if 'file' not in request.files or 'algorithm' not in request.form:
            return jsonify({'error': 'Missing file or algorithm'}), 400

        file = request.files['file']
        algorithm_name = request.form['algorithm']

        if not file or not file.filename:
            return jsonify({'error': 'No file selected or filename is empty'}), 400
        if not algorithm_name: return jsonify({'error': 'No algorithm selected'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        file_category = get_file_category(file_path)
        # For compression, 'pdf' files are treated as 'text' if a text algorithm is chosen
        if file_category == 'pdf' and algorithm_name in ['LZ77', 'Huffman', 'LZW', 'RLE']: # Added RLE
            effective_category_for_algo = 'text'
        else:
            effective_category_for_algo = file_category


        if algorithm_name not in ALGORITHM_MAP:
            return jsonify({'error': f'Unsupported algorithm: {algorithm_name}'}), 400

        algo_config = ALGORITHM_MAP[algorithm_name]
        if effective_category_for_algo not in algo_config['file_types']:
            return jsonify({'error': f'{algorithm_name} is not supported for {file_category} files (mapped to {effective_category_for_algo})'}), 400

        output_path = None
        if 'class' in algo_config:
            compressor_instance = algo_config['class']()
            # Pass any specific parameters needed by the new algorithms if available in the request form
            # For Dithering, we might need num_colors. For DCT, quality/block_size.
            # This example assumes default parameters for simplicity.
            if algorithm_name == 'Dithering':
                 # Example: Get num_colors from form, default to 16
                 num_colors = int(request.form.get('num_colors', 16))
                 output_path = compressor_instance.compress(file_path, COMPRESSED_FOLDER, num_colors=num_colors)
            elif algorithm_name == 'DCT':
                 # Example: Get quality from form, default to 50
                 quality = int(request.form.get('quality', 50))
                 output_path = compressor_instance.compress(file_path, COMPRESSED_FOLDER, quality=quality)
            else:
                 output_path = getattr(compressor_instance, algo_config['method'])(file_path, COMPRESSED_FOLDER)

        elif 'function' in algo_config:
            base, ext = os.path.splitext(filename)
            # Ensure the compressed extension is meaningful, e.g., _hevc.mp4
            output_filename = f"{base}_{algorithm_name.lower()}_compressed{ext}"
            output_path_for_video = os.path.join(COMPRESSED_FOLDER, output_filename)
            algo_config['function'](file_path, output_path_for_video)
            output_path = output_path_for_video

        if not output_path or not os.path.exists(output_path):
            app.logger.error(f"Compression failed. Output path: {output_path}")
            return jsonify({'error': 'Compression failed or output file not found.'}), 500

        # Clean up uploaded file after successful compression
        if os.path.exists(file_path):
            os.remove(file_path)

        return send_file(output_path, as_attachment=True, download_name=os.path.basename(output_path))

    except Exception as e:
        app.logger.error(f"Compression error: {str(e)}", exc_info=True)
        # Clean up uploaded file in case of error too
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


# --- Decompression Endpoint ---
@app.route('/decompress', methods=['POST'])
def decompress():
    try:
        if 'file' not in request.files or 'algorithm' not in request.form:
            return jsonify({'error': 'Missing file or algorithm for decompression'}), 400

        file = request.files['file']
        algorithm_name = request.form['algorithm'] # e.g., 'LZ77', 'Huffman', 'LZW', 'RLE', 'DCT'

        if not file or not file.filename:
            return jsonify({'error': 'No file selected for decompression or filename is empty'}), 400
        if not algorithm_name:
            return jsonify({'error': 'No decompression algorithm selected'}), 400

        filename = secure_filename(file.filename)
        # Save to UPLOAD_FOLDER first, as decompress methods expect a path
        uploaded_compressed_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(uploaded_compressed_path)

        if algorithm_name not in DECOMPRESSION_ALGORITHM_MAP:
            # Clean up uploaded compressed file
            if os.path.exists(uploaded_compressed_path):
                os.remove(uploaded_compressed_path)
            return jsonify({'error': f'Unsupported decompression algorithm: {algorithm_name}'}), 400

        algo_config = DECOMPRESSION_ALGORITHM_MAP[algorithm_name]

        # Optional: Check if file extension matches expected for the algorithm
        # _, ext = os.path.splitext(filename)
        # if ext.lower() != algo_config['expected_extension']:
        #     if os.path.exists(uploaded_compressed_path): os.remove(uploaded_compressed_path)
        #     return jsonify({'error': f"File extension {ext} doesn't match expected {algo_config['expected_extension']} for {algorithm_name}"}), 400

        output_path = None
        if 'class' in algo_config:
            decompressor_instance = algo_config['class']()
            # The decompress method should save to DECOMPRESSED_FOLDER
            output_path = getattr(decompressor_instance, algo_config['method'])(uploaded_compressed_path, DECOMPRESSED_FOLDER)
        else: # Should not happen with current DECOMPRESSION_ALGORITHM_MAP
            if os.path.exists(uploaded_compressed_path): os.remove(uploaded_compressed_path)
            return jsonify({'error': 'Decompression method misconfigured.'}), 500


        if not output_path or not os.path.exists(output_path):
            app.logger.error(f"Decompression failed. Expected output_path: {output_path}")
            if os.path.exists(uploaded_compressed_path): os.remove(uploaded_compressed_path) # Clean up
            return jsonify({'error': 'Decompression failed or output file not found. Check server logs.'}), 500

        # Clean up the uploaded compressed file from UPLOAD_FOLDER after successful decompression
        if os.path.exists(uploaded_compressed_path):
            os.remove(uploaded_compressed_path)

        return send_file(output_path, as_attachment=True, download_name=os.path.basename(output_path))

    except Exception as e:
        app.logger.error(f"Decompression error: {str(e)}", exc_info=True)
        # Clean up uploaded compressed file in case of error
        if 'uploaded_compressed_path' in locals() and os.path.exists(uploaded_compressed_path):
            os.remove(uploaded_compressed_path)
        return jsonify({'error': f'An unexpected error occurred during decompression: {str(e)}'}), 500


# --- Analysis Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files or 'analysis_type' not in request.form:
            return jsonify({'error': 'Missing file or analysis_type'}), 400

        file = request.files['file']
        analysis_type = request.form['analysis_type']

        if not file or not file.filename: return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path) # Save for analysis

        file_category = get_file_category(file_path)
        analysis_results = {'filename': filename, 'analysis_type': analysis_type, 'category': file_category}
        analysis_results['file_size_bytes'] = os.path.getsize(file_path)
        analysis_results['mime_type'] = mimetypes.guess_type(file_path)[0] or 'unknown'

        if file_category == 'text' or file_category == 'pdf':
            try:
                text_content = ""
                if file_category == 'pdf' and extract_text:
                    text_content = extract_text(file_path)
                elif file_category == 'text' or (file_category == 'pdf' and not extract_text) : # Treat PDF as text if pdfminer fails or for non-text analysis types
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()

                if not text_content and file_category == 'pdf' and not extract_text:
                     analysis_results['text_analysis_error'] = "pdfminer.six not installed or PDF is image-based. Cannot extract text."


                if analysis_type == 'Word Count':
                    words = text_content.split()
                    analysis_results['word_count'] = len(words)
                elif analysis_type == 'Character Count':
                    analysis_results['character_count'] = len(text_content)
                elif analysis_type == 'Keyword Density':
                    words = [word.lower() for word in re.findall(r'\b[a-z]{3,}\b', text_content.lower())] # Min 3 char words
                    total_words_for_density = len(words)
                    if total_words_for_density > 0:
                        word_counts = collections.Counter(words)
                        top_keywords = [{'word': w, 'count': c, 'percentage': round((c / total_words_for_density) * 100, 2)}
                                        for w, c in word_counts.most_common(10)]
                        analysis_results['top_keywords'] = top_keywords
                        analysis_results['total_analyzed_words'] = total_words_for_density
                    else:
                        analysis_results['top_keywords'] = []
                        analysis_results['info'] = "Not enough words for keyword density."
            except Exception as e:
                analysis_results['text_analysis_error'] = f"Text analysis error: {str(e)}"
                app.logger.error(f"Text analysis error for {filename}: {str(e)}", exc_info=True)

        elif file_category == 'image':
            if Image and analysis_type in ['Dimension & Size', 'Color Palette', 'Metadata']:
                try:
                    with Image.open(file_path) as img:
                        if analysis_type == 'Dimension & Size':
                            analysis_results['dimensions'] = f"{img.width}x{img.height}"
                            analysis_results['format'] = img.format
                            analysis_results['mode'] = img.mode
                        elif analysis_type == 'Color Palette':
                            img_for_palette = img.convert('RGB').resize((100,100)) # Smaller for faster palette
                            raw_colors = img_for_palette.getcolors(maxcolors=256*256) # Max colors
                            if raw_colors:
                                sorted_colors = sorted(raw_colors, key=lambda x: x[0], reverse=True)
                                palette = []
                                total_pixels = sum(c[0] for c in sorted_colors)
                                for count, pixel_value in sorted_colors[:10]: # Top 10
                                    palette.append({'hex': rgb_to_hex(pixel_value), 'count': count,
                                                    'percentage': round((count/total_pixels)*100,2) if total_pixels else 0})
                                analysis_results['color_palette'] = palette
                            else: analysis_results['color_palette'] = []
                        elif analysis_type == 'Metadata':
                            metadata = {}
                            if hasattr(img, 'info'):
                                for k, v in img.info.items():
                                    try: metadata[k] = v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v
                                    except: metadata[k] = str(v)
                            exif_data = img.getexif()
                            if exif_data:
                                # from PIL.ExifTags import TAGS # Consider adding this for human-readable EXIF
                                metadata['EXIF'] = {k: v for k,v in exif_data.items() if isinstance(v, (str,int,float,bytes))}
                            analysis_results['metadata'] = metadata if metadata else "No metadata."
                except Exception as e:
                    analysis_results['image_analysis_error'] = f"Image analysis error: {str(e)}"
                    app.logger.error(f"Image analysis error for {filename}: {str(e)}", exc_info=True)
            elif not Image: analysis_results['image_analysis_error'] = "Pillow (PIL) not installed."
            else: analysis_results['info'] = f"'{analysis_type}' not available for images or Pillow issue."

        elif file_category == 'video':
            video_info, error = get_video_info_ffprobe(file_path)
            if error: analysis_results['video_analysis_error'] = error
            elif video_info:
                if analysis_type == 'Duration & Codec':
                    analysis_results.update({'duration_seconds': video_info.get('duration_seconds'), 'codec_info': video_info.get('codec_info')})
                elif analysis_type == 'Resolution': analysis_results['resolution'] = video_info.get('resolution')
                elif analysis_type == 'Frame Rate': analysis_results['frame_rate'] = video_info.get('frame_rate')
            else: analysis_results['video_analysis_error'] = "Could not retrieve video info."

        elif file_category == 'audio':
             analysis_results['audio_info'] = "Audio analysis for duration/bitrate/format not yet implemented."


        # Clean up uploaded file for analysis
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify(analysis_results), 200

    except Exception as e:
        app.logger.error(f"Analysis endpoint error: {str(e)}", exc_info=True)
        if 'file_path' in locals() and os.path.exists(file_path): # Clean up if error during analysis
            os.remove(file_path)
        return jsonify({'error': f'Unexpected analysis error: {str(e)}'}), 500

# --- Archiving Endpoint ---
@app.route('/archive', methods=['POST'])
def archive_files():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided for archiving'}), 400
        files = request.files.getlist('files[]')
        if not files or all(not f.filename for f in files): # Check if all files are empty
             return jsonify({'error': 'No files selected or files are empty'}), 400

        archive_name = request.form.get('archive_name', 'archive.zip')
        if not archive_name.lower().endswith('.zip'): # Ensure .zip extension
            archive_name += '.zip'

        # Note: FileArchiver saves files to UPLOAD_FOLDER then zips them, then cleans them up.
        archive_path = file_archiver.archive_files(files, archive_name) # UPLOAD_FOLDER is passed to FileArchiver init

        return send_file(archive_path, as_attachment=True, download_name=os.path.basename(archive_path))

    except ValueError as ve: # From FileArchiver if no files
        app.logger.error(f"Archiving ValueError: {str(ve)}")
        return jsonify({'error': f'Archiving failed: {str(ve)}'}), 400
    except RuntimeError as re: # From FileArchiver for other issues
        app.logger.error(f"Archiving RuntimeError: {str(re)}", exc_info=True)
        return jsonify({'error': f'Archiving failed: {str(re)}'}), 500
    except Exception as e:
        app.logger.error(f"Archiving endpoint error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected error during archiving: {str(e)}'}), 500


# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Backend is running and healthy!'}), 200

# --- Main execution block ---
if __name__ == '__main__':
    print("Required libraries and tools:")
    print("- Pillow: pip install Pillow")
    print("- pdfminer.six: pip install pdfminer.six")
    print("- OpenCV (for DCT): pip install opencv-python numpy") # Added OpenCV and NumPy
    print("- FFmpeg (includes ffprobe): Install from ffmpeg.org/download.html (ensure in PATH)")
    print("- Flask, Flask-Cors, Werkzeug: pip install Flask Flask-Cors Werkzeug")
    print("Ensure local compression modules (LZ77.py, HUFFMANN.py, lzw.py, RLE.py, DCTCompression.py, DitheringCompression.py, etc.) are present.") # Updated list
    app.run(debug=True, host='0.0.0.0', port=5000)
