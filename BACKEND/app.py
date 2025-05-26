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
# These imports need to point to your actual module files.
# If they are in the same directory, the current imports should work.
# Example: from your_module_LZ77 import LZ77 if LZ77.py is named your_module_LZ77.py

try:
    from LZ77 import LZ77
except ImportError:
    print("Warning: LZ77 module not found. LZ77 compression/decompression will not work.")
    LZ77 = None

try:
    from HUFFMANN import HuffmanCoding # Corrected typical spelling to Huffman
except ImportError:
    try:
        from HUFFMANN import HuffmanCoding # Common alternative spelling
        print("Note: Imported HuffmanCoding from HUFFMAN module.")
    except ImportError:
        print("Warning: HuffmanCoding module (HUFFMANN or HUFFMAN) not found. Huffman compression/decompression will not work.")
        HuffmanCoding = None

try:
    from lzw import LZWCompression
except ImportError:
    print("Warning: LZWCompression module not found. LZW compression/decompression will not work.")
    LZWCompression = None

try:
    from COLORLOSSY import ColorLossyImageCompressor
except ImportError:
    print("Warning: ColorLossyImageCompressor module not found. ColorLossy compression will not work.")
    ColorLossyImageCompressor = None

try:
    from DEFLATE import DeflateCoding
except ImportError:
    print("Warning: DeflateCoding module not found. Deflate compression/decompression will not work.")
    DeflateCoding = None

try:
    from LOSSLESS import LosslessImageCompressor
except ImportError:
    print("Warning: LosslessImageCompressor module not found. Lossless compression will not work.")
    LosslessImageCompressor = None

try:
    from LOSSY import LossyImageCompressor
except ImportError:
    print("Warning: LossyImageCompressor module not found. Lossy compression will not work.")
    LossyImageCompressor = None

try:
    import HEVC_AND_AVC as VideoCompressor
except ImportError:
    print("Warning: HEVC_AND_AVC module not found. Video compression will not work.")
    VideoCompressor = None # Define it as None if import fails

try:
    from archive import FileArchiver
except ImportError:
    print("Warning: FileArchiver module not found. Archiving will not work.")
    FileArchiver = None

try:
    from DCTCompression import DCTCompression
except ImportError:
    print("Warning: DCTCompression module not found. DCT compression/decompression will not work.")
    DCTCompression = None

try:
    from DitheringCompression import DitheringCompression
except ImportError:
    print("Warning: DitheringCompression module not found. Dithering compression will not work.")
    DitheringCompression = None

try:
    from RLE import RLECompression
except ImportError:
    print("Warning: RLECompression module not found. RLE compression/decompression will not work.")
    RLECompression = None


# --- Pillow (PIL) for image analysis, pdfminer for PDF ---
try:
    from PIL import Image, ImageStat, ExifTags
    # Prepare a reverse mapping for ExifTags if Pillow is available
    # This helps in getting human-readable names for EXIF tags
    EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
except ImportError:
    print("Warning: Pillow (PIL) library not found. Image analysis features will be limited or non-functional.")
    Image = None
    ImageStat = None
    EXIF_TAGS = {} # Define as empty if Pillow not found

try:
    from pdfminer.high_level import extract_text
except ImportError:
    print("Warning: pdfminer.six library not found. PDF text extraction will not work.")
    extract_text = None


app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Define folders for uploads, compressed files, decompressed files, and archives
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
COMPRESSED_FOLDER = os.path.join(BASE_DIR, 'compressed')
DECOMPRESSED_FOLDER = os.path.join(BASE_DIR, 'decompressed')
ARCHIVE_FOLDER = os.path.join(BASE_DIR, 'archives')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)
os.makedirs(DECOMPRESSED_FOLDER, exist_ok=True)
os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

# Initialize the FileArchiver if the class was imported successfully
if FileArchiver:
    file_archiver = FileArchiver(UPLOAD_FOLDER, ARCHIVE_FOLDER)
else:
    file_archiver = None


# --- Compression Algorithm Mapping ---
# We build this map carefully, only adding algorithms whose classes were successfully imported.
ALGORITHM_MAP = {}
if LZ77:
    ALGORITHM_MAP['LZ77'] = {'class': LZ77, 'method': 'compress', 'file_types': ['text', 'pdf']}
if HuffmanCoding:
    ALGORITHM_MAP['Huffman'] = {'class': HuffmanCoding, 'method': 'compress', 'file_types': ['text', 'pdf']}
if LZWCompression:
    ALGORITHM_MAP['LZW'] = {'class': LZWCompression, 'method': 'compress', 'file_types': ['text', 'pdf']}
if RLECompression:
    ALGORITHM_MAP['RLE'] = {'class': RLECompression, 'method': 'compress', 'file_types': ['text', 'pdf']}
if ColorLossyImageCompressor:
    ALGORITHM_MAP['ColorLossy'] = {'class': ColorLossyImageCompressor, 'method': 'compress', 'file_types': ['image']}
if DeflateCoding:
    ALGORITHM_MAP['Deflate'] = {'class': DeflateCoding, 'method': 'compress', 'file_types': ['image']} # Deflate is often for images (PNG) or general data
if LosslessImageCompressor:
    ALGORITHM_MAP['Lossless'] = {'class': LosslessImageCompressor, 'method': 'compress', 'file_types': ['image']}
if LossyImageCompressor:
    ALGORITHM_MAP['Lossy'] = {'class': LossyImageCompressor, 'method': 'compress', 'file_types': ['image']}
if DCTCompression:
    ALGORITHM_MAP['DCT'] = {'class': DCTCompression, 'method': 'compress', 'file_types': ['image']}
if DitheringCompression:
    ALGORITHM_MAP['Dithering'] = {'class': DitheringCompression, 'method': 'compress', 'file_types': ['image']}
if VideoCompressor and hasattr(VideoCompressor, 'compress_mp4_hevc'):
    ALGORITHM_MAP['HEVC'] = {'function': VideoCompressor.compress_mp4_hevc, 'file_types': ['video']}
if VideoCompressor and hasattr(VideoCompressor, 'compress_mp4_avc'):
    ALGORITHM_MAP['AVC'] = {'function': VideoCompressor.compress_mp4_avc, 'file_types': ['video']}


# --- Decompression Algorithm Mapping ---
DECOMPRESSION_ALGORITHM_MAP = {}
if LZ77:
    DECOMPRESSION_ALGORITHM_MAP['LZ77'] = {'class': LZ77, 'method': 'decompress', 'expected_extension': '.lz77'}
if HuffmanCoding:
    DECOMPRESSION_ALGORITHM_MAP['Huffman'] = {'class': HuffmanCoding, 'method': 'decompress', 'expected_extension': '.huff'}
if LZWCompression:
    DECOMPRESSION_ALGORITHM_MAP['LZW'] = {'class': LZWCompression, 'method': 'decompress', 'expected_extension': '.lzw'} # Or .cmp
if RLECompression:
    DECOMPRESSION_ALGORITHM_MAP['RLE'] = {'class': RLECompression, 'method': 'decompress', 'expected_extension': '.rle'}
if DCTCompression:
    DECOMPRESSION_ALGORITHM_MAP['DCT'] = {'class': DCTCompression, 'method': 'decompress', 'expected_extension': '.dctz'}
if DeflateCoding:
     # Deflate typically outputs raw data, often wrapped in zlib format or as part of PNG.
     # The extension '.zlib' is a common convention for raw zlib/deflate streams.
    DECOMPRESSION_ALGORITHM_MAP['Deflate'] = {'class': DeflateCoding, 'method': 'decompress', 'expected_extension': '.zlib'}
# Note: ColorLossy, Lossless, Lossy, Dithering, HEVC, AVC typically output standard file formats (e.g., PNG, MP4)
# that don't require a custom "decompression" step through these algorithms. They are "decompressed" by standard viewers/players.


# --- NEW: Algorithm Details for Frontend Info Card ---
ALGORITHM_INFO_DETAILS = {
    'LZ77': {
        'text': { # Also used for PDF via frontend logic
            'description': 'Good for repetitive text/data patterns. Finds repeated sequences and replaces them with references.',
            'pros': ["Effective for data with repeating sequences (e.g., long strings of text, structured data).", "Lossless compression: no data is lost.", "Relatively simple concept."],
            'cons': ["Can be less effective for random or highly diverse data with few repetitions.", "Dictionary overhead can be an issue for very small files or if repetitions are sparse.", "Performance can vary based on window size and search buffer implementation."],
            'expectedOutcome': "Expect significant size reduction for text files with notable repeated sequences. The output will be a .lz77 compressed file."
        }
    },
    'Huffman': {
        'text': { # Also used for PDF
            'description': 'Optimal for character frequency encoding. Assigns shorter codes to more frequent characters.',
            'pros': ["Optimal prefix code generation for a given symbol frequency distribution (achieves best possible per-symbol average code length).", "Lossless compression.", "Relatively straightforward to implement decoder."],
            'cons': ["Requires two passes over the data (one to build frequency table, one to encode) or statistical modeling.", "Less effective if character/symbol frequencies are very uniform.", "Each symbol must be encoded with at least one bit."],
            'expectedOutcome': "Good compression for text where some characters are much more frequent than others. Output will be a .huff file containing the Huffman tree and encoded data."
        }
    },
    'LZW': {
        'text': { # Also used for PDF
            'description': 'Dictionary-based lossless compression. Builds a string translation table from the input data.',
            'pros': ["Good general-purpose lossless compression, effective on a variety of data types.", "Builds its dictionary on the fly from the input data.", "No need for two passes over the data for dictionary building (unlike static Huffman)."],
            'cons': ["Some older forms were patented (many patents have now expired).", "The dictionary can become large, potentially consuming memory.", "Performance can degrade if the dictionary fills and needs resetting or complex management."],
            'expectedOutcome': "Effective compression for many text and data files, especially those with common substrings. Output will be a .lzw file."
        }
    },
    'RLE': {
        'text': { # Also used for PDF
            'description': 'Simple run-length encoding. Replaces sequences of identical data values (runs) with a count and a single value.',
            'pros': ["Very fast and extremely simple to implement.", "Highly effective for data with long runs of identical characters/bytes (e.g., simple bitmaps, uncompressed TIFFs, certain types of sparse data).", "Lossless compression."],
            'cons': ["Ineffective, and can even increase file size, for data without significant runs (e.g., random data, most natural language text).", "Basic RLE is limited in handling variations within runs."],
            'expectedOutcome': "Significant compression only for files with long sequences of the same byte/character. For typical text, it might not be very effective. Output: .rle file."
        }
    },
    'ColorLossy': {
        'image': {
            'description': 'Achieves higher compression by discarding some image data that may be less perceptible to the human eye while retaining most of the color information.',
            'pros': ["Can significantly reduce file size by quantizing the pixel values.", "The level of quantization is optimized to retain most of the information while compressing the size.", "Useful for images where exact color fidelity is not paramount (e.g., web graphics, icons)."],
            'cons': ["Lossy: Image quality is degraded, some original color information is lost.", "Visible banding or posterization can occur if color reduction is too aggressive.", "Not suitable for photographic images requiring high color accuracy or smooth gradients."],
            'expectedOutcome': "A significantly smaller file size, but with some reduction in image quality."
        }
    },
    'Deflate': { # Typically for images within PNG, or general data (zlib)
        'image': {
            'description': 'General-purpose lossless compression algorithm combining LZ77 and Huffman coding. Used in PNG, zlib, gzip.',
            'pros': ["Good lossless compression ratio for many image types (especially indexed color or those with repetitive patterns).", "Widely supported and a standard in many file formats and protocols.", "No loss of image quality."],
            'cons': ["Can be slower than simpler lossless methods.", "Not as effective as dedicated lossy image codecs (like JPEG/DCT) for photographic images if file size is the primary concern."],
            'expectedOutcome': "Reduces file size without any loss of image quality. The output from this system will be a raw Deflate stream, typically saved as a .zlib file."
        }
    },
    'Lossless': { # Generic lossless image compression
        'image': {
            'description': 'Preserves image quality completely by only working on the metadata.',
            'pros': ["No loss of image information or quality.", "Suitable for archival, or when any quality degradation is unacceptable.", "Can still offer good compression for certain image types (e.g., graphics, images with large flat areas) where metadata is not required."],
            'cons': ["Compression ratios are generally lower than lossy methods, especially for photographic content.", "Can be slower than some lossy methods."],
            'expectedOutcome': "A smaller file size with perfect image fidelity retained. The output will be an image in a lossless format (e.g., PNG) or a custom lossless compressed format."
        }
    },
    'Lossy': { # Generic lossy image compression
        'image': {
            'description': 'Achieves higher compression by discarding some image data that may be less perceptible to the human eye.Looses Color information.',
            'pros': ["Achieves much higher compression ratios than lossless methods, leading to significantly smaller file sizes.", "Very effective for photographic images where some detail loss is often acceptable.", "The level of quantization is optimized to retain most of the information while compressing the size."],
            'cons': ["Irreversible loss of image data; original quality cannot be recovered.", "Artifacts (e.g., blocking, ringing, blurring) can appear, especially at high compression levels or with repeated re-compression.", "Not suitable for images where perfect accuracy is critical (e.g., medical scans, line art)."],
            'expectedOutcome': "A significantly smaller file size, but with some reduction in image quality and Black and White in color."
        }
    },
    'DCT': {
        'image': {
            'description': 'Discrete Cosine Transform based compression, a core part of JPEG. It is a lossy technique.',
            'pros': ["Very effective for photographic and naturalistic images, providing good compression ratios.", "Allows for fine-grained quality control, balancing file size against visual fidelity.", "Well-understood and widely implemented (basis of JPEG)."],
            'cons': ["Lossy: Introduces artifacts like blocking (especially at edges) and ringing at low quality settings.", "Computationally more intensive than some simpler methods.", "Not ideal for images with sharp lines or text, where artifacts can be more noticeable."],
            'expectedOutcome': "Substantial size reduction for images, particularly photos. The quality is adjustable via a quality parameter (1-100). The output file will be in a custom .dctz format."
        }
    },
    'Dithering': {
        'image': {
            'description': 'Reduces the color palette but uses visual noise (patterns) to simulate more colors. Primarily a visual enhancement for color quantization.',
            'pros': ["Improves perceived image quality when reducing to a very limited color palette by minimizing visible banding.", "Can make low-color images (e.g., 16-color GIF) look more natural or detailed.", "File size is reduced due to the smaller color palette."],
            'cons': ["Introduces a noticeable noise pattern or texture to the image.", "Technically lossy due to color reduction, though it aims to preserve the visual impression.", "Effectiveness varies greatly with the image and the dithering algorithm used."],
            'expectedOutcome': "The image will have a reduced number of actual colors, but dithering patterns will attempt to simulate the original color range, leading to a smaller file. Output is typically a standard image format (e.g., PNG) with the dithered effect applied."
        }
    },
    'HEVC': {
        'video': {
            'description': 'High Efficiency Video Coding (H.265). A modern video codec offering improved compression over AVC.',
            'pros': ["Significantly better compression efficiency than AVC (H.264) – roughly 50% smaller file size for the same visual quality.", "Supports higher resolutions (e.g., 4K, 8K) and frame rates more efficiently.", "Offers advanced features like improved color depth and dynamic range support."],
            'cons': ["More computationally intensive to encode and decode, requiring more powerful hardware.", "May have licensing/royalty considerations for commercial use.", "Older devices may lack hardware decoding support, leading to high CPU usage during playback."],
            'expectedOutcome': "A much smaller video file size for the same visual quality compared to older codecs like AVC, or significantly better quality at similar file sizes. The output will be an .mp4 (or other container) with HEVC (H.265) encoded video."
        }
    },
    'AVC': {
        'video': {
            'description': 'Advanced Video Coding (H.264). A widely used video codec known for good quality and reasonable file sizes.',
            'pros': ["Widely supported across almost all devices, browsers, and software.", "Mature technology with broad hardware acceleration for encoding and decoding.", "Offers a good balance between compression efficiency and computational requirements."],
            'cons': ["Less efficient in terms of compression compared to newer codecs like HEVC or AV1 (i.e., larger files for the same quality).", "May struggle with very high resolutions (4K+) compared to HEVC."],
            'expectedOutcome': "Good compression for video, resulting in .mp4 (or other container) files that are highly compatible. Quality is generally good for the file size."
        }
    }
}


# --- Helper function to determine file category ---
def get_file_category(filepath):
    """Determines the general category of a file based on its MIME type or extension."""
    mime_type, _ = mimetypes.guess_type(filepath)
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    # Check for specific compressed extensions first, as these are primary for decompression logic
    compressed_extensions = [
        DECOMPRESSION_ALGORITHM_MAP[algo]['expected_extension']
        for algo in DECOMPRESSION_ALGORITHM_MAP
        if DECOMPRESSION_ALGORITHM_MAP[algo].get('expected_extension')
    ]
    if ext in compressed_extensions:
        return 'compressed'
    if ext == '.cmp': # LZW variant often uses .cmp
        return 'compressed'


    if mime_type:
        if mime_type.startswith('text/'): return 'text'
        if mime_type == 'application/pdf': return 'pdf'
        if mime_type.startswith('image/'): return 'image'
        if mime_type.startswith('video/'): return 'video'
        if mime_type.startswith('audio/'): return 'audio'
        if mime_type in ['application/zip', 'application/gzip', 'application/x-tar', 'application/x-rar-compressed']: return 'archive' # General archive types

    # Fallback to extension-based categorization if MIME type is not definitive
    if ext in ['.txt', '.csv', '.log', '.py', '.js', '.html', '.css', '.md', '.json', '.xml', '.rtf', '.tex']: return 'text'
    if ext == '.pdf': return 'pdf'
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp', '.svg', '.ico']: return 'image'
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']: return 'video'
    if ext in ['.mp3', '.wav', '.ogg', '.aac', '.flac', '.m4a', '.wma']: return 'audio'
    if ext in ['.zip', '.gz', '.tar', '.rar', '.7z']: return 'archive'

    return 'other'

# --- Helper function to convert RGB to Hex ---
def rgb_to_hex(rgb_tuple):
    """Converts an RGB tuple (e.g., (255, 0, 0)) to a hex color string (e.g., '#ff0000')."""
    if isinstance(rgb_tuple, int): # Grayscale in some PIL modes
        return '#{0:02x}{0:02x}{0:02x}'.format(rgb_tuple)
    if isinstance(rgb_tuple, (tuple, list)) and len(rgb_tuple) >= 3:
        return '#{:02x}{:02x}{:02x}'.format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])
    return '#000000' # Default to black if format is unexpected

# --- Helper function to get video info using ffprobe ---
def get_video_info_ffprobe(filepath):
    """
    Extracts video information (duration, resolution, frame rate, codec)
    using the ffprobe command-line tool.
    Returns a dictionary with info or None, and an error message string or None.
    """
    if not os.path.exists(filepath):
        return None, "File not found for ffprobe analysis."

    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', # Select only video streams
        '-show_entries', 'stream=duration,width,height,avg_frame_rate,codec_name,codec_long_name,bit_rate,pix_fmt',
        '-of', 'json', filepath
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15) # Added timeout
        ffprobe_output = json.loads(result.stdout)
        video_info = {}
        if 'streams' in ffprobe_output and len(ffprobe_output['streams']) > 0:
            stream = ffprobe_output['streams'][0] # Use the first video stream

            if 'duration' in stream:
                try: video_info['duration_seconds'] = round(float(stream['duration']), 2)
                except (ValueError, TypeError): video_info['duration_seconds'] = stream.get('duration', 'N/A')
            else: video_info['duration_seconds'] = 'N/A'

            if 'width' in stream and 'height' in stream:
                video_info['resolution'] = f"{stream['width']}x{stream['height']}"
            else: video_info['resolution'] = 'N/A'

            if 'avg_frame_rate' in stream:
                try:
                    num_str, den_str = stream['avg_frame_rate'].split('/')
                    num, den = int(num_str), int(den_str)
                    video_info['frame_rate_fps'] = round(num / den, 2) if den != 0 else 'N/A'
                except (ValueError, ZeroDivisionError, AttributeError): video_info['frame_rate_fps'] = stream.get('avg_frame_rate', 'N/A')
            else: video_info['frame_rate_fps'] = 'N/A'

            video_info['codec_name'] = stream.get('codec_name', 'N/A')
            video_info['codec_long_name'] = stream.get('codec_long_name', 'N/A')
            if 'bit_rate' in stream:
                try: video_info['bit_rate_kbps'] = round(int(stream['bit_rate']) / 1000, 2)
                except (ValueError, TypeError): video_info['bit_rate_kbps'] = stream.get('bit_rate', 'N/A')
            else: video_info['bit_rate_kbps'] = 'N/A'
            video_info['pixel_format'] = stream.get('pix_fmt', 'N/A')

            return video_info, None
        else:
            return None, "No video stream found in the file."
    except FileNotFoundError:
        return None, "ffprobe not found. Please ensure FFmpeg (which includes ffprobe) is installed and in your system's PATH."
    except subprocess.CalledProcessError as e:
        return None, f"ffprobe execution error: {e.stderr.strip() if e.stderr else e.stdout.strip()}"
    except subprocess.TimeoutExpired:
        return None, "ffprobe command timed out. The file might be too large or corrupted."
    except json.JSONDecodeError:
        return None, "Failed to parse ffprobe output. The output was not valid JSON."
    except Exception as e:
        app.logger.error(f"Unexpected ffprobe error: {str(e)}", exc_info=True)
        return None, f"An unexpected error occurred with ffprobe: {str(e)}"


# --- NEW Endpoint to serve algorithm details ---
@app.route('/api/algorithm_details', methods=['GET'])
def get_algorithm_details():
    return jsonify(ALGORITHM_INFO_DETAILS)


# --- Compression Endpoint ---
@app.route('/compress', methods=['POST'])
def compress():
    try:
        if 'file' not in request.files or 'algorithm' not in request.form:
            return jsonify({'error': 'Missing file or algorithm in the request.'}), 400

        file = request.files['file']
        algorithm_name = request.form['algorithm']

        if not file or not file.filename:
            return jsonify({'error': 'No file selected or filename is empty.'}), 400
        if not algorithm_name:
            return jsonify({'error': 'No compression algorithm selected.'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        file_category = get_file_category(file_path)
        # For PDF, if using a text-based algorithm, treat its category as 'text' for algorithm compatibility check
        effective_category_for_algo = file_category
        if file_category == 'pdf' and algorithm_name in ['LZ77', 'Huffman', 'LZW', 'RLE']:
            effective_category_for_algo = 'text'


        if algorithm_name not in ALGORITHM_MAP:
            if os.path.exists(file_path): os.remove(file_path)
            return jsonify({'error': f'Unsupported or unavailable compression algorithm: {algorithm_name}. Check server logs for missing modules.'}), 400

        algo_config = ALGORITHM_MAP[algorithm_name]
        if effective_category_for_algo not in algo_config.get('file_types', []):
            if os.path.exists(file_path): os.remove(file_path)
            return jsonify({'error': f'{algorithm_name} is not supported for {file_category} files (effectively treated as {effective_category_for_algo} for this algorithm).'}), 400

        output_path = None
        if 'class' in algo_config:
            CompressorClass = algo_config['class']
            if not CompressorClass: # Check if the class itself is None due to import failure
                 if os.path.exists(file_path): os.remove(file_path)
                 return jsonify({'error': f'Compression class for {algorithm_name} is not available (module import likely failed).'}), 500

            compressor_instance = CompressorClass()
            method_to_call = getattr(compressor_instance, algo_config['method'])

            # Handle algorithms with specific parameters
            if algorithm_name == 'Dithering':
                 num_colors = int(request.form.get('num_colors', 16)) # Default to 16 if not provided
                 output_path = method_to_call(file_path, COMPRESSED_FOLDER, num_colors=num_colors)
            elif algorithm_name == 'DCT':
                 quality = int(request.form.get('quality', 50)) # Default to 50 if not provided
                 output_path = method_to_call(file_path, COMPRESSED_FOLDER, quality=quality)
            else:
                 output_path = method_to_call(file_path, COMPRESSED_FOLDER)

        elif 'function' in algo_config:
            compression_function = algo_config['function']
            if not compression_function: # Check if the function is None
                if os.path.exists(file_path): os.remove(file_path)
                return jsonify({'error': f'Compression function for {algorithm_name} is not available (module import likely failed).'}), 500

            base, ext = os.path.splitext(filename)
            # For video, the output extension is usually part of the function's responsibility or a standard like .mp4
            # We construct a descriptive output filename.
            output_filename_default = f"{base}_{algorithm_name.lower()}_compressed{ext if ext else '.bin'}" # Add a default extension if original had none
            output_path_for_video = os.path.join(COMPRESSED_FOLDER, output_filename_default)

            # Video compression functions might return the actual output path or just perform the operation
            # Assuming they take input_path and output_path as arguments
            returned_path = compression_function(file_path, output_path_for_video)
            output_path = returned_path if returned_path and os.path.exists(returned_path) else output_path_for_video


        if not output_path or not os.path.exists(output_path):
            app.logger.error(f"Compression failed for {filename} with {algorithm_name}. Expected output path: {output_path} was not found or not created.")
            if os.path.exists(file_path): os.remove(file_path)
            return jsonify({'error': f'Compression failed for {algorithm_name}. The output file was not created. Check server logs.'}), 500

        # Clean up original uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        return send_file(output_path, as_attachment=True, download_name=os.path.basename(output_path))

    except Exception as e:
        app.logger.error(f"Compression endpoint error: {str(e)}", exc_info=True)
        # Clean up uploaded file in case of any error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'An unexpected error occurred during compression: {str(e)}'}), 500


# --- Decompression Endpoint ---
@app.route('/decompress', methods=['POST'])
def decompress():
    try:
        if 'file' not in request.files or 'algorithm' not in request.form:
            return jsonify({'error': 'Missing file or algorithm for decompression.'}), 400

        file = request.files['file']
        algorithm_name = request.form['algorithm']

        if not file or not file.filename:
            return jsonify({'error': 'No file selected for decompression or filename is empty.'}), 400
        if not algorithm_name:
            return jsonify({'error': 'No decompression algorithm selected.'}), 400

        filename = secure_filename(file.filename)
        uploaded_compressed_path = os.path.join(UPLOAD_FOLDER, filename) # Save to uploads first
        file.save(uploaded_compressed_path)

        if algorithm_name not in DECOMPRESSION_ALGORITHM_MAP:
            if os.path.exists(uploaded_compressed_path): os.remove(uploaded_compressed_path)
            return jsonify({'error': f'Unsupported or unavailable decompression algorithm: {algorithm_name}. Check server logs for missing modules.'}), 400

        algo_config = DECOMPRESSION_ALGORITHM_MAP[algorithm_name]
        DecompressorClass = algo_config.get('class')

        if not DecompressorClass: # Check if the class itself is None due to import failure
            if os.path.exists(uploaded_compressed_path): os.remove(uploaded_compressed_path)
            return jsonify({'error': f'Decompression class for {algorithm_name} is not available (module import likely failed).'}), 500

        # Optional: File extension check (can be a hint, but algorithm selection is primary)
        # _, input_ext = os.path.splitext(filename)
        # expected_ext = algo_config.get('expected_extension')
        # if expected_ext and input_ext.lower() != expected_ext:
        #     # This could be a warning rather than an error, or handled by frontend logic
        #     app.logger.warning(f"File extension {input_ext} for {filename} doesn't match expected {expected_ext} for {algorithm_name}.")


        output_path = None
        if 'class' in algo_config:
            decompressor_instance = DecompressorClass()
            method_to_call = getattr(decompressor_instance, algo_config['method'])
            # The decompress method should handle naming the output file appropriately
            output_path = method_to_call(uploaded_compressed_path, DECOMPRESSED_FOLDER)
        else:
            # Should not happen if map is built correctly, but as a safeguard
            if os.path.exists(uploaded_compressed_path): os.remove(uploaded_compressed_path)
            return jsonify({'error': f'Decompression method for {algorithm_name} is misconfigured on the server.'}), 500


        if not output_path or not os.path.exists(output_path):
            app.logger.error(f"Decompression failed for {filename} with {algorithm_name}. Expected output path: {output_path} was not found or not created.")
            if os.path.exists(uploaded_compressed_path): os.remove(uploaded_compressed_path)
            return jsonify({'error': f'Decompression failed for {algorithm_name}. The output file was not created. Check server logs.'}), 500

        # Clean up original uploaded compressed file
        if os.path.exists(uploaded_compressed_path):
            os.remove(uploaded_compressed_path)

        return send_file(output_path, as_attachment=True, download_name=os.path.basename(output_path))

    except Exception as e:
        app.logger.error(f"Decompression endpoint error: {str(e)}", exc_info=True)
        if 'uploaded_compressed_path' in locals() and os.path.exists(uploaded_compressed_path):
            os.remove(uploaded_compressed_path)
        return jsonify({'error': f'An unexpected error occurred during decompression: {str(e)}'}), 500


# --- Analysis Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    file_path_local = None # Initialize to ensure it's always defined for cleanup
    try:
        if 'file' not in request.files or 'analysis_type' not in request.form:
            return jsonify({'error': 'Missing file or analysis_type in the request.'}), 400

        file = request.files['file']
        analysis_type = request.form['analysis_type']

        if not file or not file.filename:
            return jsonify({'error': 'No file selected or filename is empty.'}), 400

        filename = secure_filename(file.filename)
        file_path_local = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path_local)

        file_category = get_file_category(file_path_local)
        analysis_results = {
            'filename': filename,
            'analysis_type': analysis_type,
            'category': file_category,
            'file_size_bytes': os.path.getsize(file_path_local),
            'mime_type': mimetypes.guess_type(file_path_local)[0] or 'unknown'
        }

        if file_category == 'text' or file_category == 'pdf':
            text_content = ""
            try:
                if file_category == 'pdf':
                    if extract_text: # Check if pdfminer function is available
                        text_content = extract_text(file_path_local)
                    else:
                        analysis_results['text_analysis_error'] = "pdfminer.six library not installed or PDF is image-based/unreadable. Cannot extract text for PDF analysis."
                elif file_category == 'text':
                    # Try common encodings; add more if needed
                    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252']
                    for enc in encodings_to_try:
                        try:
                            with open(file_path_local, 'r', encoding=enc) as f:
                                text_content = f.read()
                            break # Successfully read
                        except UnicodeDecodeError:
                            if enc == encodings_to_try[-1]: # If last attempt fails
                                raise # Re-raise the last error
                    if not text_content:
                         analysis_results['text_analysis_error'] = "File is empty or could not be decoded with common text encodings."


                if text_content: # Proceed only if text was extracted
                    if analysis_type == 'Word Count':
                        words = re.findall(r'\b\w+\b', text_content) # More robust word splitting
                        analysis_results['word_count'] = len(words)
                    elif analysis_type == 'Character Count':
                        analysis_results['character_count'] = len(text_content) # Includes spaces and newlines
                        analysis_results['character_count_no_spaces'] = len(text_content.replace(" ", "").replace("\n", "").replace("\r", ""))
                    elif analysis_type == 'Keyword Density':
                        # Simple keyword extraction: lowercase, alphanumeric words of length >= 3
                        words = [word.lower() for word in re.findall(r'\b[a-z0-9]{3,}\b', text_content.lower())]
                        total_words_for_density = len(words)
                        if total_words_for_density > 0:
                            word_counts = collections.Counter(words)
                            # Get top 10 keywords or fewer if not that many unique words
                            top_keywords = [{'word': w, 'count': c, 'percentage': round((c / total_words_for_density) * 100, 2)}
                                            for w, c in word_counts.most_common(10)]
                            analysis_results['top_keywords'] = top_keywords
                            analysis_results['total_analyzed_words'] = total_words_for_density
                        else:
                            analysis_results['top_keywords'] = []
                            analysis_results['info'] = "Not enough suitable words found for keyword density analysis."
            except Exception as e:
                analysis_results['text_analysis_error'] = f"Text analysis error: {str(e)}"
                app.logger.error(f"Text analysis error for {filename}: {str(e)}", exc_info=True)

        elif file_category == 'image':
            if not Image: # Check if Pillow is available
                analysis_results['image_analysis_error'] = "Pillow (PIL) library not installed. Cannot perform image analysis."
            elif analysis_type in ['Dimension & Size', 'Color Palette', 'Metadata']:
                try:
                    with Image.open(file_path_local) as img:
                        if analysis_type == 'Dimension & Size':
                            analysis_results['dimensions'] = f"{img.width}x{img.height}"
                            analysis_results['format'] = img.format
                            analysis_results['mode'] = img.mode # e.g., RGB, RGBA, L (grayscale)
                            analysis_results['info'] = img.info.get('comment', None) or img.info.get('description', None) # Common info fields

                        elif analysis_type == 'Color Palette':
                            # Resize for faster processing, convert to RGB for consistent color handling
                            img_for_palette = img.convert('RGB').resize((100, 100), Image.Resampling.NEAREST) # Use NEAREST for speed
                            raw_colors = img_for_palette.getcolors(maxcolors=img_for_palette.width * img_for_palette.height) # Get all colors
                            if raw_colors:
                                # Sort by count (descending) to get dominant colors
                                sorted_colors = sorted(raw_colors, key=lambda x: x[0], reverse=True)
                                palette = []
                                total_pixels = sum(c[0] for c in sorted_colors)
                                for count, pixel_value in sorted_colors[:10]: # Top 10 dominant colors
                                    palette.append({
                                        'hex': rgb_to_hex(pixel_value),
                                        'rgb': pixel_value,
                                        'count': count,
                                        'percentage': round((count / total_pixels) * 100, 2) if total_pixels else 0
                                    })
                                analysis_results['color_palette'] = palette
                                if ImageStat: # If ImageStat is available for more stats
                                    stats = ImageStat.Stat(img.convert('RGB'))
                                    analysis_results['average_color_rgb'] = tuple(map(int, stats.mean))
                                    analysis_results['average_color_hex'] = rgb_to_hex(tuple(map(int, stats.mean)))
                            else:
                                analysis_results['color_palette'] = []
                                analysis_results['info'] = "Could not extract color palette (e.g., image might be empty or have too many unique colors for simple getcolors)."

                        elif analysis_type == 'Metadata':
                            metadata = {}
                            # Basic image info
                            if hasattr(img, 'info'):
                                for k, v_raw in img.info.items():
                                    # Attempt to decode bytes, otherwise use string representation
                                    try:
                                        v = v_raw.decode('utf-8', errors='replace') if isinstance(v_raw, bytes) else str(v_raw)
                                        metadata[str(k)] = v[:200] + '...' if len(v) > 200 else v # Truncate long values
                                    except Exception:
                                        metadata[str(k)] = str(v_raw)[:200] + '...' if len(str(v_raw)) > 200 else str(v_raw)

                            # EXIF data
                            exif_data_raw = img.getexif()
                            if exif_data_raw:
                                exif_data_processed = {}
                                for tag_id, value_raw in exif_data_raw.items():
                                    tag_name = EXIF_TAGS.get(tag_id, tag_id) # Get human-readable name if available
                                    # Decode bytes if necessary, handle specific types carefully
                                    if isinstance(value_raw, bytes):
                                        try:
                                            value = value_raw.decode('utf-8', errors='replace').strip('\x00')
                                        except UnicodeDecodeError:
                                            value = f"Bytes (len {len(value_raw)})" # Fallback for non-decodable bytes
                                    elif isinstance(value_raw, tuple) and all(isinstance(item, int) for item in value_raw):
                                        value = ", ".join(map(str,value_raw)) # e.g. for GPS coordinates
                                    else:
                                        value = str(value_raw)
                                    exif_data_processed[str(tag_name)] = value[:200] + '...' if len(value) > 200 else value # Truncate
                                if exif_data_processed:
                                     metadata['EXIF'] = exif_data_processed

                            analysis_results['metadata'] = metadata if metadata else "No readily extractable metadata found."
                except FileNotFoundError:
                    analysis_results['image_analysis_error'] = "Image file not found during analysis (should not happen if upload succeeded)."
                except UnboundLocalError as ule: # Catch if Image was None
                    analysis_results['image_analysis_error'] = f"Pillow (PIL) might not be properly initialized: {str(ule)}"
                except Exception as e:
                    analysis_results['image_analysis_error'] = f"Image analysis error: {str(e)}"
                    app.logger.error(f"Image analysis error for {filename}: {str(e)}", exc_info=True)
            else: # analysis_type not supported for image or Pillow issue
                analysis_results['info'] = f"Analysis type '{analysis_type}' is not available for images, or Pillow (PIL) library is missing/failed to load."

        elif file_category == 'video':
            video_info, ffprobe_error = get_video_info_ffprobe(file_path_local)
            if ffprobe_error:
                analysis_results['video_analysis_error'] = ffprobe_error
            elif video_info:
                # Populate based on what ffprobe returned and the requested analysis type
                if analysis_type == 'Duration & Codec':
                    analysis_results.update({
                        'duration_seconds': video_info.get('duration_seconds'),
                        'codec_name': video_info.get('codec_name'),
                        'codec_long_name': video_info.get('codec_long_name'),
                        'bit_rate_kbps': video_info.get('bit_rate_kbps')
                    })
                elif analysis_type == 'Resolution':
                    analysis_results['resolution'] = video_info.get('resolution')
                    analysis_results['pixel_format'] = video_info.get('pixel_format')
                elif analysis_type == 'Frame Rate':
                    analysis_results['frame_rate_fps'] = video_info.get('frame_rate_fps')
                # Always add all retrieved info for completeness, regardless of analysis_type
                analysis_results['full_video_stream_info'] = video_info
            else: # Should be caught by ffprobe_error, but as a fallback
                analysis_results['video_analysis_error'] = "Could not retrieve video information using ffprobe."

        elif file_category == 'audio':
            # Basic audio analysis could be added here using ffprobe similarly to video
            # For now, placeholder:
            audio_info, ffprobe_error = get_audio_info_ffprobe(file_path_local) # You'd need to implement this function
            if ffprobe_error:
                analysis_results['audio_analysis_error'] = ffprobe_error
            elif audio_info:
                 analysis_results.update(audio_info) # Add all fields from audio_info
            else:
                analysis_results['audio_analysis_error'] = "Audio analysis (e.g., duration, bitrate, codec via ffprobe) not yet fully implemented or ffprobe failed."


        # Fallback for 'other' or 'compressed' categories if no specific analysis is done
        elif file_category in ['other', 'compressed', 'archive']:
            analysis_results['info'] = f"Basic file properties displayed. No specific content analysis for '{file_category}' category with type '{analysis_type}'."


        return jsonify(analysis_results), 200

    except Exception as e:
        app.logger.error(f"Analysis endpoint error: {str(e)}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred during analysis: {str(e)}'}), 500
    finally:
        # Ensure uploaded file is cleaned up
        if file_path_local and os.path.exists(file_path_local):
            try:
                os.remove(file_path_local)
            except Exception as e_remove:
                app.logger.error(f"Error removing uploaded file {file_path_local} after analysis: {str(e_remove)}")


def get_audio_info_ffprobe(filepath):
    """
    Extracts audio information (duration, codec, sample rate, channels, bit rate)
    using the ffprobe command-line tool.
    Returns a dictionary with info or None, and an error message string or None.
    """
    if not os.path.exists(filepath):
        return None, "File not found for ffprobe audio analysis."

    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0', # Select only audio streams
        '-show_entries', 'stream=duration,codec_name,codec_long_name,sample_rate,channels,channel_layout,bit_rate',
        '-of', 'json', filepath
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
        ffprobe_output = json.loads(result.stdout)
        audio_info = {}
        if 'streams' in ffprobe_output and len(ffprobe_output['streams']) > 0:
            stream = ffprobe_output['streams'][0] # Use the first audio stream

            if 'duration' in stream:
                try: audio_info['duration_seconds'] = round(float(stream['duration']), 2)
                except (ValueError, TypeError): audio_info['duration_seconds'] = stream.get('duration', 'N/A')
            else: audio_info['duration_seconds'] = 'N/A'

            audio_info['codec_name'] = stream.get('codec_name', 'N/A')
            audio_info['codec_long_name'] = stream.get('codec_long_name', 'N/A')
            audio_info['sample_rate_hz'] = stream.get('sample_rate', 'N/A')
            audio_info['channels'] = stream.get('channels', 'N/A')
            audio_info['channel_layout'] = stream.get('channel_layout', 'N/A')

            if 'bit_rate' in stream:
                try: audio_info['bit_rate_kbps'] = round(int(stream['bit_rate']) / 1000, 2)
                except (ValueError, TypeError): audio_info['bit_rate_kbps'] = stream.get('bit_rate', 'N/A')
            else: audio_info['bit_rate_kbps'] = 'N/A'

            return audio_info, None
        else:
            return None, "No audio stream found in the file."
    except FileNotFoundError:
        return None, "ffprobe not found. Please ensure FFmpeg is installed and in your system's PATH."
    except subprocess.CalledProcessError as e:
        return None, f"ffprobe execution error (audio): {e.stderr.strip() if e.stderr else e.stdout.strip()}"
    except subprocess.TimeoutExpired:
        return None, "ffprobe command timed out for audio analysis. The file might be too large or corrupted."
    except json.JSONDecodeError:
        return None, "Failed to parse ffprobe output for audio. The output was not valid JSON."
    except Exception as e:
        app.logger.error(f"Unexpected ffprobe error (audio): {str(e)}", exc_info=True)
        return None, f"An unexpected error occurred with ffprobe for audio: {str(e)}"


# --- Archiving Endpoint ---
@app.route('/archive', methods=['POST'])
def archive_files_route(): # Renamed to avoid conflict with 'archive' module if it were a function
    if not file_archiver:
        return jsonify({'error': 'Archiving service is not available (FileArchiver module failed to load).'}), 503

    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided for archiving in files[] field.'}), 400

        files = request.files.getlist('files[]')
        if not files or all(not f.filename for f in files):
             return jsonify({'error': 'No files selected or all selected files are empty.'}), 400

        archive_name_req = request.form.get('archive_name', 'archive.zip').strip()
        if not archive_name_req: # Ensure archive_name is not empty after stripping
            archive_name = 'archive.zip'
        else:
            archive_name = secure_filename(archive_name_req) # Secure the filename

        if not archive_name.lower().endswith('.zip'):
            archive_name += '.zip'

        # The FileArchiver class should handle saving files to UPLOAD_FOLDER and then creating the archive
        archive_path = file_archiver.archive_files(files, archive_name)

        if not archive_path or not os.path.exists(archive_path):
            app.logger.error(f"Archiving failed. Expected archive path: {archive_path} was not found or not created.")
            return jsonify({'error': 'Archiving failed. The archive file was not created. Check server logs.'}), 500

        return send_file(archive_path, as_attachment=True, download_name=os.path.basename(archive_path))

    except ValueError as ve: # Specific errors from FileArchiver logic
        app.logger.error(f"Archiving ValueError: {str(ve)}")
        return jsonify({'error': f'Archiving failed: {str(ve)}'}), 400
    except RuntimeError as re_err:
        app.logger.error(f"Archiving RuntimeError: {str(re_err)}", exc_info=True)
        return jsonify({'error': f'Archiving failed due to a runtime issue: {str(re_err)}'}), 500
    except Exception as e:
        app.logger.error(f"Archiving endpoint error: {str(e)}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred during archiving: {str(e)}'}), 500
    finally:
        # FileArchiver's archive_files method should handle cleanup of individual uploaded files
        # If archive_path was created, it will be sent and then implicitly cleaned by browser download or if send_file fails.
        # If archive creation failed before send_file, ensure no partial archive is left, if applicable.
        # This depends on FileArchiver's implementation.
        pass


# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health():
    # Check availability of key dependencies
    dependencies_status = {
        "Pillow (Image Analysis)": "Available" if Image else "Not Available",
        "pdfminer.six (PDF Text Extraction)": "Available" if extract_text else "Not Available",
        "FFmpeg/ffprobe (Video/Audio Analysis)": "Assumed Available (checked at runtime by ffprobe calls)",
        "LZ77_Module": "Available" if LZ77 else "Not Available",
        "Huffman_Module": "Available" if HuffmanCoding else "Not Available",
        "LZW_Module": "Available" if LZWCompression else "Not Available",
        "RLE_Module": "Available" if RLECompression else "Not Available",
        "DCT_Module": "Available" if DCTCompression else "Not Available",
        "Dithering_Module": "Available" if DitheringCompression else "Not Available",
        "Deflate_Module": "Available" if DeflateCoding else "Not Available",
        "FileArchiver_Module": "Available" if FileArchiver else "Not Available"
    }
    # Check if all algorithm classes in ALGORITHM_MAP are non-None
    missing_compression_algos = [algo for algo, conf in ALGORITHM_MAP.items() if not (conf.get('class') or conf.get('function'))]
    if missing_compression_algos:
        dependencies_status["Compression Algorithms Status"] = f"Missing modules for: {', '.join(missing_compression_algos)}"
    else:
        dependencies_status["Compression Algorithms Status"] = "All configured compression algorithms appear available."


    return jsonify({
        'status': 'Backend is running.',
        'upload_folder': UPLOAD_FOLDER,
        'compressed_folder': COMPRESSED_FOLDER,
        'decompressed_folder': DECOMPRESSED_FOLDER,
        'archive_folder': ARCHIVE_FOLDER,
        'dependencies': dependencies_status
        }), 200

# --- Main execution block ---
if __name__ == '__main__':
    print("--- File Management and Analysis System Backend ---")
    print(f"Serving from base directory: {BASE_DIR}")
    print(f"Uploads folder: {UPLOAD_FOLDER}")
    print(f"Compressed files folder: {COMPRESSED_FOLDER}")
    print(f"Decompressed files folder: {DECOMPRESSED_FOLDER}")
    print(f"Archives folder: {ARCHIVE_FOLDER}")
    print("\n--- Required Libraries & Tools (Please ensure they are installed) ---")
    print("- Flask, Flask-Cors, Werkzeug: pip install Flask Flask-Cors Werkzeug")
    print("- Pillow (for image analysis): pip install Pillow")
    print("- pdfminer.six (for PDF text extraction): pip install pdfminer.six")
    print("- FFmpeg (includes ffprobe, for video/audio analysis): Install from ffmpeg.org/download.html and ensure it's in your system PATH.")
    print("\n--- Custom Compression Modules (Ensure these .py files are in the same directory or Python path) ---")
    print("  LZ77.py, HUFFMANN.py (or HUFFMAN.py), lzw.py, RLE.py, COLORLOSSY.py, DEFLATE.py, ")
    print("  LOSSLESS.py, LOSSY.py, DCTCompression.py, DitheringCompression.py, HEVC_AND_AVC.py, archive.py")
    print("\n--- Starting Flask Development Server ---")
    print("Access the frontend application, which should point to this backend (default: http://localhost:5000).")
    print("Health check available at: http://localhost:5000/health")
    print("Algorithm details available at: http://localhost:5000/api/algorithm_details")


    # For development, debug=True is fine. For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(debug=True, host='0.0.0.0', port=5000)
