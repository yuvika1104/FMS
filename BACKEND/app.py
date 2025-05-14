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
from LZ77 import LZ77
from HUFFMANN import HuffmanCoding
from COLORLOSSY import ColorLossyImageCompressor
from DEFLATE import DeflateCoding
from LOSSLESS import LosslessImageCompressor
from LOSSY import LossyImageCompressor
# Assuming HEVC_AND_AVC module exists and handles video compression
import HEVC_AND_AVC as VideoCompressor

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

# --- moviepy removed ---
# try:
#     from moviepy.editor import VideoFileClip  # For video analysis
# except ImportError:
#     VideoFileClip = None  # moviepy not installed


app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Define folders for uploads and compressed files
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
COMPRESSED_FOLDER = os.path.join(BASE_DIR, 'compressed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

# --- Compression Algorithm Mapping ---
ALGORITHM_MAP = {
    'LZ77': {'class': LZ77, 'method': 'compress', 'file_types': ['text']},
    'Huffman': {'class': HuffmanCoding, 'method': 'compress', 'file_types': ['text']},
    'ColorLossy': {'class': ColorLossyImageCompressor, 'method': 'compress', 'file_types': ['image']},
    'Deflate': {'class': DeflateCoding, 'method': 'compress', 'file_types': ['image']},
    'Lossless': {'class': LosslessImageCompressor, 'method': 'compress', 'file_types': ['image']},
    'Lossy': {'class': LossyImageCompressor, 'method': 'compress', 'file_types': ['image']},
    # Assuming VideoCompressor methods handle the actual compression
    'HEVC': {'function': VideoCompressor.compress_mp4_hevc, 'file_types': ['video']},
    'AVC': {'function': VideoCompressor.compress_mp4_avc, 'file_types': ['video']},
}

# --- Helper function to determine file category ---
def get_file_category(filepath):
    """Determines the general category of a file based on its MIME type."""
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type:
        if mime_type.startswith('text/'):
            return 'text'
        if mime_type.startswith('image/'):
            return 'image'
        if mime_type.startswith('video/'):
            return 'video'
        if mime_type.startswith('audio/'):
            return 'audio'
    # Fallback for unknown or common extensions if MIME type is not specific enough
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.txt', '.csv', '.log', '.py', '.js', '.html', '.css', '.md']:
        return 'text'
    if ext in ['.pdf']:  # PDF can be text but often handled separately
        return 'text'  # or 'pdf' if you want specific handling
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
    Requires FFmpeg to be installed and in the system's PATH.
    """
    if not os.path.exists(filepath):
        return None, "File not found."

    # Command to run ffprobe
    # -v error: Suppress verbose output, only show errors
    # -select_streams v: Only show video streams
    # -show_entries stream=duration,width,height,avg_frame_rate,codec_name: Specify fields to show
    # -of json: Output format as JSON
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0', # Select the first video stream
        '-show_entries', 'stream=duration,width,height,avg_frame_rate,codec_name',
        '-of', 'json',
        filepath
    ]

    try:
        # Run the ffprobe command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        ffprobe_output = json.loads(result.stdout)

        # Parse the JSON output
        video_info = {}
        if 'streams' in ffprobe_output and len(ffprobe_output['streams']) > 0:
            stream = ffprobe_output['streams'][0] # Get info from the first video stream

            # Duration
            if 'duration' in stream:
                try:
                    video_info['duration_seconds'] = round(float(stream['duration']), 2)
                except (ValueError, TypeError):
                     video_info['duration_seconds'] = 'N/A'

            # Resolution
            if 'width' in stream and 'height' in stream:
                video_info['resolution'] = f"{stream['width']}x{stream['height']}"
            else:
                video_info['resolution'] = 'N/A'

            # Frame Rate (represented as a fraction, e.g., "30000/1001")
            if 'avg_frame_rate' in stream:
                 try:
                     num, den = map(int, stream['avg_frame_rate'].split('/'))
                     video_info['frame_rate'] = round(num / den, 2) if den != 0 else 'N/A'
                 except (ValueError, AttributeError):
                     video_info['frame_rate'] = 'N/A'
            else:
                 video_info['frame_rate'] = 'N/A'


            # Codec Name
            if 'codec_name' in stream:
                video_info['codec_info'] = stream['codec_name']
            else:
                 video_info['codec_info'] = 'N/A'

            return video_info, None # Return info and no error

        else:
             return None, "No video stream found in the file."


    except FileNotFoundError:
        return None, "ffprobe command not found. Is FFmpeg installed and in your system's PATH?"
    except subprocess.CalledProcessError as e:
        # ffprobe returned a non-zero exit code (an error occurred)
        return None, f"ffprobe error: {e.stderr.strip()}"
    except json.JSONDecodeError:
         return None, "Failed to parse ffprobe output (invalid JSON)."
    except Exception as e:
        # Catch any other unexpected errors
        return None, f"An unexpected error occurred during ffprobe execution: {str(e)}"


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

        if not algorithm_name:
            return jsonify({'error': 'No algorithm selected'}), 400

        filename = secure_filename(file.filename)  # Secure the filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        file_category = get_file_category(file_path)

        if algorithm_name not in ALGORITHM_MAP:
            return jsonify({'error': f'Unsupported algorithm: {algorithm_name}'}), 400

        algo_config = ALGORITHM_MAP[algorithm_name]
        if file_category not in algo_config['file_types']:
            return jsonify({'error': f'{algorithm_name} is not supported for {file_category} files'}), 400

        output_path = None

        if 'class' in algo_config:
            compressor_instance = algo_config['class']()
            # Assuming compress method returns the output file path
            output_path = getattr(compressor_instance, algo_config['method'])(file_path, COMPRESSED_FOLDER)
        elif 'function' in algo_config:  # For video and other function-based compressors
            base, ext = os.path.splitext(filename)
            output_filename = f"{base}_{algorithm_name.lower()}_compressed{ext}"
            output_path_for_video = os.path.join(COMPRESSED_FOLDER, output_filename)
            # Assuming the function handles the compression and saves to output_path_for_video
            algo_config['function'](file_path, output_path_for_video)
            output_path = output_path_for_video

        if not output_path or not os.path.exists(output_path):
            app.logger.error(f"Compression failed. Expected output_path: {output_path}")
            # Log contents of compressed folder for debugging
            app.logger.error(f"Files in COMPRESSED_FOLDER: {os.listdir(COMPRESSED_FOLDER)}")
            return jsonify({'error': 'Compression failed or output file not found. Check server logs.'}), 500

        # Send the compressed file back
        return send_file(output_path, as_attachment=True, download_name=os.path.basename(output_path))

    except Exception as e:
        # Log the full traceback for better debugging
        app.logger.error(f"Compression error: {str(e)}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred during compression: {str(e)}'}), 500

# --- Analysis Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files or 'analysis_type' not in request.form:
            return jsonify({'error': 'Missing file or analysis_type'}), 400

        file = request.files['file']
        analysis_type = request.form['analysis_type']

        if not file or not file.filename:
            return jsonify({'error': 'No file selected or filename is empty'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        file_category = get_file_category(file_path)
        analysis_results = {'filename': filename, 'analysis_type': analysis_type, 'category': file_category}

        # Add basic file info available for all types
        analysis_results['file_size_bytes'] = os.path.getsize(file_path)
        analysis_results['mime_type'] = mimetypes.guess_type(file_path)[0] or 'unknown'


        if file_category == 'text':
            try:
                text_content = ""
                # Use pdfminer.six for PDF, standard open for others
                if filename.lower().endswith('.pdf') and extract_text:
                    text_content = extract_text(file_path)
                else:
                    # Read with UTF-8 encoding, ignoring errors for robustness
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()

                if analysis_type == 'Word Count':
                    words = text_content.split()
                    analysis_results['word_count'] = len(words)
                elif analysis_type == 'Character Count':
                    analysis_results['character_count'] = len(text_content)
                elif analysis_type == 'Keyword Density':
                    # Simple keyword density: filter non-alpha words, convert to lower, count
                    words = [word.lower() for word in text_content.split() if word.isalpha() and len(word) > 2]
                    total_words_for_density = len(words)
                    if total_words_for_density > 0:
                        word_counts = collections.Counter(words)
                        top_keywords = []
                        # Get top 10 most common keywords
                        for word, count in word_counts.most_common(10):
                            top_keywords.append({
                                'word': word,
                                'count': count,
                                'percentage': round((count / total_words_for_density) * 100, 2)
                            })
                        analysis_results['top_keywords'] = top_keywords
                        analysis_results['total_analyzed_words'] = total_words_for_density
                    else:
                        analysis_results['top_keywords'] = []
                        analysis_results['info'] = "Not enough words for keyword density analysis."

            except Exception as e:
                # Report specific text analysis error
                analysis_results['text_analysis_error'] = f"Error during text analysis: {str(e)}"
                app.logger.error(f"Text analysis error for {filename}: {str(e)}", exc_info=True)

        elif file_category == 'image':
            # Check if Pillow is installed and analysis type is relevant for images
            if Image and analysis_type in ['Dimension & Size', 'Color Palette', 'Metadata']:
                try:
                    with Image.open(file_path) as img:
                        if analysis_type == 'Dimension & Size':
                            analysis_results['dimensions'] = f"{img.width}x{img.height}"
                            analysis_results['format'] = img.format
                            analysis_results['mode'] = img.mode
                        elif analysis_type == 'Color Palette':
                            # Convert to RGB for consistent color processing
                            img_for_palette = img.convert('RGB')
                            # Resize for performance on large images
                            if img_for_palette.width > 500 or img_for_palette.height > 500:
                                img_for_palette.thumbnail((300, 300)) # Keep aspect ratio
                            # Get colors; maxcolors=None gets all, but can be slow. Use a reasonable limit.
                            # Using width*height can still be too many for complex images.
                            # Let's try getting up to 256 colors first, then fall back if needed.
                            raw_colors = img_for_palette.getcolors(maxcolors=256) # Try 256 first
                            if raw_colors is None: # If more than 256 colors, getcolors returns None
                                raw_colors = img_for_palette.getcolors(maxcolors=img_for_palette.width * img_for_palette.height) # Fallback to all

                            if raw_colors:
                                # Sort colors by count (most frequent first)
                                sorted_colors = sorted(raw_colors, key=lambda x: x[0], reverse=True)
                                palette = []
                                total_pixels = sum(c[0] for c in sorted_colors) # Sum of counts
                                # Take top 10 dominant colors
                                for count, pixel_value in sorted_colors[:10]:
                                    hex_color = rgb_to_hex(pixel_value)
                                    palette.append({
                                        'hex': hex_color,
                                        'count': count,
                                        'percentage': round((count / total_pixels) * 100, 2) if total_pixels > 0 else 0
                                    })
                                analysis_results['color_palette'] = palette
                                analysis_results['image_mode'] = img.mode # Original mode
                            else:
                                analysis_results['color_palette'] = []
                                analysis_results['info'] = "Could not extract detailed color palette (e.g., too many unique colors or complex image)."
                        elif analysis_type == 'Metadata':
                            metadata = {}
                            # Get basic info dictionary
                            if hasattr(img, 'info'):
                                for k, v in img.info.items():
                                    # Attempt to decode bytes to string
                                    if isinstance(v, bytes):
                                        try:
                                            metadata[k] = v.decode('utf-8', errors='replace')
                                        except:
                                            metadata[k] = str(v) # Fallback to string representation
                                    else:
                                        metadata[k] = v
                            # Get EXIF data specifically
                            exif_data = img.getexif()
                            if exif_data:
                                # Convert EXIF tags to human-readable names if possible (requires PIL.ExifTags)
                                # For simplicity, keeping raw tag IDs for now or converting known ones
                                # You might need: from PIL.ExifTags import TAGS
                                # metadata['EXIF'] = {TAGS.get(k, k): v for k, v in exif_data.items() if isinstance(v, (str, int, float, bytes))}
                                # Keeping raw items that are basic types
                                metadata['EXIF'] = {k: v for k, v in exif_data.items() if isinstance(v, (str, int, float, bytes))}

                            analysis_results['metadata'] = metadata if metadata else "No standard metadata found."
                except Exception as e:
                    # Report specific image analysis error
                    analysis_results['image_analysis_error'] = f"Error during image analysis: {str(e)}"
                    app.logger.error(f"Image analysis error for {filename}: {str(e)}", exc_info=True)
            # Message if Pillow is not installed
            elif not Image and analysis_type in ['Dimension & Size', 'Color Palette', 'Metadata']:
                analysis_results['image_analysis_error'] = "Pillow (PIL) library not installed for image analysis."
            # Message if analysis type is not supported for images
            else:
                analysis_results['info'] = f"Selected image analysis ('{analysis_type}') not available or Pillow not installed."


        elif file_category == 'video':
            # Use ffprobe for video analysis
            video_info, error = get_video_info_ffprobe(file_path)

            if error:
                analysis_results['video_analysis_error'] = f"Video analysis failed: {error}"
                app.logger.error(f"Video analysis error for {filename}: {error}")
            elif video_info:
                # Map ffprobe results to analysis_results structure
                if analysis_type == 'Duration & Codec':
                    analysis_results['duration_seconds'] = video_info.get('duration_seconds', 'N/A')
                    analysis_results['codec_info'] = video_info.get('codec_info', 'N/A')
                elif analysis_type == 'Resolution':
                    analysis_results['resolution'] = video_info.get('resolution', 'N/A')
                elif analysis_type == 'Frame Rate':
                    analysis_results['frame_rate'] = video_info.get('frame_rate', 'N/A')
                else:
                     # Should not happen if frontend analysis types match backend logic
                     analysis_results['info'] = f"Selected video analysis ('{analysis_type}') not supported by backend implementation."
            else:
                 # This case should ideally be covered by the 'error' check above, but as a fallback:
                 analysis_results['video_analysis_error'] = "Could not retrieve video information."


        elif file_category == 'audio':
             # Add check for necessary audio analysis libraries if implemented
            if analysis_type in ['Duration & Bitrate', 'Format Info']:
                # Placeholder for future audio analysis implementation
                analysis_results['audio_info'] = "Audio duration and bitrate analysis not yet implemented."
            else:
                analysis_results['info'] = f"Selected audio analysis ('{analysis_type}') not available."

        else:
            # Default analysis for 'other' category or if specific analysis fails
            if analysis_type not in ['File Size', 'MIME Type']: # These are already included
                 analysis_results['info'] = f"No specific analysis for '{analysis_type}' on this file type."


        # Return the analysis results as JSON
        return jsonify(analysis_results), 200

    except Exception as e:
        # Catch any unexpected errors during the overall analysis process
        app.logger.error(f"Analysis endpoint error: {str(e)}", exc_info=True)
        # Return a generic error message for unexpected issues
        return jsonify({'error': f'An unexpected error occurred during analysis endpoint processing: {str(e)}'}), 500

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health():
    """Simple health check to see if the backend is running."""
    return jsonify({'status': 'Backend is running and healthy!'}), 200

# --- Main execution block ---
if __name__ == '__main__':
    # List required libraries and tools for the user
    print("Required libraries and tools:")
    print("- Pillow: pip install Pillow (for image analysis)")
    print("- pdfminer.six: pip install pdfminer.six (for PDF text extraction)")
    print("- FFmpeg (includes ffprobe): Install from https://ffmpeg.org/download.html (for video analysis)")
    print("  Ensure 'ffprobe' is in your system's PATH.")
    print("- collections: Built-in (for keyword density)")
    print("- Flask: pip install Flask Flask-Cors werkzeug")
    # Note: Compression libraries (LZ77, HUFFMANN, etc.) are assumed to be local modules.
    # Note: HEVC_AND_AVC is assumed to be a local module for video compression.

    # Run the Flask application
    # debug=True provides detailed error pages and auto-reloads on code changes
    # host='0.0.0.0' makes the server accessible externally (useful in some environments)
    # port=5000 is the standard port for this application
    app.run(debug=True, host='0.0.0.0', port=5000)
