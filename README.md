# File Compression and Analysis System

## Overview
This project is a web-based file compression and analysis system built with Python and Flask. It provides a robust backend for compressing, decompressing, archiving, and analyzing various file types, including text, images, videos, and PDFs. The system supports multiple compression algorithms, both lossless and lossy, and includes file analysis features such as text extraction, image metadata retrieval, and video/audio stream information.

The system is designed to be extensible, with a modular architecture that allows easy integration of new compression algorithms or analysis tools. It includes a Flask-based REST API for integration with a frontend interface (not fully provided in the codebase but referenced as frontend.html).

## Features

### Compression Algorithms
**Lossless Compression:**
- **LZ77**: Dictionary-based compression for text and PDFs with repetitive patterns.
- **Huffman Coding**: Frequency-based encoding for text and PDFs.
- **LZW**: Dictionary-based compression for text and PDFs, used in GIF and TIFF formats.
- **RLE (Run-Length Encoding)**: Effective for data with long runs of identical bytes.
- **Deflate**: Combines LZ77 and Huffman coding, used for images (PNG-like).
- **Lossless Image Compression**: Preserves image quality using PNG or WebP formats.

**Lossy Compression:**
- **DCT (Discrete Cosine Transform)**: JPEG-like compression for grayscale images.
- **Dithering**: Reduces color palette with Floyd-Steinberg dithering for images.
- **Color Lossy**: Quantizes RGB channels for color images.
- **Lossy Image Compression**: Quantizes grayscale images for smaller file sizes.
- **HEVC (H.265)**: High-efficiency video compression.
- **AVC (H.264)**: Widely supported video compression.

### File Archiving
- **ZIP Archiving**: Combines multiple files into a ZIP archive using Deflate compression.

### File Analysis
**Text/PDF:**
- Word count, character count, and keyword density analysis.

**Images:**
- Dimensions, color palette extraction, and metadata (EXIF, format, mode).

**Video/Audio:**
- Duration, resolution, frame rate, codec, bit rate, and pixel format (via ffprobe).

### API Endpoints
- `/compress`: Compress a file using a specified algorithm.
- `/decompress`: Decompress a file using the corresponding algorithm.
- `/archive`: Create a ZIP archive from multiple files.
- `/analyze`: Perform content analysis based on file type and analysis type.
- `/api/algorithm_details`: Retrieve details about available algorithms (description, pros, cons).
- `/health`: Check server status and dependency availability.

## System Requirements

### Software
- Python 3.8+
- FFmpeg (with ffprobe) for video/audio analysis
- Operating System: Linux, macOS, or Windows

### Python Dependencies
Install the required Python packages using:
```
pip install flask flask-cors werkzeug pillow pdfminer.six opencv-python numpy
```

### External Tools
- **FFmpeg**: Install from ffmpeg.org and ensure ffprobe is in your system PATH.

## Installation
1. **Clone the Repository:**
   ```
   git clone <repository-url>
   cd file-compression-system
   ```

2. **Set Up a Virtual Environment (recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies:**
   ```
   pip install -r requirements.txt
   ```
   Create a `requirements.txt` with:
   ```
   flask
   flask-cors
   werkzeug
   pillow
   pdfminer.six
   opencv-python
   numpy
   ```

4. **Install FFmpeg:**
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from FFmpeg website and add to PATH.

5. **Verify Directory Structure:** Ensure the following files are in the project root:
   - `app.py`
   - `archive.py`
   - `COLORLOSSY.py`
   - `DCTCompression.py`
   - `DEFLATE.py`
   - `DitheringCompression.py`
   - `HEVC_AND_AVC.py`
   - `HUFFMANN.py`
   - `LOSSLESS.py`
   - `LOSSY.py`
   - `LZ77.py`
   - `RLE.py`
   - `lzw.py`
   - `frontend.html` (if implementing the frontend)

6. **Run the Flask Server:**
   ```
   python app.py
   ```
   The server runs on `http://localhost:5000` by default.

## Usage

### Running the Backend
1. Start the Flask server:
   ```
   python app.py
   ```

2. Check server health:
   ```
   curl http://localhost:5000/health
   ```

### API Usage
Use tools like `curl`, Postman, or a custom frontend to interact with the API. Example requests:

- **Compress a File:**
   ```
   curl -X POST -F "file=@image.jpg" -F "algorithm=DCT" -F "quality=50" http://localhost:5000/compress --output compressed_image.dctz
   ```

- **Decompress a File:**
   ```
   curl -X POST -F "file=@compressed_image.dctz" -F "algorithm=DCT" http://localhost:5000/decompress --output decompressed_image.png
   ```

- **Archive Files:**
   ```
   curl -X POST -F "files[]=@file1.txt" -F "files[]=@file2.jpg" -F "archive_name=my_archive.zip" http://localhost:5000/archive --output my_archive.zip
   ```

- **Analyze a File:**
   ```
   curl -X POST -F "file=@document.pdf" -F "analysis_type=Word Count" http://localhost:5000/analyze
   ```

### Frontend Integration
The `frontend.html` file is referenced but not fully provided. To create a frontend:
1. Develop an HTML/JavaScript interface using a framework like React or vanilla JavaScript.
2. Use the `/api/algorithm_details` endpoint to populate algorithm selection options.
3. Implement file upload forms to interact with `/compress`, `/decompress`, `/archive`, and `/analyze` endpoints.

## System Architecture

### Directory Structure
```
file-compression-system/
├── app.py                # Flask backend and API endpoints
├── archive.py            # ZIP archiving functionality
├── COLORLOSSY.py         # Lossy color image compression
├── DCTCompression.py     # DCT-based image compression
├── DEFLATE.py            # Deflate compression for images
├── DitheringCompression.py # Dithering-based image compression
├── HEVC_AND_AVC.py       # Video compression (HEVC, AVC)
├── HUFFMANN.py           # Huffman coding for text/PDF
├── LOSSLESS.py           # Lossless image compression
├── LOSSY.py              # Lossy grayscale image compression
├── LZ77.py               # LZ77 compression for text/PDF
├── RLE.py                # Run-Length Encoding
├── lzw.py                # LZW compression for text/PDF
├── frontend.html         # Placeholder for frontend (incomplete)
├── Uploads/              # Temporary storage for uploaded files
├── compressed/           # Compressed file output
├── decompressed/         # Decompressed file output
├── archives/             # ZIP archive output
└── README.md             # This file
```

### Backend Workflow
1. **File Upload**: Files are saved to the `Uploads` folder using secure filenames.
2. **Compression/Decompression**:
   - The `app.py` maps algorithms to their respective classes or functions.
   - Compression outputs are saved to the `compressed` folder.
   - Decompression outputs are saved to the `decompressed` folder.
3. **Archiving**: The `archive.py` module creates ZIP archives in the `archives` folder.
4. **Analysis**: Uses libraries like Pillow, pdfminer, and ffprobe to extract file metadata and content.
5. **Cleanup**: Temporary files are removed after processing.

### Algorithm Details
Refer to the `/api/algorithm_details` endpoint for detailed information on each algorithm, including:
- Description
- Pros and cons
- Expected outcomes
- Supported file types (text, image, video, PDF)

## Development Notes

### Extensibility
- Add new algorithms by creating a Python module with `compress` and `decompress` methods, then update `ALGORITHM_MAP` and `DECOMPRESSION_ALGORITHM_MAP` in `app.py`.
- Enhance analysis capabilities by adding new analysis types in the `/analyze` endpoint.

### Known Limitations
- The frontend (`frontend.html`) is incomplete; a custom UI is needed for user interaction.
- Some algorithms (e.g., RLE) are ineffective for non-repetitive data.
- Audio analysis is minimal and requires further implementation.
- Error handling is robust but may need additional edge case coverage.

### Future Improvements
- Implement automatic algorithm selection based on file type and content analysis.
- Add compression ratio and quality metrics to the API responses.
- Support parallel processing for large files or batch operations.
- Develop a fully functional frontend interface.

## Troubleshooting
- **Module Not Found**: Ensure all compression modules are in the project directory or Python path.
- **FFmpeg Errors**: Verify FFmpeg is installed and ffprobe is accessible in the system PATH.
- **File Upload Issues**: Check folder permissions for `Uploads`, `compressed`, `decompressed`, and `archives`.
- **API Errors**: Review server logs (`app.py` logs errors to console) for detailed error messages.

