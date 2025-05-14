import ffmpeg
import os

def compress_mp4_hevc(input_file: str, output_file: str, crf: int = 28):
    """
    Compress an MP4 file using HEVC (H.265) codec.

    :param input_file: Path to the input .mp4 file.
    :param output_file: Path to save the compressed .mp4 file.
    :param crf: Constant Rate Factor (default: 28, lower is higher quality).
    """
    if not os.path.exists(input_file):
        print("Input file does not exist.")
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    try:
        ffmpeg.input(input_file).output(output_file, vcodec='libx265', crf=crf, preset='slow').run(overwrite_output=True)
        print(f"HEVC Compression successful: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error during HEVC compression: {e}")
        raise e

def compress_mp4_avc(input_file: str, output_file: str, crf: int = 23):
    """
    Compress an MP4 file using AVC (H.264) codec.

    :param input_file: Path to the input .mp4 file.
    :param output_file: Path to save the compressed .mp4 file.
    :param crf: Constant Rate Factor (default: 23, lower is higher quality).
    """
    if not os.path.exists(input_file):
        print("Input file does not exist.")
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    try:
        ffmpeg.input(input_file).output(output_file, vcodec='libx264', crf=crf, preset='slow').run(overwrite_output=True)
        print(f"AVC Compression successful: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error during AVC compression: {e}")
        raise e