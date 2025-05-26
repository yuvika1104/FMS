import os
from PIL import Image
import numpy as np

class DitheringCompression:
    """
    Image compression using color quantization with Floyd-Steinberg dithering (lossy).
    The output is a standard image format (e.g., PNG) with a reduced color palette.
    "Decompression" is simply opening the resulting image file.
    """

    def __init__(self):
        pass

    def _quantize_and_dither_pixel(self, old_pixel, new_palette):
        """Finds the closest color in the new_palette for a given pixel."""
        # old_pixel is (R, G, B)
        # new_palette is a list of (R, G, B) tuples
        closest_color = min(new_palette, key=lambda color: sum((c1 - c2)**2 for c1, c2 in zip(old_pixel, color)))
        return np.array(closest_color, dtype=np.uint8)

    def compress(self, input_file_path, output_folder, num_colors=16):
        """
        Compresses an image by reducing its color palette and applying Floyd-Steinberg dithering.
        :param input_file_path: Path to the input image.
        :param output_folder: Folder to save the processed image.
        :param num_colors: The number of colors for the output image's palette.
        :return: Path to the processed (compressed) image file.
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input image file not found: {input_file_path}")

        try:
            img = Image.open(input_file_path).convert('RGB') # Ensure RGB
        except Exception as e:
            raise IOError(f"Error reading image file {input_file_path}: {e}")

        base_name = os.path.basename(input_file_path)
        file_name_no_ext, _ = os.path.splitext(base_name)
        # Output as PNG to preserve the dithered effect without further lossy compression artifacts
        output_file_path = os.path.join(output_folder, file_name_no_ext + f"_dithered_{num_colors}colors.png")
        os.makedirs(output_folder, exist_ok=True)

        # 1. Create a new palette (e.g., by quantizing the original image or using a fixed one)
        # For simplicity, we'll use Pillow's built-in quantize method to get a palette,
        # then apply dithering manually for demonstration, or use its dithering if available.
        # Pillow's quantize method can already apply dithering.
        # If we want to show manual Floyd-Steinberg:
        
        # Create a simple palette (e.g. by taking `num_colors` from a uniformly quantized space)
        if num_colors <= 0: num_colors = 2
        if num_colors > 256: num_colors = 256 # PIL quantize to P mode supports max 256

        # Use Pillow's quantize with dithering for a robust solution
        # Image.Dither.FLOYDSTEINBERG is the default if dither is not None.
        try:
            # Quantize to a palette of `num_colors`.
            # If `num_colors` is small (e.g. <= 256), it can convert to "P" (palette) mode.
            # For more colors, it might stay in RGB but with reduced unique colors.
            # Using quantize with a specific number of colors.
            if hasattr(Image, 'Quantize'): # Newer Pillow versions
                 quantized_img = img.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
            else: # Older Pillow versions might use this
                 quantized_img = img.quantize(colors=num_colors, method=2, dither=Image.FLOYDSTEINBERG) # method 2 is MEDIANCUT

            # If the image is not already in P mode with a limited palette, convert it.
            # This ensures the output PNG actually has a small palette if num_colors is small.
            if quantized_img.mode != 'P' and num_colors <= 256:
                quantized_img = quantized_img.convert('P', palette=Image.Palette.ADAPTIVE, colors=num_colors, dither=Image.Dither.FLOYDSTEINBERG)

            quantized_img.save(output_file_path, "PNG")

        except Exception as e:
            # Fallback to a simpler manual approach if advanced quantize fails or for learning
            # This manual Floyd-Steinberg is more illustrative but might be slower/less optimized.
            # For this fallback, let's just save with reduced colors without manual dithering.
            # (A full manual Floyd-Steinberg is a bit long to add here robustly)
            try:
                img_p = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=num_colors)
                img_p.save(output_file_path, "PNG")
                print(f"Warning: Advanced dithering failed, used basic palette conversion. Error: {str(e)}")
            except Exception as e2:
                 raise IOError(f"Error processing image with dithering for {input_file_path}: {str(e2)}")

        return output_file_path

    # No specific decompress method needed as output is a standard image.