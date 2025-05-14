import os
from PIL import Image

class LosslessImageCompressor:
    def __init__(self, format="PNG"):
        """
        Initialize the compressor with a default format.
        Recommended: PNG or WEBP for lossless.
        """
        self.format = format.upper()

    def compress(self, input_path, output_folder):
        try:
            print("Compressing...")

            filename = os.path.splitext(os.path.basename(input_path))[0]
            # Set the extension based on the chosen format
            extension = self.format.lower()
            output_path = os.path.join(output_folder, f"{filename}_compressed.{extension}")

            with Image.open(input_path) as img:
                # Strip metadata
                img_no_metadata = Image.new(img.mode, img.size)
                img_no_metadata.putdata(list(img.getdata()))

                save_args = {"optimize": True}
                if self.format == "WEBP":
                    save_args["lossless"] = True  # Use lossless compression for WebP

                img_no_metadata.save(output_path, format=self.format, **save_args)

            return output_path

        except Exception as e:
            print(f"Error compressing image: {e}")
            return None
