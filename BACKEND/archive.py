import os
import zipfile
from werkzeug.utils import secure_filename

class FileArchiver:
    def __init__(self, upload_folder, archive_folder):
        """
        Initialize the FileArchiver with paths to upload and archive folders.
        
        :param upload_folder: Folder where uploaded files are stored.
        :param archive_folder: Folder where ZIP archives will be saved.
        """
        self.upload_folder = upload_folder
        self.archive_folder = archive_folder
        os.makedirs(self.archive_folder, exist_ok=True)

    def archive_files(self, files, archive_name="archive.zip"):
        """
        Archives a list of files into a ZIP file.
        
        :param files: List of file objects from Flask request.files.
        :param archive_name: Name of the output ZIP file (default: archive.zip).
        :return: Path to the created ZIP archive.
        """
        if not files or len(files) == 0:
            raise ValueError("No files provided for archiving.")

        # Secure the archive filename
        archive_name = secure_filename(archive_name)
        if not archive_name.lower().endswith('.zip'):
            archive_name += '.zip'
        archive_path = os.path.join(self.archive_folder, archive_name)

        # Save uploaded files temporarily
        temp_file_paths = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(self.upload_folder, filename)
                file.save(file_path)
                temp_file_paths.append((file_path, filename))

        # Create ZIP archive
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path, filename in temp_file_paths:
                    if os.path.exists(file_path):
                        zipf.write(file_path, filename)
                    else:
                        raise FileNotFoundError(f"File {filename} not found after saving.")
        except Exception as e:
            # Clean up temporary files in case of error
            for file_path, _ in temp_file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
            raise RuntimeError(f"Failed to create archive: {str(e)}")

        # Clean up temporary files
        for file_path, _ in temp_file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

        if not os.path.exists(archive_path):
            raise RuntimeError("Archive creation failed: Output file not found.")

        return archive_path