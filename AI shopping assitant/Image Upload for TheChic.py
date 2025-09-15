import fitz  # This is the PyMuPDF library
import os

# --- Configuration ---
# The full path to your PDF file.
PDF_FILE = r"C:\Users\amarachukwuemenike\Downloads\Receipt.pdf"

# --- This is the section you requested to change ---
# Get the path to the user's Downloads folder
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")

# Set the output folder to be a new folder inside your Downloads folder
OUTPUT_DIR = os.path.join(downloads_path, "renamed_pdf_images")
# --- End of changed section ---

# The prefix for your new image filenames.
PREFIX = "cb_"

# --- Step 1: Extract and Save Images from the PDF ---
def extract_and_save_images(pdf_path, output_dir):
    """
    Extracts images from a PDF and saves them to a specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    doc = fitz.open(pdf_path)
    image_paths = []
    image_count = 0

    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_num)):
            image_count += 1
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            temp_filename = f"image_{page_num+1}_{img_index+1}.{image_ext}"
            image_path = os.path.join(output_dir, temp_filename)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(image_path)
            
    doc.close()
    
    if image_count == 0:
        print("No images found in the PDF.")
    else:
        print(f"Successfully extracted {image_count} image(s) to the '{output_dir}' folder.")
        
    return image_paths

# --- Step 2: Rename the Extracted Images ---
def rename_images_sequentially(image_paths, prefix):
    """
    Renames a list of image files with a specific prefix and an increment of 2.
    """
    counter = 1
    for old_path in image_paths:
        directory = os.path.dirname(old_path)
        extension = os.path.splitext(old_path)[1]
        
        new_name = f"{prefix}{counter}{extension}"
        new_path = os.path.join(directory, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed '{os.path.basename(old_path)}' to '{new_name}'")
        
        counter += 2

# --- Main part of the script that runs everything ---
if __name__ == "__main__":
    if not os.path.exists(PDF_FILE):
        print(f"Error: The file was not found at the path: {PDF_FILE}")
        print("Please make sure the PDF_FILE path at the top of the script is correct.")
    else:
        extracted_files = extract_and_save_images(PDF_FILE, OUTPUT_DIR)
        
        if extracted_files:
            rename_images_sequentially(extracted_files, PREFIX)
            print("\nProcess complete!")
            print(f"You can find your renamed images in your Downloads folder, inside a new folder named: 'renamed_pdf_images'")