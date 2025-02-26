import os
import time
import tkinter as tk
from tkinter import filedialog
from pypdf import PdfWriter  # Updated from PdfMerger to PdfWriter

# Open file dialog for user to select multiple PDFs
root = tk.Tk()
root.withdraw()
pdf_files = filedialog.askopenfilenames(title="Select PDF files to merge", filetypes=[("PDF Files", "*.pdf")])
root.destroy()

# Exit if no files were selected
if not pdf_files:
    print("No PDF files selected. Exiting.")
    exit()

def merge_pdfs(pdf_list):
    """Merges selected PDF files and saves the output with a timestamp."""
    writer = PdfWriter()  # Updated from PdfMerger

    for pdf in pdf_list:
        if os.path.exists(pdf):
            writer.append(pdf)
        else:
            print(f"Warning: {pdf} not found. Skipping.")

    # Generate output filename in the same directory as the first selected PDF
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(os.path.dirname(pdf_list[0]), f"merged_{timestr}.pdf")

    # Save merged PDF
    with open(output_file, "wb") as out_pdf:
        writer.write(out_pdf)  # Use `write()` inside `with open()`

    print(f"Merged PDF saved as: {output_file}")

# Call the function
merge_pdfs(pdf_files)