import os
import comtypes.client
import tkinter as tk
from tkinter import filedialog


# Set the folder containing .pptx files
# source_dir = os.getcwd()  # Current working directory (change if needed)
root = tk.Tk(); root.withdraw(); Fold_list = []; DataFold_list = []; extension = ['.pptx']
source_dir = filedialog.askdirectory()
saveFolderName = 'PDF_version'

def convert_pptx_to_pdf(pptx_file, pdf_file):
    """Convert PowerPoint .pptx to .pdf using COM automation."""
    powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
    powerpoint.Visible = 1  # Make it visible for debugging (set to 0 to hide)

    presentation = powerpoint.Presentations.Open(pptx_file, WithWindow=False)  # should be r"folder\path"
    presentation.SaveAs(pdf_file, 32)  # 32 = ppSaveAsPDF
    presentation.Close()
    powerpoint.Quit()  # Ensure PowerPoint is closed


# Check if temporal folder for saving PDF exists
tmp_pdfFolderCheck = os.path.join(source_dir + '/' + saveFolderName)
tmp_pdfFolder = tmp_pdfFolderCheck.replace('/', '\\')
if not os.path.exists(tmp_pdfFolder):  os.makedirs(tmp_pdfFolder)
# Process all .pptx files in the folder
for filename in os.listdir(source_dir):
    if filename.lower().endswith(".pptx"):
        # print(source_dir + ' || ' + filename)
        tmp_pptx = os.path.join(source_dir, filename)
        pptx_file = tmp_pptx.replace('/', '\\')
        # pdf_file = os.path.splitext(pptx_file)[0] + ".pdf"
        tmp_pdf = os.path.join(source_dir + '/' + saveFolderName + '/' + os.path.splitext(filename)[0] + '.pdf')
        pdf_file = tmp_pdf.replace('/', '\\')

        print(f"Converting: {pptx_file} -> {pdf_file}")
        convert_pptx_to_pdf(pptx_file, pdf_file)

print("Conversion completed.")