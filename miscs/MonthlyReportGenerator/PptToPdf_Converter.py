import os
import comtypes.client
import tkinter as tk
from tkinter import filedialog
from comtypes import COMError

# Set the folder containing .pptx files
# source_dir = os.getcwd()  # Current working directory (change if needed)
root = tk.Tk();
root.withdraw();
Fold_list = [];
DataFold_list = [];
extension = ['.pptx']
source_dir = filedialog.askdirectory()
saveFolderName = 'PDF_version'


def convert_pptx_to_pdf(pptx_file, pdf_file):
    """Convert PowerPoint .pptx to .pdf using COM automation."""
    powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
    powerpoint.Visible = 1  # Make it visible for debugging (set to 0 to hide)

    try:
        presentation = powerpoint.Presentations.Open(pptx_file, WithWindow=False)  # should be r"folder\path"
        presentation.SaveAs(pdf_file, 32)  # 32 = ppSaveAsPDF
        presentation.Close()
        powerpoint.Quit()  # Ensure PowerPoint is closed
    except COMError as ce:
        if ce.text == '未指定的错误' or ce.text == 'Unspecified error':
            print('Folder / file does not exist')


def checkFileExist_createTmpFolder(source_dir, tmp_pdfFolder):
    for j in os.listdir(source_dir):
        if j.lower().endswith(extension[0]):
            if not os.path.exists(tmp_pdfFolder):
                os.makedirs(tmp_pdfFolder)
            break


# Check if temporal folder for saving PDF exists
tmp_pdfFolder = os.path.join(source_dir + '/' + saveFolderName).replace('/', '\\')
checkFileExist_createTmpFolder(source_dir, tmp_pdfFolder)
# Process all .pptx files in the folder
for filename in os.listdir(source_dir):
    if filename.lower().endswith(extension[0]):
        # print(source_dir + ' || ' + filename)
        tmp_pptx = os.path.join(source_dir, filename)
        pptx_file = tmp_pptx.replace('/', '\\')
        # pdf_file = os.path.splitext(pptx_file)[0] + ".pdf"
        tmp_pdf = os.path.join(source_dir + '/' + saveFolderName + '/' + os.path.splitext(filename)[0] + '.pdf')
        pdf_file = tmp_pdf.replace('/', '\\')

        print(f"Converting: {pptx_file} -> {pdf_file}")
        convert_pptx_to_pdf(pptx_file, pdf_file)

print("Conversion completed.")
