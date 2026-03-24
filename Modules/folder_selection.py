import os
import tkinter as tk
from tkinter import filedialog, messagebox
from natsort import os_sorted


def select_multiple_folders():
    global _last_initial_dir
    def on_confirm():
        indices = listbox.curselection()  # 获取所有选中项的索引
        if not indices:
            messagebox.showwarning("Warning", "Select at least one Data folder")
            return
        for i in indices:
            selected_paths.append(os.path.join(parent_dir, subfolders[i]))
        dialog.destroy()

    def on_cancel():
        dialog.destroy()

    def on_selection_change(event):
        count = len(listbox.curselection())
        status_var.set(f"Selected：{count} Data folders")

    root = tk.Tk()
    root.withdraw()
    parent_dir = filedialog.askdirectory(title="Select Parent Folder that contains all data folders")
    if not parent_dir:
        root.destroy()
        return []
    
    subfolders = os_sorted([
        f for f in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, f))
    ])

    if not subfolders:
        messagebox.showinfo("Error", "No data folders found in the selected directory.")
        root.destroy()
        return []

    # 弹出多选列表窗口 ──
    selected_paths = []
    dialog = tk.Toplevel(root)
    dialog.title("Select data folders（multiple available (Shift / Ctrl）")
    dialog.geometry("350x550")
    dialog.resizable(True, True)
    dialog.grab_set()  # 模态窗口，必须先操作这个才能动其他窗口
    
    tk.Label(dialog, text=f"Parent folder：{parent_dir}", anchor="w", fg="gray", wraplength=330).pack(fill="x", padx=10, pady=(10, 0))
    tk.Label(dialog, text="Click | Shift+Click | Ctrl+Click", anchor="w", fg="#555").pack(fill="x", padx=10)
    
    frame = tk.Frame(dialog)
    frame.pack(fill="both", expand=True, padx=10, pady=5)
    scrollbar = tk.Scrollbar(frame, orient="vertical")
    listbox = tk.Listbox(frame, selectmode=tk.EXTENDED,   # 支持 Shift/Ctrl 多选
        yscrollcommand=scrollbar.set, font=("Consolas", 11), activestyle="dotbox", height=20)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    for folder in subfolders:
        listbox.insert(tk.END, folder)
        
    status_var = tk.StringVar(value="Selected：0 Data folder")
    status_label = tk.Label(dialog, textvariable=status_var, fg="#333")
    status_label.pack()

    listbox.bind("<<ListboxSelect>>", on_selection_change)
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=8)
    tk.Button(btn_frame, text="✔ Confirm", width=12, bg="#4CAF50", fg="white", command=on_confirm).pack(side="left", padx=5)
    tk.Button(btn_frame, text="✘ Cancel", width=12, command=on_cancel).pack(side="left", padx=5)

    root.wait_window(dialog)
    root.destroy()
    return selected_paths

if __name__ == "__main__":

    print("Testing select_multiple_folders()...")
    selected = select_multiple_folders()
    if selected:
        print("\nSelected folders:")
        for path in selected:
            print(f" - {path}")
    else:
        print("\nNo folders selected or operation cancelled.")
