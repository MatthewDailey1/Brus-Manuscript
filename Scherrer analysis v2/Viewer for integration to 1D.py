### You are going to run this first to view the 1D integrated curves from your TIF dataset. this will allow you to view the data set to make sure the poni you selected and the data set is converted correctly to the 1D then allows you to export all the frames to a single CSV file with the time stamps and file names in the header for later analysis.
# You can then run the "peak isolation for tifs.py" script to select a specific frame, perform background subtraction, and then isolate and fit peaks for further analysis.
# imported ITO.poni
import os
import glob
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import pyFAI
import fabio  # Used for precise metadata header reading

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class IntegrationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("2D to 1D Integration Explorer & Exporter")
        self.root.geometry("950x780")

        # Core Data Variables
        self.poni_path = ""
        self.folder_path = ""
        self.tif_files = []
        self.file_timestamps = []  # Accurate relative times (seconds)
        self.ai = None             # Azimuthal Integrator instance

        self.setup_ui()

    def setup_ui(self):
        # --- Top Control Panel ---
        control_frame = ttk.LabelFrame(self.root, text=" Setup & Controls ", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # PONI Selection
        ttk.Button(control_frame, text="Select PONI File", command=self.browse_poni).grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.lbl_poni = ttk.Label(control_frame, text="No PONI file selected", foreground="gray")
        self.lbl_poni.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        # Folder Selection
        ttk.Button(control_frame, text="Select TIF Folder", command=self.browse_folder).grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        self.lbl_folder = ttk.Label(control_frame, text="No folder selected", foreground="gray")
        self.lbl_folder.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # Load Button
        self.btn_load = ttk.Button(control_frame, text="Initialize & Parse Headers", command=self.load_data, state=tk.DISABLED)
        self.btn_load.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)

        # --- Slider & Export Panel (Hidden until data is loaded) ---
        self.slider_frame = ttk.Frame(self.root, padding=10)

        # Info Matrix Layout (Left side info, Right side export button)
        info_container = ttk.Frame(self.slider_frame)
        info_container.pack(fill=tk.X)

        text_info_frame = ttk.Frame(info_container)
        text_info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Labels (Counter & Time)
        self.lbl_counter = ttk.Label(text_info_frame, text="TIF: 0 / 0", font=("TkDefaultFont", 10, "bold"))
        self.lbl_counter.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_time = ttk.Label(text_info_frame, text="Time: 0.00 s", font=("TkDefaultFont", 10, "bold"), foreground="blue")
        self.lbl_time.pack(side=tk.LEFT)

        self.lbl_filename = ttk.Label(self.slider_frame, text="Filename: -", foreground="gray")
        self.lbl_filename.pack(anchor="w", pady=(2, 5))

        # Export Button (Placed on the right)
        self.btn_export = ttk.Button(info_container, text="💾 Export All to One CSV", command=self.export_to_csv, style="Accent.TButton")
        self.btn_export.pack(side=tk.RIGHT, padx=5)

        self.slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.pack(fill=tk.X, expand=True, pady=5)

        # --- Matplotlib Plot Panel ---
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def browse_poni(self):
        file_path = filedialog.askopenfilename(filetypes=[("PONI files", "*.poni"), ("All files", "*.*")])
        if file_path:
            self.poni_path = file_path
            self.lbl_poni.config(text=os.path.basename(file_path), foreground="black")
            self.check_ready_to_load()

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_path = folder_path
            self.lbl_folder.config(text=folder_path, foreground="black")
            self.check_ready_to_load()

    def check_ready_to_load(self):
        if self.poni_path and self.folder_path:
            self.btn_load.config(state=tk.NORMAL)

    def load_data(self):
        try:
            self.ai = pyFAI.load(self.poni_path)
            self.tif_files = sorted(glob.glob(os.path.join(self.folder_path, "*.tif*")))

            if not self.tif_files:
                messagebox.showerror("Error", "No .tif or .tiff files found in the selected folder.")
                return

            # --- Extract High-Accuracy Timestamps from Headers ---
            self.file_timestamps = []
            base_time = None
            fallback_used = False

            for f in self.tif_files:
                img = fabio.open(f)
                header = img.header

                raw_time = (
                    header.get("exposure_start") or
                    header.get("time_of_day") or
                    header.get("count_time")
                )

                if raw_time is None:
                    raw_time = os.path.getmtime(f)
                    fallback_used = True
                else:
                    try:
                        raw_time = float(raw_time)
                    except (ValueError, TypeError):
                        raw_time = os.path.getmtime(f)
                        fallback_used = True

                if base_time is None:
                    base_time = raw_time

                self.file_timestamps.append(raw_time - base_time)

            num_files = len(self.tif_files)
            self.slider.config(from_=0, to=num_files - 1)
            self.slider.set(0)

            # Show the navigation slider panel
            self.slider_frame.pack(fill=tk.X, padx=10, pady=5, before=self.plot_frame)
            self.update_plot(0)

            msg = f"Successfully loaded {num_files} images."
            if fallback_used:
                msg += "\n\n⚠️ Note: OS file write times used as fallback timing."
            else:
                msg += "\n\n🔬 Success: High-accuracy metadata headers loaded."

            messagebox.showinfo("Success", msg)

        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to parse files:\n{str(e)}")

    def on_slider_move(self, value):
        idx = int(float(value))
        self.update_plot(idx)

    def update_plot(self, idx):
        if not self.tif_files or self.ai is None:
            return

        file_path = self.tif_files[idx]
        filename = os.path.basename(file_path)
        total_files = len(self.tif_files)
        current_time = self.file_timestamps[idx]

        self.lbl_counter.config(text=f"TIF: {idx + 1} / {total_files}")
        self.lbl_time.config(text=f"Time: {current_time:.2f} s")
        self.lbl_filename.config(text=f"Filename: {filename}")

        try:
            img = fabio.open(file_path)
            data = img.data

            # You can change npt=1000 or unit="2th_deg" as needed
            res = self.ai.integrate1d(data, npt=1000, unit="2th_deg")
            x, y = res.radial, res.intensity

            self.ax.clear()
            self.ax.plot(x, y, color="crimson", linewidth=1.5)
            self.ax.set_title(f"1D Integration Profile — {filename}")
            self.ax.set_xlabel(res.unit.label)
            self.ax.set_ylabel("Intensity (a.u.)")
            self.ax.grid(True, linestyle="--", alpha=0.5)
            self.ax.set_xlim(x.min(), x.max())
            self.ax.set_ylim(y.min() - (y.max() * 0.02), y.max() * 1.05)

            self.canvas.draw()

        except Exception as e:
            print(f"Error drawing frame {idx}: {e}")

    def export_to_csv(self):
        if not self.tif_files or self.ai is None:
            return

        # Create 'processed_data' sub-directory safely
        output_dir = os.path.join(self.folder_path, "processed_data")
        os.makedirs(output_dir, exist_ok=True)

        csv_file_path = os.path.join(output_dir, "integrated_profiles_master.csv")

        # Change state of button to show it is working
        self.btn_export.config(state=tk.DISABLED, text="Processing Data...")
        self.root.update_idletasks()

        try:
            # We'll use the first file to initialize our grid framework
            first_img = fabio.open(self.tif_files[0])
            # Set target points (npt) identical to the UI view settings
            npt_points = 1000
            unit_type = "2th_deg"

            res_first = self.ai.integrate1d(first_img.data, npt=npt_points, unit=unit_type)
            x_axis = res_first.radial
            x_label = res_first.unit.label

            # Construct CSV Column Headers
            # Metadata Row 1: Time Stamps
            row_times = [x_label] + [f"Time: {t:.2f}s" for t in self.file_timestamps]
            # Metadata Row 2: Original Filenames
            row_filenames = ["Filename:"] + [os.path.basename(f) for f in self.tif_files]

            # Prepare an array matrix to collect all the column values sequentially
            all_intensities = []

            # Process all arrays on-the-fly sequentially
            for idx, file_path in enumerate(self.tif_files):
                img = fabio.open(file_path)
                res = self.ai.integrate1d(img.data, npt=npt_points, unit=unit_type)
                all_intensities.append(res.intensity)

            # Convert to numpy array and transpose so each frame is a clean vertical column
            intensity_columns = np.column_stack(all_intensities)

            # Stream directly out to file row by row to maintain memory efficiency
            with open(csv_file_path, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)

                # Write descriptive header tags
                writer.writerow(row_times)
                writer.writerow(row_filenames)

                # Write data rows
                for i in range(len(x_axis)):
                    row_data = [x_axis[i]] + list(intensity_columns[i])
                    writer.writerow(row_data)

            messagebox.showinfo("Export Complete", f"Successfully compiled all data!\n\nSaved to:\n{csv_file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save batch CSV compile:\n{str(e)}")

        finally:
            # Re-enable the export button
            self.btn_export.config(state=tk.NORMAL, text="💾 Export All to One CSV")


if __name__ == "__main__":
    root = tk.Tk()

    # Simple styling cleanup for modern buttons
    style = ttk.Style()
    style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"), foreground="darkgreen")

    app = IntegrationViewer(root)
    root.mainloop()
