import pandas as pd
import matplotlib.pyplot as plt

def plot_temperature_profile(filepath):
    header_row_index = 0

    # 1. Scan the file to find the row that contains the actual headers
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if "Time of Day" in line and "Time (s)" in line:
                header_row_index = i
                break

    # 2. Load the data using pandas
    df = pd.read_csv(filepath, sep='\t', skiprows=header_row_index)
    df.columns = df.columns.str.strip()

    # 3. Extract Time and Pyrometer (Temperature) data
    time_data = df["Time (s)"]
    temp_data = df["Pyrometer"]

    # 4. Find the maximum temperature and the time it occurred
    max_idx = temp_data.idxmax()
    max_temp = temp_data.iloc[max_idx]
    max_time = time_data.iloc[max_idx]

    print(f"Maximum Temperature: {max_temp:.2f}")
    print(f"Time of Max Temp:    {max_time:.2f} s")

    # 5. Plot the data
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot the main temperature curve
    ax.plot(time_data, temp_data, color='#D35A45', linewidth=2, label='Pyrometer Trace')

    # Place a prominent dot at the maximum temperature
    ax.plot(max_time, max_temp, 'ko', markersize=8,
            label=f'Maximum: {max_temp:.1f} at {max_time:.1f}s')

    # Formatting the plot
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Pyrometer Signal / Temperature", fontsize=12)
    ax.set_title("Heating Profile", fontsize=14, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    FILE_PATH = r"C:\Users\matth\OneDrive\Desktop\Manuscript data\MAPI_1pct_ABA_S1_30_tube_5min 003294.txt"

    try:
        plot_temperature_profile(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find file at {FILE_PATH}")
    except KeyError as e:
        print(f"Error: Could not find column {e}. Check the file headers.")
