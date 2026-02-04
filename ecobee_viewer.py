import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, CheckButtons, TextBox
from tkinter import Tk, filedialog
import os

# Import the analyzer module
try:
    from thermal_analysis import ThermalAnalyzer
except ImportError:
    print("Warning: thermal_analysis.py not found. Analysis buttons will fail.")
    ThermalAnalyzer = None

class ThermostatApp:
    def __init__(self):
        self.df = pd.DataFrame()
        self.power_files = []
        self.files_loaded = []
        self.sq_ft_val = "1500" # Updated Default
        
        # Setup the main plot window
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, 
                                                      gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(bottom=0.25)
        
        # --- UI CONTROLS ---
        
        # 1. Load Ecobee CSV (Left)
        ax_load = plt.axes([0.05, 0.05, 0.1, 0.075])
        self.btn_load = Button(ax_load, 'Load Ecobee')
        self.btn_load.on_clicked(self.load_data)

        # 2. Load Power XML (Left-Center)
        ax_load_pwr = plt.axes([0.16, 0.05, 0.1, 0.075])
        self.btn_load_pwr = Button(ax_load_pwr, 'Load Power')
        self.btn_load_pwr.on_clicked(self.load_power)

        # 3. Checkboxes (Center)
        ax_checks = plt.axes([0.28, 0.05, 0.15, 0.1])
        self.check_labels = ['Night Only', '30m Buffer', 'Zoomed Only']
        self.check = CheckButtons(ax_checks, self.check_labels, [False, True, False])
        ax_checks.set_frame_on(False)

        # 4. Square Footage Input (Center-Right)
        ax_box = plt.axes([0.44, 0.08, 0.07, 0.05])
        self.text_box = TextBox(ax_box, 'Sq Ft: ', initial=self.sq_ft_val)
        self.text_box.on_submit(self.submit_sq_ft)

        # 5. Timezone Input (Right-of-Center)
        self.tz_val = "US/Pacific"
        ax_tz = plt.axes([0.55, 0.08, 0.10, 0.05])
        self.text_tz = TextBox(ax_tz, 'TZ: ', initial=self.tz_val)
        self.text_tz.on_submit(self.submit_tz)

        # 6. Calculate Button (Right)
        ax_calc = plt.axes([0.68, 0.05, 0.2, 0.075])
        self.btn_calc = Button(ax_calc, 'Calculate Profile')
        self.btn_calc.on_clicked(self.run_calculation)
        
        self.set_empty_view()
        plt.show()

    def submit_sq_ft(self, text):
        self.sq_ft_val = text

    def submit_tz(self, text):
        self.tz_val = text

    def set_empty_view(self):
        self.ax1.set_title("No Data Loaded. Click 'Load Ecobee' to select CSV files.")
        self.ax1.grid(True, alpha=0.3)
        self.ax2.set_xlabel("Time")

    def load_data(self, event):
        root = Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
        root.destroy()
        
        if not file_paths: return

        new_dfs = []
        for fp in file_paths:
            if fp not in self.files_loaded:
                try:
                    # Ecobee format usually has 5 rows of metadata
                    temp_df = pd.read_csv(fp, skiprows=5, index_col=False)
                    new_dfs.append(temp_df)
                    self.files_loaded.append(fp)
                    print(f"Loaded CSV: {fp}")
                except Exception as e:
                    print(f"Error loading {fp}: {e}")

        if new_dfs:
            if not self.df.empty: new_dfs.append(self.df)
            full_df = pd.concat(new_dfs)
            # Create standardized Timestamp
            full_df['Timestamp'] = pd.to_datetime(full_df['Date'] + ' ' + full_df['Time'])
            full_df = full_df.drop_duplicates(subset=['Timestamp']).sort_values('Timestamp')
            self.df = full_df
            self.update_plot()

    def load_power(self, event):
        root = Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames(filetypes=[("XML Files", "*.xml")])
        root.destroy()
        
        if file_paths:
            self.power_files.extend(file_paths)
            print(f"Loaded {len(file_paths)} Power XML files. (Will process during calculation)")
            self.ax1.set_title(f"Ecobee Data | {len(self.files_loaded)} CSVs | {len(self.power_files)} Power XMLs")
            self.fig.canvas.draw()

    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        
        df = self.df
        if df.empty: return

        # --- Pre-processing for Visualization ---
        if 'Program Mode' in df.columns:
            df['Program Mode'] = df['Program Mode'].ffill()
        
        # Identify System States for Background Fills
        df['is_cooling'] = df['System Mode'].str.contains('CoolStage', na=False) & df['System Mode'].str.contains('On', na=False)
        df['is_heating'] = df['System Mode'].str.contains('HeatStage', na=False) & df['System Mode'].str.contains('On', na=False)

        # --- TOP SUBPLOT: Temperatures ---
        times = df['Timestamp']
        # Determine Y-axis limits dynamically
        cols = ['Cool Set Temp (F)', 'Heat Set Temp (F)', 'Current Temp (F)', 'Outdoor Temp (F)']
        valid_cols = [c for c in cols if c in df.columns]
        y_min = df[valid_cols].min().min() - 2
        y_max = df[valid_cols].max().max() + 2

        # Background Fills (System State)
        self.ax1.fill_between(times, y_min, y_max, where=df['is_cooling'], color='blue', alpha=0.15, label='Cooling On')
        self.ax1.fill_between(times, y_min, y_max, where=df['is_heating'], color='red', alpha=0.15, label='Heating On')

        # Line Plots
        if 'Cool Set Temp (F)' in df.columns:
            self.ax1.plot(times, df['Cool Set Temp (F)'], color='blue', linestyle='--', alpha=0.7, label='Cool Set')
        if 'Heat Set Temp (F)' in df.columns:
            self.ax1.plot(times, df['Heat Set Temp (F)'], color='red', linestyle='--', alpha=0.7, label='Heat Set')
        if 'Outdoor Temp (F)' in df.columns:
            self.ax1.plot(times, df['Outdoor Temp (F)'], color='green', linestyle='-', alpha=0.8, linewidth=1, label='Outdoor Temp')
        if 'Current Temp (F)' in df.columns:
            self.ax1.plot(times, df['Current Temp (F)'], color='black', linestyle='-', linewidth=1.5, label='Indoor Temp')

        self.ax1.set_ylabel('Temperature (°F)')
        self.ax1.set_ylim(y_min, y_max)
        self.ax1.legend(loc='upper left', fontsize='small', framealpha=0.9)
        self.ax1.set_title(f"Ecobee Data | {df['Timestamp'].min().date()} - {df['Timestamp'].max().date()} | {len(self.power_files)} Power Files")
        self.ax1.grid(True, alpha=0.3)

        # --- BOTTOM SUBPLOT: Program Mode ---
        # Colors: Sleep=Purple, Home=Green, Away=Grey
        if 'Program Mode' in df.columns:
            is_sleep = df['Program Mode'] == 'Sleep'
            is_home = df['Program Mode'] == 'Home'
            is_away = df['Program Mode'] == 'Away'
            
            self.ax2.fill_between(times, 0, 1, where=is_sleep, color='purple', alpha=0.4, step='mid')
            self.ax2.fill_between(times, 0, 1, where=is_home, color='green', alpha=0.4, step='mid')
            self.ax2.fill_between(times, 0, 1, where=is_away, color='grey', alpha=0.4, step='mid')

            patches = [
                mpatches.Patch(color='purple', label='Sleep', alpha=0.4),
                mpatches.Patch(color='green', label='Home', alpha=0.4),
                mpatches.Patch(color='grey', label='Away', alpha=0.4)
            ]
            self.ax2.legend(handles=patches, loc='upper left', fontsize='small')
        
        self.ax2.set_yticks([])
        self.ax2.set_ylabel('Program')
        
        # Date Formatting
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        self.fig.autofmt_xdate()
        
        self.fig.canvas.draw()

    def get_options(self):
        status = self.check.get_status()
        try:
            sq = float(self.sq_ft_val)
        except:
            sq = 1500
        return {
            'night_only': status[0],
            'buffer_minutes': 30 if status[1] else 0,
            'use_zoomed': status[2],
            'sq_ft': sq,
            'timezone': self.tz_val
        }

    def _generate_file_suffix(self, opts):
        suffix = ""
        if opts['night_only']: suffix += "_night"
        if opts['buffer_minutes'] > 0: suffix += f"_buffer{opts['buffer_minutes']}"
        suffix += f"_{int(opts['sq_ft'])}sqft"
        # We don't necessarily need TZ in filename unless user wants it, but it helps tracking? 
        # For now, keep it simple as requested.
        return suffix

    def run_calculation(self, event):
        if self.df.empty:
            print("No data loaded.")
            return

        opts = self.get_options()
        target_df = self.df.copy()

        # Handle Zoomed Range Logic
        if opts['use_zoomed']:
            x_min, x_max = self.ax1.get_xlim()
            try:
                d_start = mdates.num2date(x_min).replace(tzinfo=None)
                d_end = mdates.num2date(x_max).replace(tzinfo=None)
                
                mask = (target_df['Timestamp'] >= d_start) & (target_df['Timestamp'] <= d_end)
                target_df = target_df.loc[mask]
                
                if len(target_df) < 50:
                    print("Zoomed range contains too little data (<50 pts). Analysis aborted.")
                    return
                print(f"Using Zoomed Range: {d_start} to {d_end}")
            except Exception as e:
                print(f"Error processing zoomed range: {e}")
                return

        print(f"Running Analysis... SqFt={opts['sq_ft']}, Night={opts['night_only']}, TZ={opts['timezone']}")
        
        if ThermalAnalyzer:
            analyzer = ThermalAnalyzer(dataframe=target_df.set_index('Timestamp'))
            
            # Load Power Data if available
            if self.power_files:
                analyzer.load_power_data(self.power_files)
            
            # Run Analysis
            try:
                analyzer.analyze(
                    night_only=opts['night_only'], 
                    buffer_minutes=opts['buffer_minutes'], 
                    sq_ft=opts['sq_ft'],
                    timezone=opts['timezone']
                )
                
                # Generate Filenames
                start_str = target_df['Timestamp'].min().strftime('%Y%m%d')
                end_str = target_df['Timestamp'].max().strftime('%Y%m%d')
                suffix = self._generate_file_suffix(opts)
                base_name = f"{start_str}-{end_str}{suffix}"
                
                # Outputs
                analyzer.generate_report(filename=f"thermal_results_{base_name}.txt")
                analyzer.plot_results(filename=f"thermal_plot_{base_name}.png", show=True)
                
                if self.power_files:
                    analyzer.plot_inverter_curve(filename=f"inverter_profile_{base_name}.png", show=True)
                    analyzer.plot_energy_profile(filename=f"energy_total_{base_name}.png", show=True)
                    
            except Exception as e:
                print(f"Analysis Failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    app = ThermostatApp()