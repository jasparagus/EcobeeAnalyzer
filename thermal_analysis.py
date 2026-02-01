import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import xml.etree.ElementTree as ET

class ThermalAnalyzer:
    """
    Analyzes Ecobee data to determine building envelope properties (UA, Time Constant)
    and estimates HVAC power usage using Reverse Calorimetry and Green Button Data.
    """
    
    def __init__(self, file_paths=None, dataframe=None):
        self.file_paths = file_paths
        self.df = dataframe
        self.power_df = pd.DataFrame() # Store Green Button Data
        
        self.coasting_df = None
        self.cooling_data = None
        self.heating_data = None
        self.fit_cooling = None
        self.fit_heating = None
        self.daily_combined = None
        
        # Physics Constants
        self.sq_ft = 1500  # Default updated to 1500
        self.C_est = 0     
        self.UA_est = 0    
        self.baseload_kw = 0.5 # Default estimate
        
    def load_data(self):
        if self.df is not None: return

        if not self.file_paths:
            raise ValueError("No file paths or DataFrame provided.")

        dfs = []
        for f in self.file_paths:
            try:
                temp_df = pd.read_csv(f, skiprows=5, index_col=False)
                dfs.append(temp_df)
                print(f"Loaded: {f}")
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not dfs:
            raise ValueError("No data loaded.")
            
        self.df = pd.concat(dfs)
        self.df['Timestamp'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'])
        self.df = self.df.drop_duplicates(subset=['Timestamp']).sort_values('Timestamp')
        self.df = self.df.set_index('Timestamp')

    def load_power_data(self, file_paths):
        """Parses Green Button XML files and merges them."""
        readings = []
        for fp in file_paths:
            try:
                tree = ET.parse(fp)
                root = tree.getroot()
                # Namespace-agnostic parsing: iterate all elements
                for elem in root.iter():
                    if elem.tag.endswith('IntervalReading'):
                        start = None
                        duration = None
                        value = None
                        for child in elem.iter():
                            if child.tag.endswith('start'): start = int(child.text)
                            elif child.tag.endswith('duration'): duration = int(child.text)
                            elif child.tag.endswith('value'): value = int(child.text)
                        
                        # FILTER: Only accept roughly daily readings (86400s)
                        # This ignores Monthly summaries (duration > 25 hours)
                        if start is not None and value is not None:
                            if duration and duration < 90000: 
                                readings.append({'Start': start, 'Duration': duration, 'Value': value})
            except Exception as e:
                print(f"Error parsing XML {fp}: {e}")
                
        if readings:
            pdf = pd.DataFrame(readings)
            pdf['Timestamp'] = pd.to_datetime(pdf['Start'], unit='s')
            # Assuming 'Value' is Wh. For Daily files, it's often kWh directly or Wh scaled.
            # Based on user snippet, values like 30-50 for daily suggest kWh.
            pdf['kWh'] = pdf['Value'] 
            pdf = pdf.set_index('Timestamp').sort_index()
            
            # Resample to Daily sum to ensure alignment
            self.power_df = pdf['kWh'].resample('D').sum().to_frame()
            print(f"Loaded {len(self.power_df)} valid daily power readings.")
        else:
            print("No valid daily readings found in XML (checked for duration < 25h).")

    def analyze(self, night_only=False, buffer_minutes=30, sq_ft=1500):
        if self.df is None: self.load_data()
        self.sq_ft = sq_ft
        
        # 1. Estimate Thermal Mass (C)
        self.C_est = self.sq_ft * 3.5 

        # 2. Pre-process Ecobee Data
        self.df = self.df.sort_index()
        required_cols = ['Current Temp (F)', 'Outdoor Temp (F)', 'Cool Stage 1 (sec)', 'Heat Stage 1 (sec)']
        if not all(col in self.df.columns for col in required_cols):
             raise ValueError(f"Missing columns: {required_cols}")

        resampled = self.df[required_cols].resample('15min').mean()
        resampled['dT_dt'] = resampled['Current Temp (F)'].diff() * 4 
        resampled['Delta_T'] = resampled['Outdoor Temp (F)'] - resampled['Current Temp (F)']
        
        # Coasting Logic (for UA)
        resampled['is_off_now'] = (resampled['Cool Stage 1 (sec)'] < 60) & (resampled['Heat Stage 1 (sec)'] < 60)
        window_count = int(np.ceil(buffer_minutes / 15)) + 1
        resampled['is_stable_off'] = resampled['is_off_now'].rolling(window=window_count).min() == 1

        if night_only:
            mask_night = (resampled.index.hour >= 23) | (resampled.index.hour <= 6)
            resampled['is_valid_time'] = mask_night
        else:
            resampled['is_valid_time'] = True

        self.coasting_df = resampled[
            resampled['is_stable_off'] & 
            resampled['is_valid_time'] & 
            (resampled['dT_dt'].abs() < 5)
        ].dropna()
        
        self.cooling_data = self.coasting_df[self.coasting_df['Delta_T'] > 2]
        self.heating_data = self.coasting_df[self.coasting_df['Delta_T'] < -2]
        
        # Calculate Fits (Slope = UA/C)
        if len(self.heating_data) > 20:
            self.fit_heating = stats.linregress(self.heating_data['Delta_T'], self.heating_data['dT_dt'])[:2]
        if len(self.cooling_data) > 20:
            self.fit_cooling = stats.linregress(self.cooling_data['Delta_T'], self.cooling_data['dT_dt'])[:2]

        # 3. Inverter Power Analysis (Daily)
        if not self.power_df.empty:
            self._analyze_inverter_performance()

    def _analyze_inverter_performance(self):
        """Correlates Daily Ecobee Runtime with Daily Green Button Energy."""
        # Aggregate Ecobee data to Day
        cols = ['Cool Stage 1 (sec)', 'Heat Stage 1 (sec)', 'Fan (sec)']
        daily_ecobee = self.df[cols].resample('D').sum()
        
        daily_temps = self.df[['Current Temp (F)', 'Outdoor Temp (F)']].resample('D').mean()
        
        # Calculate Runtime in Hours
        daily_ecobee['Runtime_Cool_Hr'] = daily_ecobee['Cool Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Heat_Hr'] = daily_ecobee['Heat Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Total_Hr'] = daily_ecobee['Runtime_Cool_Hr'] + daily_ecobee['Runtime_Heat_Hr']
        
        # Merge Ecobee + Power
        combined = pd.concat([daily_ecobee, daily_temps, self.power_df], axis=1).dropna()
        
        if combined.empty:
            print("Warning: No date overlap between Thermostat and Power data.")
            return

        # Estimate Baseload (Lowest usage days where HVAC was mostly off)
        # We look for days with < 0.2 hours of HVAC
        low_usage_days = combined[combined['Runtime_Total_Hr'] < 0.2]
        
        if not low_usage_days.empty:
            self.baseload_kw = low_usage_days['kWh'].mean() / 24.0
            print(f"Baseload estimated from off-days: {self.baseload_kw:.2f} kW")
        else:
            # Fallback: 5th percentile of total daily power
            self.baseload_kw = (combined['kWh'] / 24.0).quantile(0.05)
            print(f"Baseload estimated (quantile): {self.baseload_kw:.2f} kW")

        # Calculate HVAC Specific Energy
        combined['kWh_HVAC'] = combined['kWh'] - (self.baseload_kw * 24.0)
        
        # Calculate Inverter Power (kW)
        # We need to filter out low-runtime days to avoid the "1/x" asymptote noise
        # FILTER: Require at least 2.5 hours of runtime to calculate average power
        valid = combined[combined['Runtime_Total_Hr'] > 2.5].copy()
        
        valid['Inverter_Power_kW'] = valid['kWh_HVAC'] / valid['Runtime_Total_Hr']
        
        # Calculate Delta T (Signed: Outdoor - Indoor)
        # Heating days will be negative, Cooling days positive
        valid['Delta_T_Daily'] = valid['Outdoor Temp (F)'] - valid['Current Temp (F)']
        
        self.daily_combined = valid

    def plot_inverter_curve(self, filename='inverter_profile.png', show=True):
        if self.daily_combined is None or self.daily_combined.empty:
            print("No valid daily overlap data for Inverter Curve.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # --- PLOT 1: Power vs Delta T ---
        # Signed Delta T: Left = Heating, Right = Cooling
        
        heating = self.daily_combined[self.daily_combined['Delta_T_Daily'] < 0]
        cooling = self.daily_combined[self.daily_combined['Delta_T_Daily'] > 0]
        
        if not heating.empty:
            ax1.scatter(heating['Delta_T_Daily'], heating['Inverter_Power_kW'], 
                       c='red', label='Heating Days', alpha=0.7)
            # Quadratic Fit
            if len(heating) > 5:
                try:
                    z = np.polyfit(heating['Delta_T_Daily'], heating['Inverter_Power_kW'], 2)
                    p = np.poly1d(z)
                    xr = np.linspace(heating['Delta_T_Daily'].min(), heating['Delta_T_Daily'].max(), 50)
                    ax1.plot(xr, p(xr), 'r--', alpha=0.5, label='Heating Fit')
                except: pass

        if not cooling.empty:
            ax1.scatter(cooling['Delta_T_Daily'], cooling['Inverter_Power_kW'], 
                       c='blue', label='Cooling Days', alpha=0.7)
            if len(cooling) > 5:
                try:
                    z = np.polyfit(cooling['Delta_T_Daily'], cooling['Inverter_Power_kW'], 2)
                    p = np.poly1d(z)
                    xr = np.linspace(cooling['Delta_T_Daily'].min(), cooling['Delta_T_Daily'].max(), 50)
                    ax1.plot(xr, p(xr), 'b--', alpha=0.5, label='Cooling Fit')

        ax1.set_title(f"Inverter Power Output (Est. Baseload: {self.baseload_kw:.2f} kW)")
        ax1.set_xlabel("Outdoor - Indoor Temp (°F) [Neg=Heating, Pos=Cooling]")
        ax1.set_ylabel("Avg Operating Power (kW)")
        ax1.axvline(0, color='k', lw=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # --- PLOT 2: Raw Energy Sanity Check ---
        # Plotting Total Daily kWh vs Delta T (should be linear-ish V shape)
        # This confirms if the data itself is valid, independent of Runtime division
        
        ax2.scatter(self.daily_combined['Delta_T_Daily'], self.daily_combined['kWh'],
                    c='grey', alpha=0.6, label='Total Daily Energy')
        
        ax2.set_title("Sanity Check: Total Daily Energy vs Temp")
        ax2.set_xlabel("Outdoor - Indoor Temp (°F)")
        ax2.set_ylabel("Total House Energy (kWh/day)")
        ax2.axvline(0, color='k', lw=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._ensure_results_dir()
        if filename: plt.savefig(os.path.join("Results", filename))
        if show: plt.show()
        else: plt.close()

    def generate_report(self, filename='thermal_results.txt'):
        self._ensure_results_dir()
        filepath = os.path.join("Results", filename)
        
        lines = []
        lines.append(f"Thermal Analysis & Inverter Report")
        lines.append(f"==================================")
        lines.append(f"House Size: {self.sq_ft} sq ft")
        if not self.power_df.empty:
             lines.append(f"Baseload Power Est: {self.baseload_kw:.2f} kW")
        
        # Fits
        def print_fit(fit, name):
            if fit:
                slope, intercept = fit
                tau = 1/slope if slope > 0.001 else 0
                lines.append(f"{name}: Slope={slope:.4f}, Tau={tau:.1f}h")
            else:
                lines.append(f"{name}: Insufficient Data")
        
        lines.append("\n-- Envelope Properties --")
        print_fit(self.fit_heating, "Heating (Winter)")
        print_fit(self.fit_cooling, "Cooling (Summer)")
        
        try:
            with open(filepath, 'w') as f: f.write("\n".join(lines))
        except: pass

    def plot_results(self, filename='thermal_fit.png', show=False):
        self._ensure_results_dir()
        filepath = os.path.join("Results", filename)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.coasting_df is not None:
            ax.scatter(self.coasting_df['Delta_T'], self.coasting_df['dT_dt'], 
                       alpha=0.3, color='grey', label='Coasting Data')
            
        if self.fit_heating:
            m, b = self.fit_heating
            x = np.linspace(self.heating_data['Delta_T'].min(), self.heating_data['Delta_T'].max(), 100)
            ax.plot(x, m*x+b, 'r-', label='Heating Fit')
            
        if self.fit_cooling:
            m, b = self.fit_cooling
            x = np.linspace(self.cooling_data['Delta_T'].min(), self.cooling_data['Delta_T'].max(), 100)
            ax.plot(x, m*x+b, 'b-', label='Cooling Fit')
            
        ax.set_title("Thermal Response (Coasting)")
        ax.set_xlabel("Delta T (F)")
        ax.set_ylabel("dT/dt (F/hr)")
        ax.grid(True)
        ax.legend()
        if filename: plt.savefig(filepath)
        if show: plt.show()
        else: plt.close()

    def _ensure_results_dir(self):
        if not os.path.exists("Results"): os.makedirs("Results")