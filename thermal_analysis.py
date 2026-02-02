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
        self.active_filters = {}
        
        # Physics Constants
        self.sq_ft = 1500  # Default
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
                # Namespace-agnostic parsing
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
                        if start is not None and value is not None:
                            if duration and duration < 90000: 
                                readings.append({'Start': start, 'Duration': duration, 'Value': value})
            except Exception as e:
                print(f"Error parsing XML {fp}: {e}")
                
        if readings:
            pdf = pd.DataFrame(readings)
            pdf['Timestamp'] = pd.to_datetime(pdf['Start'], unit='s')
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
        self.active_filters = {
            'night_only': night_only, 
            'buffer_minutes': buffer_minutes,
            'sq_ft': sq_ft
        }
        
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
        # --- 1. Calculate Active Delta T (Weighted by Runtime + Padding) ---
        # Create a boolean mask for "Active or Near Active"
        # 30 minute padding = 6 intervals of 5 mins (since rolling is centered, window=13 covers +/- 30m)
        is_active = (self.df['Cool Stage 1 (sec)'] > 0) | (self.df['Heat Stage 1 (sec)'] > 0)
        padded_active = is_active.rolling(window=13, center=True, min_periods=1).max().fillna(0).astype(bool)
        
        # Extract subset of data that is "Active"
        active_df = self.df.loc[padded_active, ['Current Temp (F)', 'Outdoor Temp (F)']]
        
        # Resample to Daily Mean to get "Active Daily Temps"
        daily_active_temps = active_df.resample('D').mean()
        daily_active_temps.columns = ['Active_Indoor', 'Active_Outdoor']
        
        # --- 2. Aggregate Daily Runtime ---
        cols = ['Cool Stage 1 (sec)', 'Heat Stage 1 (sec)', 'Fan (sec)']
        daily_ecobee = self.df[cols].resample('D').sum()
        
        # Merge Active Temps with Runtime
        daily_ecobee = pd.concat([daily_ecobee, daily_active_temps], axis=1)
        
        daily_ecobee['Runtime_Cool_Hr'] = daily_ecobee['Cool Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Heat_Hr'] = daily_ecobee['Heat Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Total_Hr'] = daily_ecobee['Runtime_Cool_Hr'] + daily_ecobee['Runtime_Heat_Hr']
        
        # --- 3. Merge with Power Data ---
        combined = pd.concat([daily_ecobee, self.power_df], axis=1).dropna()
        
        if combined.empty:
            print("Warning: No date overlap between Thermostat and Power data.")
            return

        # Estimate Baseload
        low_usage_days = combined[combined['Runtime_Total_Hr'] < 0.2]
        
        if not low_usage_days.empty:
            self.baseload_kw = low_usage_days['kWh'].mean() / 24.0
            print(f"Baseload estimated from off-days: {self.baseload_kw:.2f} kW")
        else:
            self.baseload_kw = (combined['kWh'] / 24.0).quantile(0.05)
            print(f"Baseload estimated (quantile): {self.baseload_kw:.2f} kW")

        # Calculate HVAC Specific Energy (Total Daily kWh)
        combined['kWh_HVAC'] = combined['kWh'] - (self.baseload_kw * 24.0)
        combined.loc[combined['kWh_HVAC'] < 0, 'kWh_HVAC'] = 0 
        
        # --- 4. Calculate Active Delta T ---
        # Fallback for days with no runtime (Active Temps might be NaN): Use 24h avg? 
        # Actually, if no runtime, we filter them out anyway.
        combined['Active_Delta_T'] = combined['Active_Outdoor'] - combined['Active_Indoor']
        
        # Calculate Average Inverter Power (kW)
        # Avoid division by zero
        combined['Inverter_Power_kW'] = combined['kWh_HVAC'] / combined['Runtime_Total_Hr']
        
        # --- 5. Classification (Heating vs Cooling Day) ---
        # Based on Dominant Runtime, NOT Delta T
        combined['Mode'] = 'Mixed'
        combined.loc[combined['Runtime_Cool_Hr'] > combined['Runtime_Heat_Hr'], 'Mode'] = 'Cooling'
        combined.loc[combined['Runtime_Heat_Hr'] >= combined['Runtime_Cool_Hr'], 'Mode'] = 'Heating'
        
        # Filter for valid data points (significant runtime to get a stable kW reading)
        # For the Energy Plot (kWh), we can be permissive (>0.1h).
        # For the Power Plot (kW), we need stability (>1.0h).
        self.daily_combined = combined[combined['Runtime_Total_Hr'] > 0.1].copy()

    def plot_inverter_curve(self, filename='inverter_profile.png', show=True):
        if self.daily_combined is None or self.daily_combined.empty:
            print("No valid daily overlap data for Inverter Curve.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Data Subsets
        heating = self.daily_combined[self.daily_combined['Mode'] == 'Heating']
        cooling = self.daily_combined[self.daily_combined['Mode'] == 'Cooling']
        
        # --- PLOT 1: Daily Energy (kWh) vs Active Delta T ---
        if not heating.empty:
            x = heating['Active_Delta_T']
            y = heating['kWh_HVAC']
            ax1.scatter(x, y, c='red', label='Heating Days', alpha=0.6)
            if len(heating) > 5:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                xr = np.linspace(x.min(), x.max(), 50)
                ax1.plot(xr, p(xr), 'r--', label=f'Heat: {z[0]:.2f} kWh/deg + {z[1]:.1f}')

        if not cooling.empty:
            x = cooling['Active_Delta_T']
            y = cooling['kWh_HVAC']
            ax1.scatter(x, y, c='blue', label='Cooling Days', alpha=0.6)
            if len(cooling) > 5:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                xr = np.linspace(x.min(), x.max(), 50)
                ax1.plot(xr, p(xr), 'b--', label=f'Cool: {z[0]:.2f} kWh/deg + {z[1]:.1f}')

        ax1.set_title(f"Daily HVAC Energy Cost (Baseload: {self.baseload_kw:.2f} kW)")
        ax1.set_xlabel("Active Delta T (°F) [Outdoor - Indoor during Runtime]")
        ax1.set_ylabel("Daily HVAC Energy (kWh)")
        ax1.axvline(0, color='k', lw=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # --- PLOT 2: Inverter Power (kW) vs Active Delta T ---
        # Filter for stability (Runtime > 1h) to avoid asymptotic noise
        h_stable = heating[heating['Runtime_Total_Hr'] > 1.0]
        c_stable = cooling[cooling['Runtime_Total_Hr'] > 1.0]

        if not h_stable.empty:
            ax2.scatter(h_stable['Active_Delta_T'], h_stable['Inverter_Power_kW'], c='red', alpha=0.6)
            # Quadratic fit for Power Curve
            if len(h_stable) > 5:
                try:
                    z = np.polyfit(h_stable['Active_Delta_T'], h_stable['Inverter_Power_kW'], 2)
                    p = np.poly1d(z)
                    xr = np.linspace(h_stable['Active_Delta_T'].min(), h_stable['Active_Delta_T'].max(), 50)
                    ax2.plot(xr, p(xr), 'r--', alpha=0.5)
                except: pass

        if not c_stable.empty:
            ax2.scatter(c_stable['Active_Delta_T'], c_stable['Inverter_Power_kW'], c='blue', alpha=0.6)
            if len(c_stable) > 5:
                try:
                    z = np.polyfit(c_stable['Active_Delta_T'], c_stable['Inverter_Power_kW'], 2)
                    p = np.poly1d(z)
                    xr = np.linspace(c_stable['Active_Delta_T'].min(), c_stable['Active_Delta_T'].max(), 50)
                    ax2.plot(xr, p(xr), 'b--', alpha=0.5)
                except: pass

        ax2.set_title("Inverter Performance Profile (Avg kW during Operation)")
        ax2.set_xlabel("Active Delta T (°F) [Outdoor - Indoor]")
        ax2.set_ylabel("Avg Power Output (kW)")
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
        lines.append(f"Thermal Analysis & Energy Report")
        lines.append(f"================================")
        lines.append(f"Analysis Range: {self.df.index.min()} to {self.df.index.max()}")
        lines.append(f"House Size: {self.sq_ft} sq ft")
        if not self.power_df.empty:
             lines.append(f"Baseload Power Est: {self.baseload_kw:.2f} kW")
        
        lines.append(f"Filters: NightOnly={self.active_filters.get('night_only')}, Buffer={self.active_filters.get('buffer_minutes')}m")
        
        # Fits
        def print_fit(fit, name):
            if fit:
                slope, intercept = fit
                tau = 1/slope if slope > 0.001 else 0
                lines.append(f"{name}: Slope={slope:.4f}, Tau={tau:.1f}h")
            else:
                lines.append(f"{name}: Insufficient Data")
        
        lines.append("\n-- Envelope Properties (Coasting) --")
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
            
        # Plot fits
        tau_h_str = "N/A"
        tau_c_str = "N/A"
        
        if self.fit_heating:
            m, b = self.fit_heating
            if abs(m) > 0.001:
                tau_h_str = f"{1/m:.1f}h"
                x = np.linspace(self.heating_data['Delta_T'].min(), self.heating_data['Delta_T'].max(), 100)
                ax.plot(x, m*x+b, 'r-', lw=2, label=f'Heating Fit (Tau={tau_h_str})')
            
        if self.fit_cooling:
            m, b = self.fit_cooling
            if abs(m) > 0.001:
                tau_c_str = f"{1/m:.1f}h"
                x = np.linspace(self.cooling_data['Delta_T'].min(), self.cooling_data['Delta_T'].max(), 100)
                ax.plot(x, m*x+b, 'b-', lw=2, label=f'Cooling Fit (Tau={tau_c_str})')
        
        # Subtitle
        start_str = self.df.index.min().strftime('%Y-%m-%d')
        end_str = self.df.index.max().strftime('%Y-%m-%d')
        filter_str = f"Filters: [Night={self.active_filters.get('night_only')} | Buffer={self.active_filters.get('buffer_minutes')}m]"
        subtitle = f"Range: {start_str} to {end_str}\n{filter_str}\nTau: Heat={tau_h_str}, Cool={tau_c_str}"

        ax.set_title(f"Thermal Response: Indoor Rate of Change vs. Delta T\n{subtitle}", fontsize=10)
        ax.set_xlabel("Outdoor - Indoor Temperature (°F)")
        ax.set_ylabel("Indoor Rate of Change (°F/hr)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        if filename: plt.savefig(filepath)
        if show: plt.show()
        else: plt.close()

    def _ensure_results_dir(self):
        if not os.path.exists("Results"): os.makedirs("Results")