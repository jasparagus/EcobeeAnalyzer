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
        cols = ['Cool Stage 1 (sec)', 'Heat Stage 1 (sec)', 'Fan (sec)']
        daily_ecobee = self.df[cols].resample('D').sum()
        
        daily_temps = self.df[['Current Temp (F)', 'Outdoor Temp (F)']].resample('D').mean()
        
        daily_ecobee['Runtime_Cool_Hr'] = daily_ecobee['Cool Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Heat_Hr'] = daily_ecobee['Heat Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Total_Hr'] = daily_ecobee['Runtime_Cool_Hr'] + daily_ecobee['Runtime_Heat_Hr']
        
        combined = pd.concat([daily_ecobee, daily_temps, self.power_df], axis=1).dropna()
        
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
        combined.loc[combined['kWh_HVAC'] < 0, 'kWh_HVAC'] = 0 # Clamp negative
        
        # Calculate Delta T (Signed)
        combined['Delta_T_Daily'] = combined['Outdoor Temp (F)'] - combined['Current Temp (F)']
        
        # Filter for valid days (some HVAC usage, positive energy)
        # We allow small runtimes now since we are plotting Energy, not Rate
        valid = combined[combined['Runtime_Total_Hr'] > 0.1].copy()
        
        self.daily_combined = valid

    def plot_inverter_curve(self, filename='inverter_profile.png', show=True):
        if self.daily_combined is None or self.daily_combined.empty:
            print("No valid daily overlap data for Inverter Curve.")
            return

        fig, ax = plt.subplots(figsize=(10, 7))
        
        heating = self.daily_combined[self.daily_combined['Delta_T_Daily'] < 0]
        cooling = self.daily_combined[self.daily_combined['Delta_T_Daily'] > 0]
        
        # Plot Heating
        if not heating.empty:
            ax.scatter(heating['Delta_T_Daily'], heating['kWh_HVAC'], 
                       c='red', label='Heating Days', alpha=0.6, edgecolors='none')
            # Linear Fit for Energy vs Delta T
            if len(heating) > 5:
                try:
                    z = np.polyfit(heating['Delta_T_Daily'], heating['kWh_HVAC'], 1)
                    p = np.poly1d(z)
                    xr = np.linspace(heating['Delta_T_Daily'].min(), heating['Delta_T_Daily'].max(), 50)
                    ax.plot(xr, p(xr), 'r--', lw=2, label=f'Heating Trend')
                except: pass

        # Plot Cooling
        if not cooling.empty:
            ax.scatter(cooling['Delta_T_Daily'], cooling['kWh_HVAC'], 
                       c='blue', label='Cooling Days', alpha=0.6, edgecolors='none')
            # Linear Fit for Energy vs Delta T
            if len(cooling) > 5:
                try:
                    z = np.polyfit(cooling['Delta_T_Daily'], cooling['kWh_HVAC'], 1)
                    p = np.poly1d(z)
                    xr = np.linspace(cooling['Delta_T_Daily'].min(), cooling['Delta_T_Daily'].max(), 50)
                    ax.plot(xr, p(xr), 'b--', lw=2, label=f'Cooling Trend')
                except: pass

        # Build Subtitle
        start_str = self.df.index.min().strftime('%Y-%m-%d')
        end_str = self.df.index.max().strftime('%Y-%m-%d')
        subtitle = f"Range: {start_str} to {end_str}\nEst. Baseload: {self.baseload_kw:.2f} kW (Subtracted)"
        
        ax.set_title(f"Daily HVAC Energy vs. Temperature Delta\n{subtitle}", fontsize=11)
        ax.set_xlabel("Outdoor - Indoor Temp (°F) [Neg=Heating, Pos=Cooling]")
        ax.set_ylabel("Total Daily HVAC Energy (kWh)")
        ax.axvline(0, color='k', lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
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
            if m > 0.001:
                tau_h_str = f"{1/m:.1f}h"
                x = np.linspace(self.heating_data['Delta_T'].min(), self.heating_data['Delta_T'].max(), 100)
                ax.plot(x, m*x+b, 'r-', lw=2, label=f'Heating Fit (Tau={tau_h_str})')
            
        if self.fit_cooling:
            m, b = self.fit_cooling
            if m > 0.001:
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