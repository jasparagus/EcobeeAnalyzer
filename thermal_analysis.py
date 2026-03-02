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
            
            # Intelligent Resolution Selection
            hourly_mask = pdf['Duration'] <= 4000
            daily_mask = pdf['Duration'] >= 80000
            
            if hourly_mask.any():
                print(f"Found {hourly_mask.sum()} hourly intervals. Utilizing hourly resolution.")
                pdf = pdf[hourly_mask]
                self.power_resolution = 'h'
                self.power_df = pdf.set_index('Timestamp')['kWh'].resample('h').sum().to_frame()
            elif daily_mask.any():
                print(f"Found {daily_mask.sum()} daily intervals. Utilizing daily resolution.")
                pdf = pdf[daily_mask]
                self.power_resolution = 'D'
                self.power_df = pdf.set_index('Timestamp')['kWh'].resample('D').sum().to_frame()
            else:
                 self.power_resolution = 'Unknown'
                 self.power_df = pdf.set_index('Timestamp')['kWh'].to_frame()
                 
            self.power_df = self.power_df.sort_index()
            print(f"Loaded {len(self.power_df)} valid power readings at resolution: {self.power_resolution}.")
        else:
            print("No valid daily or hourly readings found in XML.")

    def analyze(self, night_only=False, buffer_minutes=30, sq_ft=1500, timezone='US/Pacific'):
        if self.df is None: self.load_data()
        self.sq_ft = sq_ft
        self.active_filters = {
            'night_only': night_only, 
            'buffer_minutes': buffer_minutes,
            'sq_ft': sq_ft,
            'timezone': timezone
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
            self._analyze_inverter_performance(timezone)

    def _analyze_inverter_performance(self, timezone):
        """Correlates Daily Ecobee Runtime with Daily Green Button Energy."""
        print(f"Aligning Ecobee Data to UTC using timezone: {timezone}")
        
        # Create a UTC-aligned copy of the dataframe for Energy aggregation
        try:
            utc_df = self.df.copy()
            # Localize to user timezone, then convert to UTC. Use 'NaT' for ambiguous times (DST fall-back)
            utc_df.index = utc_df.index.tz_localize(timezone, ambiguous='NaT', nonexistent='shift_forward').tz_convert('UTC')
            utc_df = utc_df[utc_df.index.notnull()] # Drop the ambiguous hour
            # Remove TZ info to match PowerDF (which is usually Naive UTC)
            utc_df.index = utc_df.index.tz_localize(None) 
        except Exception as e:
            print(f"Timezone conversion failed: {e}. Falling back to default alignment.")
            utc_df = self.df.copy()

        freq_str = getattr(self, 'power_resolution', 'D')
        if freq_str not in ['D', 'h']: freq_str = 'D'

        # --- 1. Calculate Active Delta T (Weighted by Runtime + Padding) ---
        is_active = (utc_df['Cool Stage 1 (sec)'] > 0) | (utc_df['Heat Stage 1 (sec)'] > 0)
        padded_active = is_active.rolling(window=5, center=True, min_periods=1).max().fillna(0).astype(bool)
        
        active_df = utc_df.loc[padded_active, ['Current Temp (F)', 'Outdoor Temp (F)']]
        active_temps = active_df.resample(freq_str).mean()
        active_temps.columns = ['Active_Indoor', 'Active_Outdoor']
        
        # --- 2. Aggregate Runtime ---
        cols = ['Cool Stage 1 (sec)', 'Heat Stage 1 (sec)', 'Fan (sec)']
        ecobee_agg = utc_df[cols].resample(freq_str).sum()
        
        ecobee_agg = pd.concat([ecobee_agg, active_temps], axis=1)
        
        ecobee_agg['Runtime_Cool_Hr'] = ecobee_agg['Cool Stage 1 (sec)'] / 3600.0
        ecobee_agg['Runtime_Heat_Hr'] = ecobee_agg['Heat Stage 1 (sec)'] / 3600.0
        ecobee_agg['Runtime_Total_Hr'] = ecobee_agg['Runtime_Cool_Hr'] + ecobee_agg['Runtime_Heat_Hr']
        
        # --- 3. Merge with Power Data ---
        combined = pd.concat([ecobee_agg, self.power_df], axis=1).dropna()
        
        if combined.empty:
            print("Warning: No date overlap between Thermostat and Power data.")
            return

        # --- NEW BASELOAD LOGIC: Linear Regression ---
        x = combined['Runtime_Total_Hr']
        y = combined['kWh']
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        reg_baseload_kw = intercept if freq_str == 'h' else intercept / 24.0
        print(f"Baseload Regression: Intercept={intercept:.2f} kWh/{freq_str} -> {reg_baseload_kw:.2f} kW (R2={r_value**2:.3f})")
        
        fallback_div = 1.0 if freq_str == 'h' else 24.0
        if reg_baseload_kw > 0.1 and reg_baseload_kw < 10.0:
            self.baseload_kw = reg_baseload_kw
            print(f"Using Regression Baseload: {self.baseload_kw:.2f} kW")
        else:
            self.baseload_kw = (combined['kWh'] / fallback_div).quantile(0.05)
            print(f"Regression yielded invalid baseload. Falling back to quantile: {self.baseload_kw:.2f} kW")

        combined['kWh_HVAC'] = combined['kWh'] - (self.baseload_kw * fallback_div)
        combined['Active_Delta_T'] = combined['Active_Outdoor'] - combined['Active_Indoor']
        combined['active_dt_abs'] = combined['Active_Delta_T'].abs()
        combined['Inverter_Power_kW'] = combined['kWh_HVAC'] / combined['Runtime_Total_Hr']
        
        combined['Mode'] = 'Mixed'
        combined.loc[combined['Runtime_Cool_Hr'] > combined['Runtime_Heat_Hr'], 'Mode'] = 'Cooling'
        combined.loc[combined['Runtime_Heat_Hr'] >= combined['Runtime_Cool_Hr'], 'Mode'] = 'Heating'
        
        self.interval_combined = combined[combined['Runtime_Total_Hr'] > (0.01 if freq_str == 'h' else 0.1)].copy()

        # Build daily_combined for Cost to Condition plots
        daily_ecobee = utc_df[cols].resample('D').sum()
        daily_temps = active_df.resample('D').mean()
        daily_temps.columns = ['Active_Indoor', 'Active_Outdoor']
        daily_ecobee = pd.concat([daily_ecobee, daily_temps], axis=1)
        daily_ecobee['Runtime_Cool_Hr'] = daily_ecobee['Cool Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Heat_Hr'] = daily_ecobee['Heat Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Total_Hr'] = daily_ecobee['Runtime_Cool_Hr'] + daily_ecobee['Runtime_Heat_Hr']
        
        daily_power = self.power_df['kWh'].resample('D').sum() if freq_str == 'h' else self.power_df['kWh']
        daily_combined = pd.concat([daily_ecobee, daily_power], axis=1).dropna()
        if not daily_combined.empty:
            daily_combined['Active_Delta_T'] = daily_combined['Active_Outdoor'] - daily_combined['Active_Indoor']
            daily_combined['Mode'] = 'Mixed'
            daily_combined.loc[daily_combined['Runtime_Cool_Hr'] > daily_combined['Runtime_Heat_Hr'], 'Mode'] = 'Cooling'
            daily_combined.loc[daily_combined['Runtime_Heat_Hr'] >= daily_combined['Runtime_Cool_Hr'], 'Mode'] = 'Heating'
            self.daily_combined = daily_combined[daily_combined['Runtime_Total_Hr'] > 0.1].copy()
        else:
            self.daily_combined = pd.DataFrame()

    def plot_inverter_curve(self, filename='inverter_profile.png', show=True):
        comb = getattr(self, 'interval_combined', getattr(self, 'daily_combined', None))
        if comb is None or comb.empty: return

        fig, ax = plt.subplots(figsize=(10, 7))
        
        heating = comb[comb['Mode'] == 'Heating']
        cooling = comb[comb['Mode'] == 'Cooling']

        # Filter for stability (Runtime > min_runtime) to avoid asymptotic noise
        min_runtime = 0.5 if getattr(self, 'power_resolution', 'D') == 'h' else 1.0
        h_stable = heating[heating['Runtime_Total_Hr'] > min_runtime]
        c_stable = cooling[cooling['Runtime_Total_Hr'] > min_runtime]

        def plot_series(data, color, label_prefix):
            if data.empty: return
            x = data['Active_Delta_T']
            y = data['Inverter_Power_kW']
            
            ax.scatter(x, y, c=color, alpha=0.6, label=f'{label_prefix} Data')
            
            # Quadratic fit
            if len(data) > 5:
                try:
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    xr = np.linspace(x.min(), x.max(), 50)
                    ax.plot(xr, p(xr), color=color, linestyle='--', alpha=0.8, lw=2)
                    
                    # Annotations
                    min_p = y.min()
                    max_p = y.max()
                    eq_str = f"{z[0]:.4f}x² + {z[1]:.3f}x + {z[2]:.2f}"
                    
                    # Add stats box
                    text_str = (f"{label_prefix} Performance:\n"
                                f"Range: {min_p:.2f} - {max_p:.2f} kW\n"
                                f"Fit: {eq_str}")
                    
                    # Position box based on heating/cooling (Heating=Left/Top, Cooling=Right/Bottom roughly)
                    yloc = 0.85 if label_prefix == 'Heating' else 0.15
                    xloc = 0.05 if label_prefix == 'Heating' else 0.65
                    
                    ax.text(xloc, yloc, text_str, transform=ax.transAxes, 
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color))
                except: pass

        plot_series(h_stable, 'red', 'Heating')
        plot_series(c_stable, 'blue', 'Cooling')

        ax.set_title("Inverter Modulation Profile (System Capacity)\nHow the system ramps power to meet demand", fontsize=11)
        ax.set_xlabel("Active Delta T (°F) [Outdoor - Indoor]", fontsize=10)
        ax.set_ylabel("Average Power Output (kW)", fontsize=10)
        ax.axvline(0, color='k', lw=0.5)
        ax.grid(True, alpha=0.3)
        
        # Interpretation Guide
        guide_text = ("Interpretation:\n"
                      "• Flat Line: Single/Two-Stage System (Fixed Capacity)\n"
                      "• Sloped/Curved: Inverter System (Modulating Capacity)\n"
                      "• Scatter: Unit cycling or defrost events")
        ax.text(0.98, 0.98, guide_text, transform=ax.transAxes, 
                horizontalalignment='right', verticalalignment='top', fontsize=8, color='#555555',
                bbox=dict(boxstyle='square', facecolor='#f0f0f0', alpha=1.0))

        ax.legend(loc='lower left')
        
        plt.tight_layout()
        self._ensure_results_dir()
        if filename: plt.savefig(os.path.join("Results", filename))
        if show: plt.show()
        else: plt.close()

    def plot_energy_profile(self, filename='total_energy_profile.png', show=True):
        """Standard Output: Total Energy Profile (Cost to Condition) & Baseload Fit."""
        comb_ref = getattr(self, 'interval_combined', getattr(self, 'daily_combined', None))
        if comb_ref is None or comb_ref.empty: return
        freq_str = getattr(self, 'power_resolution', 'D')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- PLOT 1: Baseload Regression (Runtime Method) ---
        x = comb_ref['Runtime_Total_Hr']
        y = comb_ref['kWh']
        ax1.scatter(x, y, alpha=0.5, c='purple', label=f'Data ({freq_str})')
        
        slope, intercept, r_val, p, err = stats.linregress(x, y)
        xr = np.linspace(0, x.max(), 50)
        ax1.plot(xr, slope*xr + intercept, 'k--', lw=2, label=f'Fit (R2={r_val**2:.2f})')
        
        # Annotate
        ax1.axhline(intercept, color='green', linestyle=':', alpha=0.5, label='Projected Baseload')
        b_kw = intercept if freq_str == 'h' else intercept / 24.0
        ax1.text(0.05, 0.95, 
                 f"Slope (Avg HVAC Power): {slope:.2f} kW\n"
                 f"Intercept (Baseload): {intercept:.1f} kWh/{freq_str}\n"
                 f"(= {b_kw:.2f} kW avg)", 
                 transform=ax1.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax1.set_title("Method 1: Runtime Regression\n(Determines Baseload for Subtraction)")
        ax1.set_xlabel(f"Total HVAC Runtime (Hours/{freq_str})")
        ax1.set_ylabel(f"Total Energy (kWh/{freq_str})")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right')

        # --- PLOT 2: Cost to Condition (Temperature Method) - Always Daily ---
        daily_comb = getattr(self, 'daily_combined', None)
        if daily_comb is not None and not daily_comb.empty:
            heat = daily_comb[daily_comb['Mode'] == 'Heating']
            cool = daily_comb[daily_comb['Mode'] == 'Cooling']
            
            # Plot Baseload Reference 
            d_base = self.baseload_kw * 24.0
            ax2.axhline(d_base, color='green', linestyle=':', alpha=0.5, label=f'Baseload Ref ({d_base:.1f} kWh)')
        
        if not heat.empty:
            ax2.scatter(heat['Active_Delta_T'], heat['kWh'], c='red', alpha=0.6, label='Heating Days')
            if len(heat) > 5:
                res = stats.linregress(heat['Active_Delta_T'], heat['kWh'])
                xr = np.linspace(heat['Active_Delta_T'].min(), heat['Active_Delta_T'].max(), 50)
                ax2.plot(xr, res.slope*xr + res.intercept, 'r--', label='Heating Fit')
                # Annotate Slope
                ax2.text(0.05, 0.85, 
                         f"Heating Slope: {res.slope:.2f} kWh/°F\n" + 
                         f"(Cost to Condition)", 
                         transform=ax2.transAxes, color='red', fontsize=9, fontweight='bold')

        if not cool.empty:
            ax2.scatter(cool['Active_Delta_T'], cool['kWh'], c='blue', alpha=0.6, label='Cooling Days')
            if len(cool) > 5:
                res = stats.linregress(cool['Active_Delta_T'], cool['kWh'])
                xr = np.linspace(cool['Active_Delta_T'].min(), cool['Active_Delta_T'].max(), 50)
                ax2.plot(xr, res.slope*xr + res.intercept, 'b--', label='Cooling Fit')
                # Annotate Slope
                ax2.text(0.05, 0.15, 
                         f"Cooling Slope: {res.slope:.2f} kWh/°F\n" + 
                         f"(Cost to Condition)", 
                         transform=ax2.transAxes, color='blue', fontsize=9, fontweight='bold')
                
        ax2.set_title("Method 2: Temperature Profile\n(Raw Total Energy vs Delta T)")
        ax2.set_xlabel("Outdoor - Indoor Delta T (°F)")
        ax2.set_ylabel("Total Daily Energy (kWh)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
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
             lines.append(f"Baseload Power (Regressed): {self.baseload_kw:.2f} kW")
             daily_comb = getattr(self, 'daily_combined', None)
             if daily_comb is not None:
                 heat = daily_comb[daily_comb['Mode'] == 'Heating']
                 if not heat.empty and len(heat) > 5:
                     s, i, r, _, _ = stats.linregress(heat['Active_Delta_T'], heat['kWh'])
                     lines.append(f"Heating Total Cost Slope: {s:.3f} kWh/degF (R2={r**2:.2f})")
        
        lines.append(f"Filters: NightOnly={self.active_filters.get('night_only')}, Buffer={self.active_filters.get('buffer_minutes')}m")
        
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