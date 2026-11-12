import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import xml.etree.ElementTree as ET

class ThermalAnalyzer:
    """
    Analyzes Ecobee data to determine building envelope properties (UA, Time Constant,
    Balance Point, Sizing Limits) and estimates HVAC power usage using Reverse Calorimetry
    and Green Button Data.

    Time Convention:
    - Ecobee CSV data is stored in naive local time.
    - Green Button XML data uses UTC Unix epoch timestamps.
    """
    
    def __init__(self, file_paths=None, dataframe=None):
        self.file_paths = file_paths
        self.df = dataframe
        self.power_df = pd.DataFrame() # Store Green Button Data
        self.daily_power_df = pd.DataFrame()
        
        self.coasting_df = None
        self.cooling_data = None
        self.heating_data = None
        self.fit_cooling = None
        self.fit_heating = None
        self.fit_cooling_stats = None
        self.fit_heating_stats = None
        self.daily_ecobee = pd.DataFrame()
        self.daily_combined = None
        self.interval_combined = None
        self.baseload_reg_df = None
        self.baseload_reg_stats = None
        self.active_filters = {}
        
        # Physics Constants & Derived Envelope Estimates
        self.sq_ft = 1500  # Default
        self.C_est = 0     
        self.UA_est = 0    
        self.UA_heating = None
        self.UA_cooling = None
        self.balance_point_heating = None
        self.balance_point_cooling = None
        self.design_temp_heating = None
        self.design_temp_cooling = None
        self.baseload_kw = 0.5 # Default estimate
        self.baseload_method = "default"
        
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
        """
        Parses Green Button XML files and merges them.
        Note: Green Button <start> values are UTC Unix epoch timestamps (seconds).
        """
        readings = []
        unique_paths = list(dict.fromkeys(file_paths))
        for fp in unique_paths:
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
                            if child.tag.endswith('start') and child.text:
                                start = int(float(child.text.strip()))
                            elif child.tag.endswith('duration') and child.text:
                                duration = int(float(child.text.strip()))
                            elif child.tag.endswith('value') and child.text:
                                value = float(child.text.strip())
                        
                        # FILTER: Accept hourly or daily readings (< 90000s)
                        if start is not None and value is not None:
                            if duration and duration < 90000: 
                                readings.append({'Start': start, 'Duration': duration, 'Value': value})
            except Exception as e:
                print(f"Error parsing XML {fp}: {e}")
                
        if readings:
            pdf = pd.DataFrame(readings)
            # Deduplicate overlapping readings from multiple or repeated XML loads
            pdf = pdf.drop_duplicates(subset=['Start', 'Duration'], keep='first')
            pdf['Timestamp'] = pd.to_datetime(pdf['Start'], unit='s')
            pdf['kWh'] = pdf['Value'] 
            
            # Intelligent Resolution Selection (preserving non-overlapping daily & hourly data)
            hourly_mask = pdf['Duration'] <= 4000
            daily_mask = pdf['Duration'] >= 80000
            
            hourly_df = pd.DataFrame()
            daily_only_df = pd.DataFrame()
            
            if hourly_mask.any():
                hourly_pdf = pdf[hourly_mask].drop_duplicates(subset=['Timestamp']).set_index('Timestamp').sort_index()
                hourly_df = hourly_pdf['kWh'].resample('h').sum(min_count=1).dropna().to_frame()
                
            if daily_mask.any():
                daily_pdf = pdf[daily_mask].drop_duplicates(subset=['Timestamp']).set_index('Timestamp').sort_index()
                daily_only_df = daily_pdf['kWh'].resample('D').sum(min_count=1).dropna().to_frame()

            # Build complete daily power series (preferring complete hourly days where available, keeping daily-only dates)
            if not hourly_df.empty and not daily_only_df.empty:
                h_counts = hourly_df['kWh'].resample('D').count()
                min_hrs = 20 if h_counts.max() >= 20 else 1
                hourly_as_daily = hourly_df['kWh'].resample('D').sum(min_count=min_hrs).dropna().to_frame()
                self.daily_power_df = hourly_as_daily.combine_first(daily_only_df).sort_index()
                unmatched_daily = daily_only_df.index.difference(hourly_as_daily.index)
                print(
                    f"Found {len(hourly_df)} hourly intervals and {len(unmatched_daily)} additional daily-only dates. "
                    f"Using hourly resolution for modulation profiling and combined daily series ({len(self.daily_power_df)} days) for daily energy analysis."
                )
                self.power_resolution = 'h'
                self.power_df = hourly_df
            elif not hourly_df.empty:
                print(f"Found {hourly_mask.sum()} hourly intervals. Utilizing hourly resolution.")
                self.power_resolution = 'h'
                self.power_df = hourly_df
                h_counts = hourly_df['kWh'].resample('D').count()
                min_hrs = 20 if h_counts.max() >= 20 else 1
                self.daily_power_df = hourly_df['kWh'].resample('D').sum(min_count=min_hrs).dropna().to_frame()
            elif not daily_only_df.empty:
                print(f"Found {daily_mask.sum()} daily intervals. Utilizing daily resolution.")
                self.power_resolution = 'D'
                self.power_df = daily_only_df
                self.daily_power_df = daily_only_df.copy()
            else:
                self.power_resolution = 'Unknown'
                self.power_df = pdf.drop_duplicates(subset=['Timestamp']).set_index('Timestamp')['kWh'].to_frame()
                self.daily_power_df = self.power_df['kWh'].resample('D').sum(min_count=1).dropna().to_frame()
                 
            self.power_df = self.power_df.sort_index()
            print(f"Loaded {len(self.power_df)} valid power readings at resolution: {self.power_resolution}.")
        else:
            print("No valid daily or hourly readings found in XML.")

    @staticmethod
    def _classify_mode(df_in):
        """
        Classifies intervals/days into Heating, Cooling, Mixed, or Off.
        Preserves 'Mixed' when both heating and cooling have significant runtime
        and neither accounts for >= 80% of total runtime.
        """
        mode = pd.Series('Mixed', index=df_in.index, dtype='object')
        cool_hr = df_in['Runtime_Cool_Hr']
        heat_hr = df_in['Runtime_Heat_Hr']
        tot_hr = df_in['Runtime_Total_Hr']
        safe_tot = tot_hr.replace(0, np.nan)

        is_off = tot_hr <= 0
        is_cool = (cool_hr > heat_hr) & (tot_hr > 0) & ((heat_hr < 0.25) | ((cool_hr / safe_tot) >= 0.8))
        is_heat = (heat_hr >= cool_hr) & (tot_hr > 0) & ((cool_hr < 0.25) | ((heat_hr / safe_tot) >= 0.8))

        mode.loc[is_off] = 'Off'
        mode.loc[is_cool] = 'Cooling'
        mode.loc[is_heat] = 'Heating'
        return mode

    @staticmethod
    def _plot_linear_fit_with_ci(ax, x, y, color, linestyle='--', lw=2, label=None, x_range=None, ci_alpha=0.12):
        """
        Plots a linear regression line along with a subtle, non-distracting 95% confidence band.
        Returns the scipy.stats.linregress result.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        res = stats.linregress(x_arr, y_arr)
        if x_range is not None:
            xr = np.linspace(x_range[0], x_range[1], 100)
        else:
            xr = np.linspace(x_arr.min(), x_arr.max(), 100)
        yr = res.slope * xr + res.intercept
        ax.plot(xr, yr, color=color, linestyle=linestyle, lw=lw, label=label)

        n = len(x_arr)
        if n > 2:
            x_mean = np.mean(x_arr)
            ssx = np.sum((x_arr - x_mean) ** 2)
            if ssx > 0:
                y_hat = res.slope * x_arr + res.intercept
                s_err = np.sqrt(np.sum((y_arr - y_hat) ** 2) / (n - 2))
                t_val = stats.t.ppf(0.975, df=n - 2)
                ci = t_val * s_err * np.sqrt((1.0 / n) + ((xr - x_mean) ** 2) / ssx)
                ax.fill_between(xr, yr - ci, yr + ci, color=color, alpha=ci_alpha, edgecolor='none')
        return res

    def analyze(self, night_only=False, buffer_minutes=30, sq_ft=1500, timezone='US/Pacific'):
        if self.df is None: self.load_data()
        self.sq_ft = sq_ft
        self.active_filters = {
            'night_only': night_only, 
            'buffer_minutes': buffer_minutes,
            'sq_ft': sq_ft,
            'timezone': timezone
        }
        
        # 1. Estimate Thermal Mass (C in BTU/°F)
        self.C_est = self.sq_ft * 3.5 
        self.UA_est = 0
        self.UA_heating = None
        self.UA_cooling = None
        self.fit_heating = None
        self.fit_cooling = None
        self.fit_heating_stats = None
        self.fit_cooling_stats = None

        # 2. Pre-process Ecobee Data (stored in naive local time)
        self.df = self.df.sort_index()
        required_cols = ['Current Temp (F)', 'Outdoor Temp (F)', 'Cool Stage 1 (sec)', 'Heat Stage 1 (sec)']
        if not all(col in self.df.columns for col in required_cols):
             raise ValueError(f"Missing columns: {required_cols}")

        # Detect whether thermal data has sub-daily (5m - 1h) resolution required for dT/dt coasting analysis
        if len(self.df.index) > 1:
            median_step_sec = self.df.index.to_series().diff().dropna().dt.total_seconds().median()
        else:
            median_step_sec = 86400.0
        self.has_subhourly_thermal = bool(median_step_sec <= 7200)

        if self.has_subhourly_thermal:
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
            
            # Calculate Fits (Slope = UA/C) and Derive UA (BTU/hr·°F)
            if len(self.heating_data) > 20:
                res_h = stats.linregress(self.heating_data['Delta_T'], self.heating_data['dT_dt'])
                self.fit_heating_stats = res_h
                self.fit_heating = (res_h.slope, res_h.intercept)
                if res_h.slope > 0.001:
                    self.UA_heating = self.C_est * res_h.slope
                    self.UA_est = self.UA_heating

            if len(self.cooling_data) > 20:
                res_c = stats.linregress(self.cooling_data['Delta_T'], self.cooling_data['dT_dt'])
                self.fit_cooling_stats = res_c
                self.fit_cooling = (res_c.slope, res_c.intercept)
                if res_c.slope > 0.001:
                    self.UA_cooling = self.C_est * res_c.slope
                    if self.UA_est == 0:
                        self.UA_est = self.UA_cooling
        else:
            print("Note: Sub-hourly thermal data unavailable (per-day resolution detected). Skipping coasting Tau/UA derivative analysis.")
            self.coasting_df = pd.DataFrame()
            self.cooling_data = pd.DataFrame()
            self.heating_data = pd.DataFrame()

        # 2b. Build Daily Ecobee Summary for Balance Point & Sizing Limit Analysis
        self._build_daily_ecobee_summary()

        # 3. Inverter Power Analysis (Daily / Hourly)
        if not self.power_df.empty:
            self._analyze_inverter_performance(timezone)

    def _build_daily_ecobee_summary(self):
        """Aggregates local-time Ecobee data into daily metrics for Balance Point & Sizing plots."""
        daily_temps_all = self.df[['Current Temp (F)', 'Outdoor Temp (F)']].resample('D').mean()
        daily_temps_all.columns = ['Indoor_Mean', 'Outdoor_Mean']

        is_active = (self.df['Cool Stage 1 (sec)'] > 0) | (self.df['Heat Stage 1 (sec)'] > 0)
        padded_active = is_active.rolling(window=5, center=True, min_periods=1).max().fillna(0).astype(bool)
        active_df = self.df.loc[padded_active, ['Current Temp (F)', 'Outdoor Temp (F)']].resample('D').mean()
        active_df.columns = ['Active_Indoor', 'Active_Outdoor']

        daily_run = self.df[['Cool Stage 1 (sec)', 'Heat Stage 1 (sec)']].resample('D').sum(min_count=1)
        daily_counts = self.df['Current Temp (F)'].resample('D').count()

        daily_eco = pd.concat([daily_run, daily_temps_all, active_df], axis=1)
        # Filter out partial boundary days (< 12 hours of 5-min samples) if full days exist
        min_samples = 144 if daily_counts.max() >= 144 else 1
        daily_eco = daily_eco[daily_counts >= min_samples].dropna(
            subset=['Cool Stage 1 (sec)', 'Heat Stage 1 (sec)', 'Indoor_Mean', 'Outdoor_Mean']
        )

        if daily_eco.empty:
            self.daily_ecobee = pd.DataFrame()
            return

        daily_eco['Runtime_Cool_Hr'] = daily_eco['Cool Stage 1 (sec)'] / 3600.0
        daily_eco['Runtime_Heat_Hr'] = daily_eco['Heat Stage 1 (sec)'] / 3600.0
        daily_eco['Runtime_Total_Hr'] = daily_eco['Runtime_Cool_Hr'] + daily_eco['Runtime_Heat_Hr']
        daily_eco['Duty_Cool_Pct'] = (daily_eco['Runtime_Cool_Hr'] / 24.0) * 100.0
        daily_eco['Duty_Heat_Pct'] = (daily_eco['Runtime_Heat_Hr'] / 24.0) * 100.0
        daily_eco['Duty_Total_Pct'] = (daily_eco['Runtime_Total_Hr'] / 24.0) * 100.0

        daily_eco['Active_Indoor'] = daily_eco['Active_Indoor'].fillna(daily_eco['Indoor_Mean'])
        daily_eco['Active_Outdoor'] = daily_eco['Active_Outdoor'].fillna(daily_eco['Outdoor_Mean'])
        daily_eco['Active_Delta_T'] = daily_eco['Active_Outdoor'] - daily_eco['Active_Indoor']
        daily_eco['Delta_T_Mean'] = daily_eco['Outdoor_Mean'] - daily_eco['Indoor_Mean']
        daily_eco['Mode'] = self._classify_mode(daily_eco)

        self.daily_ecobee = daily_eco

        # Pre-calculate Balance Points & 100% Duty Design Temps for reporting
        self.balance_point_heating = None
        self.balance_point_cooling = None
        self.design_temp_heating = None
        self.design_temp_cooling = None

        heat_days = daily_eco[(daily_eco['Mode'] == 'Heating') & (daily_eco['Runtime_Heat_Hr'] > 0.25)]
        if len(heat_days) >= 4:
            res_bal_h = stats.linregress(heat_days['Outdoor_Mean'], heat_days['Runtime_Heat_Hr'])
            if res_bal_h.slope < -0.01:
                self.balance_point_heating = -res_bal_h.intercept / res_bal_h.slope
            res_cap_h = stats.linregress(heat_days['Active_Delta_T'], heat_days['Duty_Heat_Pct'])
            if res_cap_h.slope < -0.1:
                dt_lim_h = (100.0 - res_cap_h.intercept) / res_cap_h.slope
                self.design_temp_heating = heat_days['Active_Indoor'].mean() + dt_lim_h

        cool_days = daily_eco[(daily_eco['Mode'] == 'Cooling') & (daily_eco['Runtime_Cool_Hr'] > 0.25)]
        if len(cool_days) >= 4:
            res_bal_c = stats.linregress(cool_days['Outdoor_Mean'], cool_days['Runtime_Cool_Hr'])
            if res_bal_c.slope > 0.01:
                self.balance_point_cooling = -res_bal_c.intercept / res_bal_c.slope
            res_cap_c = stats.linregress(cool_days['Active_Delta_T'], cool_days['Duty_Cool_Pct'])
            if res_cap_c.slope > 0.1:
                dt_lim_c = (100.0 - res_cap_c.intercept) / res_cap_c.slope
                self.design_temp_cooling = cool_days['Active_Indoor'].mean() + dt_lim_c

    def _analyze_inverter_performance(self, timezone):
        """
        Correlates Ecobee Runtime with Green Button Energy.
        Ecobee CSV timestamps are naive local time; Green Button XML timestamps are UTC.
        """
        print(f"Aligning Ecobee Data to UTC using timezone: {timezone}")
        
        # Create a UTC-aligned copy of the dataframe for Energy aggregation
        try:
            utc_df = self.df.copy()
            # Localize to user timezone, then convert to UTC. Use 'NaT' for ambiguous times (DST fall-back)
            utc_df.index = utc_df.index.tz_localize(timezone, ambiguous='NaT', nonexistent='shift_forward').tz_convert('UTC')
            utc_df = utc_df[utc_df.index.notnull()] # Drop the ambiguous hour
            # Remove TZ info to match PowerDF (which is Naive UTC)
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
        
        # Fallback temperatures for zero-runtime intervals so they are not dropped before baseload regression
        all_temps = utc_df[['Current Temp (F)', 'Outdoor Temp (F)']].resample(freq_str).mean()
        all_temps.columns = ['Active_Indoor', 'Active_Outdoor']
        active_temps = active_temps.combine_first(all_temps)
        
        # --- 2. Aggregate Runtime ---
        cols = [c for c in ['Cool Stage 1 (sec)', 'Heat Stage 1 (sec)', 'Fan (sec)'] if c in utc_df.columns]
        ecobee_agg = utc_df[cols].resample(freq_str).sum(min_count=1)
        
        ecobee_agg = pd.concat([ecobee_agg, active_temps], axis=1)
        
        ecobee_agg['Runtime_Cool_Hr'] = ecobee_agg['Cool Stage 1 (sec)'] / 3600.0
        ecobee_agg['Runtime_Heat_Hr'] = ecobee_agg['Heat Stage 1 (sec)'] / 3600.0
        ecobee_agg['Runtime_Total_Hr'] = ecobee_agg['Runtime_Cool_Hr'] + ecobee_agg['Runtime_Heat_Hr']
        
        # --- 3. Merge with Power Data & Build Daily Series ---
        combined = pd.concat([ecobee_agg, self.power_df], axis=1).dropna()
        
        daily_ecobee = utc_df[cols].resample('D').sum(min_count=1)
        daily_active_mean = active_df.resample('D').mean()
        daily_all_mean = utc_df[['Current Temp (F)', 'Outdoor Temp (F)']].resample('D').mean()
        daily_temps = daily_active_mean.combine_first(daily_all_mean)
        daily_temps.columns = ['Active_Indoor', 'Active_Outdoor']
        daily_ecobee = pd.concat([daily_ecobee, daily_temps], axis=1)
        daily_ecobee['Runtime_Cool_Hr'] = daily_ecobee['Cool Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Heat_Hr'] = daily_ecobee['Heat Stage 1 (sec)'] / 3600.0
        daily_ecobee['Runtime_Total_Hr'] = daily_ecobee['Runtime_Cool_Hr'] + daily_ecobee['Runtime_Heat_Hr']
        
        if hasattr(self, 'daily_power_df') and not self.daily_power_df.empty:
            daily_power = self.daily_power_df['kWh']
        else:
            daily_power = self.power_df['kWh'].resample('D').sum(min_count=1) if freq_str == 'h' else self.power_df['kWh']
        daily_combined = pd.concat([daily_ecobee, daily_power], axis=1).dropna()

        if combined.empty and daily_combined.empty:
            print("Warning: No date overlap between Thermostat and Power data.")
            return

        # --- BASELOAD LOGIC: Linear Regression on Daily Series (24h integration avoids diurnal appliance bias) ---
        if not daily_combined.empty and len(daily_combined) >= 5:
            self.baseload_reg_df = daily_combined.copy()
            self.baseload_reg_freq = 'D'
            reg_div = 24.0
        else:
            self.baseload_reg_df = combined.copy()
            self.baseload_reg_freq = freq_str
            reg_div = 1.0 if freq_str == 'h' else 24.0

        x = self.baseload_reg_df['Runtime_Total_Hr']
        y = self.baseload_reg_df['kWh']
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        self.baseload_reg_stats = (slope, intercept, r_value, p_value, std_err)
        
        reg_baseload_kw = intercept / reg_div
        print(f"Baseload Regression: Intercept={intercept:.2f} kWh/{self.baseload_reg_freq} -> {reg_baseload_kw:.2f} kW (R2={r_value**2:.3f})")
        
        fallback_div = 1.0 if freq_str == 'h' else 24.0
        if reg_baseload_kw > 0.1 and reg_baseload_kw < 10.0:
            self.baseload_kw = reg_baseload_kw
            self.baseload_method = "regression"
            print(f"Using Regression Baseload: {self.baseload_kw:.2f} kW")
        else:
            self.baseload_kw = (self.baseload_reg_df['kWh'] / reg_div).quantile(0.05)
            self.baseload_method = "quantile (5th pct)"
            print(f"Regression yielded invalid baseload. Falling back to quantile: {self.baseload_kw:.2f} kW")

        if not combined.empty:
            combined['kWh_HVAC'] = combined['kWh'] - (self.baseload_kw * fallback_div)
            combined['Active_Delta_T'] = combined['Active_Outdoor'] - combined['Active_Indoor']
            combined['active_dt_abs'] = combined['Active_Delta_T'].abs()
            combined['Inverter_Power_kW'] = combined['kWh_HVAC'] / combined['Runtime_Total_Hr'].replace(0, np.nan)
            combined['Mode'] = self._classify_mode(combined)
            self.interval_combined = combined[combined['Runtime_Total_Hr'] > (0.01 if freq_str == 'h' else 0.1)].copy()
        else:
            self.interval_combined = pd.DataFrame()

        if not daily_combined.empty:
            daily_combined['kWh_HVAC'] = daily_combined['kWh'] - (self.baseload_kw * 24.0)
            daily_combined['Active_Delta_T'] = daily_combined['Active_Outdoor'] - daily_combined['Active_Indoor']
            daily_combined['active_dt_abs'] = daily_combined['Active_Delta_T'].abs()
            daily_combined['Inverter_Power_kW'] = daily_combined['kWh_HVAC'] / daily_combined['Runtime_Total_Hr'].replace(0, np.nan)
            daily_combined['Mode'] = self._classify_mode(daily_combined)
            self.daily_combined = daily_combined[daily_combined['Runtime_Total_Hr'] > 0.1].copy()
        else:
            self.daily_combined = pd.DataFrame()

    def _finalize_plot(self, fig, filename, show):
        """Saves figure to Results/ and displays or closes cleanly without blocking multi-window UI loops."""
        plt.tight_layout()
        self._ensure_results_dir()
        if filename:
            out_path = os.path.join("Results", filename)
            plt.savefig(out_path)
            print(f"Saved plot: {out_path}")
        if show:
            if len(plt.get_fignums()) > 1:
                plt.show(block=False)
            else:
                plt.show()
        else:
            plt.close(fig)

    def plot_inverter_curve(self, filename='inverter_profile.png', show=True):
        """
        Plots the Inverter Modulation Profile (Instantaneous Power kW vs. Delta T).
        NOTE: Only generated when hourly-resolution ('h') Green Button power data is available.
        When only monthly exports (with daily 'D' resolution) are available, intra-day
        modulation cannot be resolved from 24-hour energy sums due to diurnal temperature
        swings and day-to-day non-HVAC appliance variance.
        """
        if getattr(self, 'power_resolution', 'D') != 'h':
            print(
                "Skipping Inverter Modulation Curve: requires hourly-resolution power data "
                "(only daily-resolution power data is loaded)."
            )
            return

        comb = getattr(self, 'interval_combined', None)
        if comb is None or comb.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 7))
        
        heating = comb[comb['Mode'] == 'Heating']
        cooling = comb[comb['Mode'] == 'Cooling']

        # Filter for stability (Runtime > min_runtime) to avoid asymptotic noise
        min_runtime = 0.5 if getattr(self, 'power_resolution', 'D') == 'h' else 1.0
        h_stable = heating[heating['Runtime_Total_Hr'] > min_runtime]
        c_stable = cooling[cooling['Runtime_Total_Hr'] > min_runtime]

        def plot_series(data, color, label_prefix):
            if data.empty: return
            x = data['Active_Delta_T'].to_numpy(dtype=float)
            y = data['Inverter_Power_kW'].to_numpy(dtype=float)
            
            ax.scatter(x, y, c=color, alpha=0.6, label=f'{label_prefix} Data (n={len(data)})')
            
            # Quadratic fit with subtle 95% confidence band
            if len(data) > 5:
                try:
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    xr = np.linspace(x.min(), x.max(), 80)
                    yr = p(xr)
                    ax.plot(xr, yr, color=color, linestyle='--', alpha=0.85, lw=2)

                    # 95% CI band on quadratic regression curve
                    n = len(x)
                    y_hat = p(x)
                    ss_res = np.sum((y - y_hat) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    if n > 3:
                        X_design = np.vstack([x ** 2, x, np.ones_like(x)]).T
                        Xr_design = np.vstack([xr ** 2, xr, np.ones_like(xr)]).T
                        s2 = ss_res / (n - 3)
                        cov = s2 * np.linalg.pinv(X_design.T @ X_design)
                        se_fit = np.sqrt(np.maximum(0, np.sum((Xr_design @ cov) * Xr_design, axis=1)))
                        t_val = stats.t.ppf(0.975, df=n - 3)
                        ax.fill_between(xr, yr - t_val * se_fit, yr + t_val * se_fit,
                                        color=color, alpha=0.12, edgecolor='none')
                    
                    # Annotations
                    min_p = y.min()
                    max_p = y.max()
                    eq_str = f"{z[0]:.4f}x² + {z[1]:.3f}x + {z[2]:.2f}"
                    
                    # Add stats box
                    text_str = (f"{label_prefix} Performance (R²={r2:.2f}):\n"
                                f"Range: {min_p:.2f} - {max_p:.2f} kW\n"
                                f"Fit: {eq_str}")
                    
                    # Position box based on heating/cooling (Heating=Left/Top, Cooling=Right/Bottom roughly)
                    yloc = 0.85 if label_prefix == 'Heating' else 0.18
                    xloc = 0.05 if label_prefix == 'Heating' else 0.62
                    
                    ax.text(xloc, yloc, text_str, transform=ax.transAxes, 
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color))
                except Exception:
                    pass

        plot_series(h_stable, 'red', 'Heating')
        plot_series(c_stable, 'blue', 'Cooling')

        start_str = self.df.index.min().strftime('%Y-%m-%d')
        end_str = self.df.index.max().strftime('%Y-%m-%d')
        ax.set_title(
            f"Inverter Modulation Profile (System Capacity)\n"
            f"Range: {start_str} to {end_str} | Baseload Subtracted: {self.baseload_kw:.2f} kW",
            fontsize=11
        )
        ax.set_xlabel("Active Delta T (°F) [Outdoor - Indoor]", fontsize=10)
        ax.set_ylabel("Average Power Output (kW)", fontsize=10)
        ax.axvline(0, color='k', lw=0.5)
        ax.grid(True, alpha=0.3)
        
        # Interpretation Guide
        guide_text = ("Interpretation:\n"
                      "• Flat Line: Single/Two-Stage System (Fixed Capacity)\n"
                      "• Sloped/Curved: Inverter System (Modulating Capacity)\n"
                      "• Shaded Band: 95% Confidence Interval of Fit")
        ax.text(0.98, 0.98, guide_text, transform=ax.transAxes, 
                horizontalalignment='right', verticalalignment='top', fontsize=8, color='#555555',
                bbox=dict(boxstyle='square', facecolor='#f0f0f0', alpha=1.0))

        ax.legend(loc='lower left')
        self._finalize_plot(fig, filename, show)

    def plot_energy_profile(self, filename='total_energy_profile.png', show=True):
        """Standard Output: Total Energy Profile (Cost to Condition), Baseload Fit & UA/COP Cross-Validation."""
        reg_df = getattr(self, 'baseload_reg_df', None)
        if reg_df is None or reg_df.empty:
            reg_df = getattr(self, 'interval_combined', getattr(self, 'daily_combined', None))
        if reg_df is None or reg_df.empty: return
        freq_str = getattr(self, 'baseload_reg_freq', getattr(self, 'power_resolution', 'D'))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
        
        # --- PLOT 1: Baseload Regression (Runtime Method) ---
        # Uses the exact same dataset (including zero-runtime points) used in _analyze_inverter_performance
        x = reg_df['Runtime_Total_Hr']
        y = reg_df['kWh']
        ax1.scatter(x, y, alpha=0.5, c='purple', label=f'Data ({freq_str}, n={len(reg_df)})')
        
        res_base = self._plot_linear_fit_with_ci(
            ax1, x, y, color='black', linestyle='--', lw=2,
            x_range=(0, x.max()),
            label=f'Fit (R²={res_base_r2:.2f})' if (res_base_r2 := stats.linregress(x, y).rvalue ** 2) >= 0 else None
        )
        slope, intercept, r_val = res_base.slope, res_base.intercept, res_base.rvalue
        
        # Annotate
        active_base_interval = self.baseload_kw if freq_str == 'h' else self.baseload_kw * 24.0
        ax1.axhline(active_base_interval, color='green', linestyle=':', alpha=0.7,
                    label=f'Active Baseload ({self.baseload_kw:.2f} kW)')
        b_kw = intercept if freq_str == 'h' else intercept / 24.0
        base_note = (
            f"Slope (Avg HVAC Power): {slope:.2f} kW\n"
            f"Regressed Intercept: {intercept:.1f} kWh/{freq_str} ({b_kw:.2f} kW)\n"
            f"Applied Baseload ({self.baseload_method}): {self.baseload_kw:.2f} kW"
        )
        ax1.text(0.05, 0.95, base_note,
                 transform=ax1.transAxes, verticalalignment='top', fontsize=9,
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
            mixed = daily_comb[daily_comb['Mode'] == 'Mixed']
            
            # Plot Baseload Reference 
            d_base = self.baseload_kw * 24.0
            ax2.axhline(d_base, color='green', linestyle=':', alpha=0.6, label=f'Baseload Ref ({d_base:.1f} kWh/d)')

            if not mixed.empty:
                ax2.scatter(mixed['Active_Delta_T'], mixed['kWh'], c='grey', alpha=0.5, label='Mixed Days')
        
            if not heat.empty:
                ax2.scatter(heat['Active_Delta_T'], heat['kWh'], c='red', alpha=0.6, label=f'Heating Days (n={len(heat)})')
                if len(heat) > 5:
                    res = self._plot_linear_fit_with_ci(
                        ax2, heat['Active_Delta_T'], heat['kWh'], color='red', linestyle='--', lw=2, label='Heating Fit'
                    )
                    slope_abs = abs(res.slope)
                    ua_from_energy = (slope_abs * 3412.14 * 3.0) / 24.0 if slope_abs > 0.001 else 0.0
                    info_lines = [
                        f"Heating Cost Slope: {res.slope:.2f} kWh/°F/d (R²={res.rvalue**2:.2f})",
                        f"Assumed Mass C: {self.C_est:.0f} BTU/°F ({int(self.sq_ft)} sq ft)"
                    ]
                    if self.UA_heating and self.UA_heating > 0 and slope_abs > 0.001:
                        cop_eff = (self.UA_heating * 24.0) / (slope_abs * 3412.14)
                        info_lines.append(f"Coasting UA: {self.UA_heating:.0f} BTU/hr·°F → Eff. COP: {cop_eff:.2f}")
                        info_lines.append(f"Cross-Check UA (@COP=3.0): {ua_from_energy:.0f} BTU/hr·°F")
                    else:
                        info_lines.append(f"Implied UA (@COP=3.0): {ua_from_energy:.0f} BTU/hr·°F")

                    ax2.text(0.04, 0.95, "\n".join(info_lines),
                             transform=ax2.transAxes, verticalalignment='top', color='#900000', fontsize=8.5,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red'))

            if not cool.empty:
                ax2.scatter(cool['Active_Delta_T'], cool['kWh'], c='blue', alpha=0.6, label=f'Cooling Days (n={len(cool)})')
                if len(cool) > 5:
                    res = self._plot_linear_fit_with_ci(
                        ax2, cool['Active_Delta_T'], cool['kWh'], color='blue', linestyle='--', lw=2, label='Cooling Fit'
                    )
                    slope_abs = abs(res.slope)
                    ua_from_energy = (slope_abs * 3412.14 * 3.0) / 24.0 if slope_abs > 0.001 else 0.0
                    info_lines = [
                        f"Cooling Cost Slope: {res.slope:.2f} kWh/°F/d (R²={res.rvalue**2:.2f})",
                    ]
                    if self.UA_cooling and self.UA_cooling > 0 and slope_abs > 0.001:
                        cop_eff = (self.UA_cooling * 24.0) / (slope_abs * 3412.14)
                        info_lines.append(f"Coasting UA: {self.UA_cooling:.0f} BTU/hr·°F → Eff. COP: {cop_eff:.2f}")
                        info_lines.append(f"Cross-Check UA (@COP=3.0): {ua_from_energy:.0f} BTU/hr·°F")
                    else:
                        info_lines.append(f"Implied UA (@COP=3.0): {ua_from_energy:.0f} BTU/hr·°F")

                    ax2.text(0.04, 0.22, "\n".join(info_lines),
                             transform=ax2.transAxes, verticalalignment='top', color='#003399', fontsize=8.5,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='blue'))
                
        ax2.set_title("Method 2: Temperature Profile & UA/COP Cross-Check\n(Daily Energy vs Delta T)")
        ax2.set_xlabel("Outdoor - Indoor Delta T (°F)")
        ax2.set_ylabel("Total Daily Energy (kWh)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize='small')
        
        self._finalize_plot(fig, filename, show)

    def plot_balance_point(self, filename='balance_point.png', show=True):
        """
        Info 1: Self-annotated plot for Building Balance Point Temperature (T_balance)
        and Degree-Day runtime sensitivity using Ecobee daily data.
        """
        if self.daily_ecobee is None or self.daily_ecobee.empty:
            self._build_daily_ecobee_summary()
        if self.daily_ecobee is None or self.daily_ecobee.empty:
            return

        df_d = self.daily_ecobee
        fig, ax = plt.subplots(figsize=(10, 6.5))

        idle = df_d[df_d['Runtime_Total_Hr'] <= 0.25]
        heat = df_d[(df_d['Mode'] == 'Heating') & (df_d['Runtime_Heat_Hr'] > 0.25)]
        cool = df_d[(df_d['Mode'] == 'Cooling') & (df_d['Runtime_Cool_Hr'] > 0.25)]
        mixed = df_d[(df_d['Mode'] == 'Mixed') & (df_d['Runtime_Total_Hr'] > 0.25)]

        if not idle.empty:
            ax.scatter(idle['Outdoor_Mean'], idle['Runtime_Total_Hr'],
                       c='grey', alpha=0.5, label=f'Idle / Coasting Days (n={len(idle)})')
        if not mixed.empty:
            ax.scatter(mixed['Outdoor_Mean'], mixed['Runtime_Total_Hr'],
                       c='purple', alpha=0.5, label=f'Mixed Mode Days (n={len(mixed)})')

        t_bal_h_str = "N/A"
        t_bal_c_str = "N/A"

        if not heat.empty:
            x_h = heat['Outdoor_Mean']
            y_h = heat['Runtime_Heat_Hr']
            ax.scatter(x_h, y_h, c='red', alpha=0.65, label=f'Heating Days (n={len(heat)})')
            if len(heat) >= 4:
                res_h = stats.linregress(x_h, y_h)
                if res_h.slope < -0.01:
                    t_bal_h = -res_h.intercept / res_h.slope
                    self.balance_point_heating = t_bal_h
                    t_bal_h_str = f"{t_bal_h:.1f}°F"
                    x_max_plot = min(85.0, max(x_h.max(), t_bal_h))
                    self._plot_linear_fit_with_ci(
                        ax, x_h, y_h, color='red', linestyle='--', lw=2,
                        x_range=(x_h.min(), x_max_plot),
                        label=f'Heating Fit (Bal={t_bal_h_str})'
                    )
                    ax.axvline(t_bal_h, color='red', linestyle=':', alpha=0.6)
                    t_in_mean = heat['Indoor_Mean'].mean()
                    gain_offset = t_in_mean - t_bal_h
                    text_h = (
                        f"Heating Balance Signature (R²={res_h.rvalue**2:.2f}):\n"
                        f"• Balance Point Temp: {t_bal_h:.1f}°F\n"
                        f"• Free Internal/Solar Offset: {gain_offset:+.1f}°F (vs {t_in_mean:.1f}°F In)\n"
                        f"• Degree-Day Slope: {abs(res_h.slope):.2f} hr/°F ({abs(res_h.slope)*60:.0f} min/HDD)"
                    )
                    ax.text(0.04, 0.86, text_h, transform=ax.transAxes,
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red'))

        if not cool.empty:
            x_c = cool['Outdoor_Mean']
            y_c = cool['Runtime_Cool_Hr']
            ax.scatter(x_c, y_c, c='blue', alpha=0.65, label=f'Cooling Days (n={len(cool)})')
            if len(cool) >= 4:
                res_c = stats.linregress(x_c, y_c)
                if res_c.slope > 0.01:
                    t_bal_c = -res_c.intercept / res_c.slope
                    self.balance_point_cooling = t_bal_c
                    t_bal_c_str = f"{t_bal_c:.1f}°F"
                    x_min_plot = max(40.0, min(x_c.min(), t_bal_c))
                    self._plot_linear_fit_with_ci(
                        ax, x_c, y_c, color='blue', linestyle='--', lw=2,
                        x_range=(x_min_plot, x_c.max()),
                        label=f'Cooling Fit (Bal={t_bal_c_str})'
                    )
                    ax.axvline(t_bal_c, color='blue', linestyle=':', alpha=0.6)
                    text_c = (
                        f"Cooling Balance Signature (R²={res_c.rvalue**2:.2f}):\n"
                        f"• Balance Point Temp: {t_bal_c:.1f}°F\n"
                        f"• Degree-Day Slope: {res_c.slope:.2f} hr/°F ({res_c.slope*60:.0f} min/CDD)"
                    )
                    ax.text(0.55, 0.55, text_c, transform=ax.transAxes,
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='blue'))

        ax.axhline(0, color='k', lw=0.8, alpha=0.5)
        ax.set_ylim(bottom=-0.5)

        start_str = self.df.index.min().strftime('%Y-%m-%d')
        end_str = self.df.index.max().strftime('%Y-%m-%d')
        subtitle = (
            f"Range: {start_str} to {end_str} | House: {int(self.sq_ft)} sq ft\n"
            f"Balance Point Temps: Heat={t_bal_h_str}, Cool={t_bal_c_str}"
        )
        ax.set_title(f"Building Balance Point & Degree-Day Signature\n{subtitle}", fontsize=10.5)
        ax.set_xlabel("Daily Mean Outdoor Temperature (°F)")
        ax.set_ylabel("Daily HVAC Runtime (Hours/day)")
        ax.grid(True, alpha=0.3)

        guide_text = (
            "Interpretation:\n"
            "• Balance Point (X-Intercept): Outdoor temp where HVAC is first needed\n"
            "• Lower Heat Bal Temp = Better insulation or higher internal heat gains\n"
            "• Shaded Band: 95% Confidence Interval of Fit"
        )
        ax.text(0.98, 0.98, guide_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top', fontsize=8, color='#555555',
                bbox=dict(boxstyle='square', facecolor='#f0f0f0', alpha=1.0))

        ax.legend(loc='upper left', fontsize='small')
        self._finalize_plot(fig, filename, show)

    def plot_sizing_limit(self, filename='sizing_limit.png', show=True):
        """
        Info 2: Self-annotated plot for Heat Pump Sizing Limit & Max Capacity Design Temperature
        by extrapolating Daily Duty Cycle (%) vs. Active Delta T to 100% (24h/day).
        """
        if self.daily_ecobee is None or self.daily_ecobee.empty:
            self._build_daily_ecobee_summary()
        if self.daily_ecobee is None or self.daily_ecobee.empty:
            return

        df_d = self.daily_ecobee
        fig, ax = plt.subplots(figsize=(10, 6.5))

        heat = df_d[(df_d['Mode'] == 'Heating') & (df_d['Runtime_Heat_Hr'] > 0.25)]
        cool = df_d[(df_d['Mode'] == 'Cooling') & (df_d['Runtime_Cool_Hr'] > 0.25)]

        ax.axhline(100.0, color='darkred', linestyle='-.', lw=1.5, alpha=0.8,
                   label='100% Duty Cycle Ceiling (24h/day Max Capacity)')

        lim_h_str = "N/A"
        lim_c_str = "N/A"

        if not heat.empty:
            x_h = heat['Active_Delta_T']
            y_h = heat['Duty_Heat_Pct']
            ax.scatter(x_h, y_h, c='red', alpha=0.65, label=f'Heating Duty Cycle (n={len(heat)})')
            if len(heat) >= 4:
                res_h = stats.linregress(x_h, y_h)
                if res_h.slope < -0.1:
                    dt_lim_h = (100.0 - res_h.intercept) / res_h.slope
                    t_in_h = heat['Active_Indoor'].mean()
                    t_out_lim_h = t_in_h + dt_lim_h
                    self.design_temp_heating = t_out_lim_h
                    lim_h_str = f"{t_out_lim_h:.1f}°F (ΔT={dt_lim_h:.1f}°F)"

                    x_left = max(dt_lim_h, min(x_h.min() - 10.0, -70.0))
                    self._plot_linear_fit_with_ci(
                        ax, x_h, y_h, color='red', linestyle='--', lw=2,
                        x_range=(x_left, x_h.max()),
                        label='Heating Capacity Projection'
                    )
                    if dt_lim_h >= x_left:
                        ax.scatter([dt_lim_h], [100.0], c='darkred', marker='D', s=55, zorder=5)

                    text_h = (
                        f"Heating Sizing & Capacity (R²={res_h.rvalue**2:.2f}):\n"
                        f"• 100% Duty Limit: ΔT = {dt_lim_h:.1f}°F (~{t_out_lim_h:.1f}°F Outdoor)\n"
                        f"• Duty Sensitivity: {abs(res_h.slope):.2f}% / °F ({abs(res_h.slope)*0.24:.2f} hr/°F)\n"
                        f"• Peak Observed: {y_h.max():.1f}% ({heat['Runtime_Heat_Hr'].max():.1f} hr/day)"
                    )
                    ax.text(0.04, 0.84, text_h, transform=ax.transAxes,
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red'))

        if not cool.empty:
            x_c = cool['Active_Delta_T']
            y_c = cool['Duty_Cool_Pct']
            ax.scatter(x_c, y_c, c='blue', alpha=0.65, label=f'Cooling Duty Cycle (n={len(cool)})')
            if len(cool) >= 4:
                res_c = stats.linregress(x_c, y_c)
                if res_c.slope > 0.1:
                    dt_lim_c = (100.0 - res_c.intercept) / res_c.slope
                    t_in_c = cool['Active_Indoor'].mean()
                    t_out_lim_c = t_in_c + dt_lim_c
                    self.design_temp_cooling = t_out_lim_c
                    lim_c_str = f"{t_out_lim_c:.1f}°F (ΔT=+{dt_lim_c:.1f}°F)"

                    x_right = min(dt_lim_c, max(x_c.max() + 10.0, 50.0))
                    self._plot_linear_fit_with_ci(
                        ax, x_c, y_c, color='blue', linestyle='--', lw=2,
                        x_range=(x_c.min(), x_right),
                        label='Cooling Capacity Projection'
                    )
                    if dt_lim_c <= x_right:
                        ax.scatter([dt_lim_c], [100.0], c='darkblue', marker='D', s=55, zorder=5)

                    text_c = (
                        f"Cooling Sizing & Capacity (R²={res_c.rvalue**2:.2f}):\n"
                        f"• 100% Duty Limit: ΔT = +{dt_lim_c:.1f}°F (~{t_out_lim_c:.1f}°F Outdoor)\n"
                        f"• Duty Sensitivity: {res_c.slope:.2f}% / °F ({res_c.slope*0.24:.2f} hr/°F)\n"
                        f"• Peak Observed: {y_c.max():.1f}% ({cool['Runtime_Cool_Hr'].max():.1f} hr/day)"
                    )
                    ax.text(0.54, 0.52, text_c, transform=ax.transAxes,
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='blue'))

        ax.axvline(0, color='k', lw=0.6, alpha=0.5)
        ax.set_ylim(-2, 112)

        start_str = self.df.index.min().strftime('%Y-%m-%d')
        end_str = self.df.index.max().strftime('%Y-%m-%d')
        subtitle = (
            f"Range: {start_str} to {end_str} | House: {int(self.sq_ft)} sq ft\n"
            f"Max Capacity Limit (100% Duty): Heat={lim_h_str}, Cool={lim_c_str}"
        )
        ax.set_title(f"Heat Pump Sizing Limit & Max Capacity Design Temperature\n{subtitle}", fontsize=10.5)
        ax.set_xlabel("Active Delta T (°F) [Outdoor - Indoor]")
        ax.set_ylabel("Daily HVAC Duty Cycle (% of 24 Hours)")
        ax.grid(True, alpha=0.3)

        guide_text = (
            "Interpretation:\n"
            "• 100% Duty Ceiling: System runs 24h continuously to hold setpoint\n"
            "• Beyond Limit: Heat pump loses setpoint or requires Aux Heat\n"
            "• Shaded Band: 95% Confidence Interval of Extrapolation"
        )
        ax.text(0.98, 0.90, guide_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top', fontsize=8, color='#555555',
                bbox=dict(boxstyle='square', facecolor='#f0f0f0', alpha=1.0))

        ax.legend(loc='lower left', fontsize='small')
        self._finalize_plot(fig, filename, show)

    def generate_report(self, filename='thermal_results.txt'):
        self._ensure_results_dir()
        filepath = os.path.join("Results", filename)
        
        lines = []
        lines.append(f"Thermal Analysis & Energy Report")
        lines.append(f"================================")
        lines.append(f"Analysis Range: {self.df.index.min()} to {self.df.index.max()}")
        lines.append(f"House Size: {self.sq_ft} sq ft (Est. Thermal Mass C = {self.C_est:.0f} BTU/degF)")
        lines.append(f"Filters: NightOnly={self.active_filters.get('night_only')}, Buffer={self.active_filters.get('buffer_minutes')}m")
        
        def print_fit(fit, fit_stats, ua_val, name, n_pts):
            if fit:
                slope, intercept = fit
                r2_str = f", R2={fit_stats.rvalue**2:.3f}" if fit_stats is not None else ""
                if slope > 0.001:
                    tau = 1.0 / slope
                    ua_str = f", UA={ua_val:.1f} BTU/hr*degF" if ua_val else ""
                    ci_str = ""
                    if fit_stats is not None and n_pts > 2 and fit_stats.stderr > 0:
                        t_val = stats.t.ppf(0.975, df=n_pts - 2)
                        m_low = slope - t_val * fit_stats.stderr
                        m_high = slope + t_val * fit_stats.stderr
                        if m_low > 0.0005:
                            ci_str = f" (95% CI: {1.0/m_high:.1f}h - {1.0/m_low:.1f}h)"
                    lines.append(f"{name}: Slope={slope:.4f}/hr, Tau={tau:.1f}h{ci_str}{ua_str}{r2_str} (n={n_pts})")
                else:
                    lines.append(f"{name}: Non-physical Slope ({slope:.4f}/hr), Tau=N/A{r2_str} (n={n_pts})")
            else:
                lines.append(f"{name}: Insufficient Data (n={n_pts})")
        
        lines.append("\n-- Envelope Properties (Coasting) --")
        if not getattr(self, 'has_subhourly_thermal', True):
            lines.append("Skipped: Sub-hourly thermal data unavailable (requires <= 2h sampling interval for dT/dt).")
        else:
            n_h = len(self.heating_data) if self.heating_data is not None else 0
            n_c = len(self.cooling_data) if self.cooling_data is not None else 0
            print_fit(self.fit_heating, self.fit_heating_stats, self.UA_heating, "Heating (Winter)", n_h)
            print_fit(self.fit_cooling, self.fit_cooling_stats, self.UA_cooling, "Cooling (Summer)", n_c)

        lines.append("\n-- Balance Point & Sizing Limits (Daily Runtime) --")
        if self.balance_point_heating is not None:
            lines.append(f"Heating Balance Point Temp: {self.balance_point_heating:.1f} degF")
        if self.balance_point_cooling is not None:
            lines.append(f"Cooling Balance Point Temp: {self.balance_point_cooling:.1f} degF")
        if self.design_temp_heating is not None:
            lines.append(f"Heating 100% Duty Design Limit: {self.design_temp_heating:.1f} degF Outdoor")
        if self.design_temp_cooling is not None:
            lines.append(f"Cooling 100% Duty Design Limit: {self.design_temp_cooling:.1f} degF Outdoor")

        if not self.power_df.empty or not getattr(self, 'daily_power_df', pd.DataFrame()).empty:
            lines.append("\n-- Energy Signature & UA/COP Cross-Validation --")
            lines.append(f"Power Data Resolution: {'Hourly + Daily' if getattr(self, 'power_resolution', 'D') == 'h' else 'Daily Only (Monthly Exports)'}")
            lines.append(f"Baseload Power ({self.baseload_method}): {self.baseload_kw:.2f} kW")
            if getattr(self, 'power_resolution', 'D') != 'h':
                lines.append("Inverter Modulation Profile: Skipped (requires hourly-resolution Green Button exports).")
            daily_comb = getattr(self, 'daily_combined', None)
            if daily_comb is not None and not daily_comb.empty:
                heat = daily_comb[daily_comb['Mode'] == 'Heating']
                if not heat.empty and len(heat) > 5:
                    s, i, r, _, _ = stats.linregress(heat['Active_Delta_T'], heat['kWh'])
                    lines.append(f"Heating Total Cost Slope: {s:.3f} kWh/degF/day (R2={r**2:.2f}, n={len(heat)})")
                    if abs(s) > 0.001:
                        ua_imp = (abs(s) * 3412.14 * 3.0) / 24.0
                        lines.append(f"Heating Implied UA (@COP=3.0): {ua_imp:.1f} BTU/hr*degF")
                        if self.UA_heating and self.UA_heating > 0:
                            cop_eff = (self.UA_heating * 24.0) / (abs(s) * 3412.14)
                            lines.append(f"Heating Effective Seasonal COP (from Coasting UA): {cop_eff:.2f}")
                cool = daily_comb[daily_comb['Mode'] == 'Cooling']
                if not cool.empty and len(cool) > 5:
                    s, i, r, _, _ = stats.linregress(cool['Active_Delta_T'], cool['kWh'])
                    lines.append(f"Cooling Total Cost Slope: {s:.3f} kWh/degF/day (R2={r**2:.2f}, n={len(cool)})")
                    if abs(s) > 0.001:
                        ua_imp = (abs(s) * 3412.14 * 3.0) / 24.0
                        lines.append(f"Cooling Implied UA (@COP=3.0): {ua_imp:.1f} BTU/hr*degF")
                        if self.UA_cooling and self.UA_cooling > 0:
                            cop_eff = (self.UA_cooling * 24.0) / (abs(s) * 3412.14)
                            lines.append(f"Cooling Effective Seasonal COP (from Coasting UA): {cop_eff:.2f}")
        
        try:
            with open(filepath, 'w') as f: f.write("\n".join(lines))
        except Exception as e:
            print(f"Error writing report {filepath}: {e}")

    def plot_results(self, filename='thermal_fit.png', show=False):
        if not getattr(self, 'has_subhourly_thermal', True):
            print("Skipping Thermal Response (dT/dt) plot: sub-hourly thermal data is unavailable.")
            return
        self._ensure_results_dir()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.coasting_df is not None:
            ax.scatter(self.coasting_df['Delta_T'], self.coasting_df['dT_dt'], 
                       alpha=0.3, color='grey', label=f'Coasting Data (n={len(self.coasting_df)})')
            
        # Plot fits (only positive physical slopes m > 0.001 produce valid positive Tau)
        tau_h_str = "N/A"
        tau_c_str = "N/A"
        ua_h_str = "N/A"
        ua_c_str = "N/A"
        
        if self.fit_heating:
            m, b = self.fit_heating
            if m > 0.001:
                tau_h = 1.0 / m
                tau_h_str = f"{tau_h:.1f}h"
                ua_h_str = f"{self.UA_heating:.0f} BTU/hr·°F" if self.UA_heating else "N/A"
                r2_h = self.fit_heating_stats.rvalue ** 2 if self.fit_heating_stats else 0.0
                self._plot_linear_fit_with_ci(
                    ax, self.heating_data['Delta_T'], self.heating_data['dT_dt'],
                    color='red', linestyle='-', lw=2,
                    label=f'Heating Fit (Tau={tau_h_str}, UA={ua_h_str}, R²={r2_h:.2f})'
                )
            
        if self.fit_cooling:
            m, b = self.fit_cooling
            if m > 0.001:
                tau_c = 1.0 / m
                tau_c_str = f"{tau_c:.1f}h"
                ua_c_str = f"{self.UA_cooling:.0f} BTU/hr·°F" if self.UA_cooling else "N/A"
                r2_c = self.fit_cooling_stats.rvalue ** 2 if self.fit_cooling_stats else 0.0
                self._plot_linear_fit_with_ci(
                    ax, self.cooling_data['Delta_T'], self.cooling_data['dT_dt'],
                    color='blue', linestyle='-', lw=2,
                    label=f'Cooling Fit (Tau={tau_c_str}, UA={ua_c_str}, R²={r2_c:.2f})'
                )
        
        # Subtitle
        start_str = self.df.index.min().strftime('%Y-%m-%d')
        end_str = self.df.index.max().strftime('%Y-%m-%d')
        filter_str = (
            f"Filters: [Night={self.active_filters.get('night_only')} | "
            f"Buffer={self.active_filters.get('buffer_minutes')}m | "
            f"SqFt={int(self.sq_ft)} (C={self.C_est:.0f} BTU/°F)]"
        )
        subtitle = (
            f"Range: {start_str} to {end_str}\n{filter_str}\n"
            f"Tau: Heat={tau_h_str} (UA={ua_h_str}), Cool={tau_c_str} (UA={ua_c_str})"
        )

        ax.set_title(f"Thermal Response: Indoor Rate of Change vs. Delta T\n{subtitle}", fontsize=10)
        ax.set_xlabel("Outdoor - Indoor Temperature (°F)")
        ax.set_ylabel("Indoor Rate of Change (°F/hr)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize='small')
        
        self._finalize_plot(fig, filename, show)

    def _ensure_results_dir(self):
        if not os.path.exists("Results"): os.makedirs("Results")