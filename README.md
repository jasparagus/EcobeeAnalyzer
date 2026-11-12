# Ecobee Thermal Envelope & Inverter Profiler

A local, physics-based analysis tool that turns your Ecobee thermostat data into a building performance lab. This tool estimates your home's **Thermal Time Constant**, **Heat Transfer Coefficient (UA)**, and **Inverter Heat Pump performance profiles** using historical runtime and temperature data.

This tool was built using Google Gemini.

## 1. Design Goals

The primary goal of this project is to overcome the "observability gap" in residential HVAC. Smart thermostats collect vast amounts of data, but native dashboards rarely provide insights into the physical properties of the building envelope or the variable-speed performance of modern inverter heat pumps.

### Key Objectives:
* **Privacy-First & Local:** No cloud API integrations required. The tool operates entirely on local CSV exports (Ecobee) and XML exports (Green Button Data), ensuring total privacy.
* **Physics Over AI:** Instead of using black-box machine learning, this tool uses first-principles thermodynamics (Newton's Law of Cooling, Thermal RC modeling) to derive actionable metrics.
* **Inverter Transparency:** By correlating thermal decay ("coasting") with energy usage, the tool reverse-engineers the power curve of inverter-driven systems, which typically do not report their real-time power consumption to thermostats.
* **Self-Documenting Output:** **All generated figures and reports must be fully self-contained.** Every plot includes the analysis date range, active filter settings (e.g., "Night Only"), and calculated metrics (Tau) directly in the title/subtitle. This ensures that a screenshot of a result is scientifically complete without external context.

---

## 2. Features

* **Interactive Data Viewer:** Visualize indoor/outdoor temperatures, setpoints, and HVAC runtime with a zoomable interface.
* **Thermal Envelope Analysis:**
    * Calculates the house's **Time Constant ($\tau$)** and **Heat Transfer Coefficient ($UA$)** with subtle 95% confidence intervals.
    * Separates analysis into Heating (Winter) and Cooling (Summer) regimes.
    * **"Washout" Filtering:** Automatically excludes data periods immediately following HVAC cycles to prevent thermal inertia from corrupting the analysis.
    * **Solar Exclusion:** Optional "Night Only" mode to isolate conductive heat loss from solar gain.
* **Balance Point & Sizing Limit Profiling:**
    * **Building Balance Point ($T_{\text{balance}}$):** Regresses daily HVAC runtime against outdoor temperature to determine the exact outdoor temperature where heating or cooling is first required, isolating "free" internal and solar heat gains.
    * **Heat Pump Sizing & Capacity Limit:** Projects daily HVAC duty cycle (%) vs. $\Delta T$ to the 100% duty-cycle ceiling (24h continuous runtime) to identify the minimum/maximum outdoor design temperature the system can sustain before losing setpoint or requiring auxiliary heat.
* **Inverter Profiling (Energy Signature & COP Cross-Validation):**
    * Ingests "Green Button" XML power data from utility providers (supporting both hourly and daily resolutions without losing non-overlapping daily files).
    * Calculates **Baseload Power** to isolate HVAC energy.
    * Generates a performance curve: **Daily Energy (kWh) vs. Temperature Delta**, showing the "cost to condition" the home at various outdoor temperatures and cross-validating envelope $UA$ against electrical input to estimate **Effective Seasonal COP**.

---

## 3. Theoretical Framework

This project models the house as a thermodynamic node using a **Thermal RC (Resistor-Capacitor) Model**.



### 3.1 The Energy Balance Equation
The change in indoor temperature is driven by conductive heat loss/gain, HVAC input, internal loads, and solar gain:

$$C \frac{dT_{in}}{dt} = UA(T_{out} - T_{in}) + Q_{HVAC} + Q_{internal} + Q_{solar}$$

Where:
* $C$: Thermal Mass of the house (BTU/°F).
* $UA$: Heat Transfer Coefficient (BTU/hr·°F).
* $Q$: Heat flow sources.

### 3.2 Deriving Envelope Properties ("Coasting" Method)
To solve for $UA$ and $C$ without knowing the exact output of the HVAC system, we analyze **Coasting Periods**—times when the HVAC is OFF ($Q_{HVAC} = 0$). By filtering for nighttime ($Q_{solar} \approx 0$), the equation simplifies to a linear regression:

$$\underbrace{\frac{dT_{in}}{dt}}_{Y} = \underbrace{\frac{UA}{C}}_{\text{Slope } m} \cdot \underbrace{(T_{out} - T_{in})}_{X} + \underbrace{\frac{Q_{internal}}{C}}_{\text{Intercept } b}$$

The inverse of the slope gives us the **Thermal Time Constant ($\tau$)**:
$$\tau = \frac{C}{UA} = \frac{1}{Slope}$$

### 3.3 Energy Signature (Daily kWh vs Delta T) & COP Cross-Validation
For inverter systems, instantaneous power varies. A more robust metric is the **Energy Signature**, which correlates daily energy input with the daily thermal load (Delta T).

$$E_{hvac}(kWh) \approx E_{total} - (P_{base} \times 24h)$$

Plotting $E_{hvac}$ vs $(T_{out} - T_{in})$ reveals the system's efficiency curve and the building's aggregate thermal load. Combining the coasting-derived $UA$ ($\text{BTU/hr}\cdot^\circ\text{F}$) with the daily electrical energy slope ($S_{\text{kWh/}^\circ\text{F}}$) provides an independent estimate of the heat pump's **Effective Seasonal COP**:

$$\text{COP}_{\text{eff}} = \frac{UA \times 24\text{ hr}}{3412.14\text{ BTU/kWh} \times |S_{\text{kWh/}^\circ\text{F}}|}$$

---

## 4. Project Structure & Usage

### Files
* **`ecobee_viewer.py`**: The GUI entry point. Handles file loading, plotting, and user interaction.
* **`thermal_analysis.py`**: The physics engine. Handles resampling, filtering (washout/night logic), and regression analysis.
* **`run.sh`**: Linux launcher script that automatically initializes a local `.venv` via `uv`, installs minimal dependencies if missing, and starts the viewer (supports double-clicking).

### Installation
1.  Clone the repository.
2.  **Quick Start on Linux (`uv`):** Double-click `run.sh` or run `./run.sh` (requires `uv` and system `tk`/`python3-tk`).
3.  **Manual Setup:** Install dependencies and run directly:
    ```bash
    pip install pandas matplotlib scipy numpy
    python ecobee_viewer.py
    ```

### Workflow
1.  **Export Data:**
    * **Ecobee:** Log in to the web portal -> Home IQ -> System Monitor -> Download Data.
    * **Power (Optional):** Download "Green Button" XML (Hourly or Daily resolution) from your utility provider.
2.  **Load:** Use the GUI buttons to load the CSVs and XMLs.
3.  **Configure:**
    * Set **Square Footage** (Default: 1500 sq ft) and **Timezone** (`TZ`, Default: `US/Pacific`).
    * Toggle **Night Only** to remove solar noise (Recommended).
    * Toggle **30m Buffer** to ignore lingering heat after HVAC cycles.
4.  **Analyze:** Click "Calculate Profile".
5.  **Results:** Check the `./Results` folder for generated plots (`.png`) and text reports (`.txt`).

---

## 5. Data Timezone & Alignment Conventions

Understanding the timestamp conventions of the two input formats is essential when inspecting or extending the pipeline:

* **Ecobee CSV Data (Naive Local Time):** Ecobee exports store `Date` and `Time` in the thermostat's **naive local wall-clock time** (with no embedded timezone offset).
* **Green Button XML Data (UTC Unix Timestamps):** Utility Green Button `<IntervalReading><start>` elements are stored as **UTC Unix epoch timestamps** (seconds since `1970-01-01T00:00:00Z`).
* **Alignment Behavior:**
    * Envelope coasting, Balance Point, and Sizing Limit analyses operate directly on the Ecobee local-time series (so nighttime filters like `23:00–06:00` naturally reflect local night).
    * When correlating Ecobee runtime with Green Button energy in `_analyze_inverter_performance`, the Ecobee timestamps are localized using the configured `TZ` (e.g., `US/Pacific`), converted to `UTC` (handling DST transitions), and aligned against the UTC Green Button intervals.
    * In the interactive viewer (`ecobee_viewer.py`), the main temperature/program axes display Ecobee naive local time, while raw Green Button bars are plotted as loaded in naive UTC unless aligned during profile calculation.

---

## 6. Data Resolution & Metric Availability

Different metrics require different temporal resolutions. The analyzer automatically inspects the loaded data resolution and generates only the physically meaningful metrics and plots:

| Metric / Plot | Required Thermal Resolution | Required Power Resolution | Notes |
| :--- | :--- | :--- | :--- |
| **Thermal Time Constant ($\tau$) & Coasting $UA$** (`thermal_plot_*.png`) | **Sub-hourly / Hourly** ($\le 2\text{h}$ step) | *None* | Requires intra-day coasting intervals ($dT_{\text{in}}/dt$ while HVAC is OFF). Skipped if only daily thermal summaries exist. |
| **Building Balance Point ($T_{\text{balance}}$)** (`balance_point_*.png`) | **Daily or finer** | *None* | Regresses total daily runtime (hrs/day) against daily mean outdoor temperature. Valid with daily-resolution data. |
| **Heat Pump Sizing & 100% Duty Limit** (`sizing_limit_*.png`) | **Daily or finer** | *None* | Regresses daily duty cycle (%) against daily $\Delta T$. Valid with daily-resolution data. |
| **Inverter Modulation Curve (kW vs. $\Delta T$)** (`inverter_profile_*.png`) | **Sub-hourly / Hourly** | **Hourly** (`duration=3600`) | Requires hourly Green Button ("Daily" utility exports) to isolate active-hour kW modulation. **Skipped when only Monthly (per-day resolution) power XMLs are loaded**, since 24h totals average out intra-day compressor modulation and part-load cycling. |
| **Daily Energy Signature, Implied $UA$ & Seasonal COP** (`energy_total_*.png`) | **Daily or finer** | **Daily or Hourly** (`duration=86400` or `3600`) | Fully supported with Monthly Green Button exports (per-day kWh resolution). If both Hourly and Monthly XMLs are loaded, hourly data is used for the Inverter Curve while all non-overlapping daily and hourly data are combined for the multi-month Energy Signature. |

---

## 7. Known Limitations & TODO

* **TODO — Direct Empirical Baseload from Zero-Runtime Intervals:**
    * *Background:* An alternative to linear regression (`kWh` vs. `Runtime_Total_Hr`) is measuring household baseload directly during intervals when the HVAC system is completely off (`Runtime == 0`).
    * *Why it is currently deferred:* This approach was previously attempted on real dataset exports but failed to produce reliable baseload subtraction because **household baseload is not constant across 24-hour cycles** due to daytime occupancy, electronics, cooking, and appliance usage. Because zero-runtime intervals are not uniformly distributed across the day (for example, in winter `Runtime == 0` occurs disproportionately during warmer afternoon hours when non-HVAC electronics/appliance loads are high, whereas in summer it occurs mostly overnight when household loads are low), a flat zero-runtime average introduces severe diurnal bias. Future work should explore time-of-day (diurnal) baseload stratification when hourly Green Button data is available.