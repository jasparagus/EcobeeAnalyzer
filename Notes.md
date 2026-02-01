# Project: Residential Heat Transfer Estimation (Thermal RC Model)

## 1. Project Objective
To estimate the **Thermal Time Constant ($\tau$)** and **Heat Transfer Coefficient ($UA$)** of a residential building envelope using historical thermostat data. This project moves beyond steady-state approximations by implementing a dynamic **Thermal RC (Resistor-Capacitor) Model** that accounts for the building's thermal mass and "coasting" behavior when the HVAC system is off.

## 2. Theoretical Framework

### 2.1 The Thermal RC Model
The house is modeled as a single thermal node with capacitance $C$ (Thermal Mass) connected to the ambient environment via a thermal resistance $R = 1/UA$. The energy balance equation is:

$$C \frac{dT_{in}}{dt} = UA(T_{out} - T_{in}) + Q_{HVAC} + Q_{internal} + Q_{solar}$$

### 2.2 The "Coasting" Derivation (HVAC Off)
To isolate the building envelope characteristics ($UA$ and $C$) from the powerful HVAC vector ($Q_{HVAC}$), we analyze periods where the system is **OFF** ($Q_{HVAC} \approx 0$).

By filtering for nighttime periods, we also eliminate solar gain ($Q_{solar} \approx 0$). The equation simplifies to:

$$C \frac{dT_{in}}{dt} = UA(T_{out} - T_{in}) + Q_{internal}$$

Dividing by $C$ yields the linear regression form $y = mx + b$:

$$\underbrace{\frac{dT_{in}}{dt}}_{Y} = \underbrace{\frac{UA}{C}}_{\text{Slope } m} \cdot \underbrace{(T_{out} - T_{in})}_{X} + \underbrace{\frac{Q_{internal}}{C}}_{\text{Intercept } b}$$

Where:
* **Slope ($m$):** The inverse of the thermal time constant ($\tau$).
    $$\tau = \frac{1}{m} = \frac{C}{UA}$$
* **Intercept ($b$):** Represents internal heat gain (occupants, appliances).

## 3. Software Architecture

### 3.1 Environment Overview
The solution is implemented as a lightweight local Python application consisting of two primary modules:
1.  **`thermostat_viewer.py`**: The GUI entry point. Handles file I/O, data visualization, and user interaction.
2.  **`thermal_analysis.py`**: The physics engine. Handles data cleaning, filtering, resampling, and regression analysis.

### 3.2 Module Specifications

#### A. `thermostat_viewer.py` (Presentation Layer)
* **Library:** `matplotlib`, `tkinter`, `pandas`
* **Key Classes:** `ThermostatApp`
* **Responsibilities:**
    * **Data Ingestion:** Loads raw CSV exports from Ecobee (skipping metadata rows).
    * **Preprocessing:** converting `Date` + `Time` columns into a unified `Timestamp` index.
    * **Visualization:**
        * *Top Subplot:* Indoor/Outdoor/Set Temperatures with colored background fills for HVAC State (Heating=Red, Cooling=Blue).
        * *Bottom Subplot:* "Program Mode" area plot (Sleep/Home/Away).
    * **Interaction:** Provides buttons for "Load Data", "Analyze Full", and "Analyze Zoomed" (filtering data based on current plot X-axis limits).
    * **Configuration:** Checkboxes for "Night Only" and "30m Buffer" filters.

#### B. `thermal_analysis.py` (Logic Layer)
* **Library:** `scipy.stats`, `pandas`, `numpy`
* **Key Classes:** `ThermalAnalyzer`
* **Algorithm Pipeline:**
    1.  **Resampling:** Downsamples raw 5-minute data to **15-minute averages**.
        * *Rationale:* Smooths quantization noise from digital sensors (0.5°F steps) to produce stable derivatives ($\frac{dT}{dt}$).
    2.  **Derivative Calculation:**
        * $Y = \frac{\Delta T_{in}}{\Delta t}$ (Normalized to °F/hr)
        * $X = \Delta T = T_{out} - T_{in}$
    3.  **Strict Filtering (The "Washout" Logic):**
        * **HVAC Off Check:** `Cool Stage` & `Heat Stage` < 60s runtime per 15m.
        * **Washout Buffer:** Checks that the system has been OFF for the preceding $N$ minutes (default 30) using `.rolling().min()`. *Rationale: Removes thermal transients from the heat exchanger/coil.*
        * **Solar Exclusion (Night Mode):** Restricts data to 11:00 PM – 06:00 AM. *Rationale: Removes $Q_{solar}$ noise.*
        * **Transient Removal:** Filters unphysical rates of change ($|\frac{dT}{dt}| > 5$ °F/hr).
    4.  **Regression:**
        * Separates data into **Heating Regime** ($\Delta T < -2$) and **Cooling Regime** ($\Delta T > 2$).
        * Performs Linear Regression (`scipy.stats.linregress`) on each regime.
    5.  **Output Generation:**
        * Generates a text report with $\tau$, Slope, $R^2$, and point counts.
        * Generates a scatter plot with regression lines overlaid on raw data.

## 4. Data Flow & Usage

1.  **Input:** User exports CSV files from Ecobee Home IQ -> System Monitor.
2.  **Load:** User selects multiple CSVs via GUI. App merges and deduplicates based on Timestamp.
3.  **Explore:** User zooms/pans the Matplotlib graph to identify clean data periods.
4.  **Analyze:**
    * **Full Analysis:** Runs on the entire loaded dataset.
    * **Zoomed Analysis:** Runs only on the time window currently visible in the plot.
5.  **Output:**
    * `thermal_results_{mode}_{date_range}.txt`: Statistical summary.
    * `thermal_plot_{mode}_{date_range}.png`: Visual regression fit.

## 5. Key Conclusions (Sample Interpretation)
* **Heating (Winter) Fit:** Typically more reliable due to larger $\Delta T$ and lack of solar interference at night.
* **Time Constant ($\tau$):** A value of ~20-40 hours indicates a well-insulated home with significant thermal mass.
* **Cooling (Summer) Fit:** Often noisier due to solar gain (even with filters) and lower $\Delta T$.