# Ecobee Thermal Envelope & Inverter Profiler

A local, physics-based analysis tool that turns your Ecobee thermostat data into a building performance lab. This tool estimates your home's **Thermal Time Constant**, **Heat Transfer Coefficient (UA)**, and **Inverter Heat Pump performance profiles** using historical runtime and temperature data.

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
    * Calculates the house's **Time Constant ($\tau$)**.
    * Separates analysis into Heating (Winter) and Cooling (Summer) regimes.
    * **"Washout" Filtering:** Automatically excludes data periods immediately following HVAC cycles to prevent thermal inertia from corrupting the analysis.
    * **Solar Exclusion:** Optional "Night Only" mode to isolate conductive heat loss from solar gain.
* **Inverter Profiling (Energy Signature):**
    * Ingests "Green Button" XML power data from utility providers.
    * Calculates **Baseload Power** to isolate HVAC energy.
    * Generates a performance curve: **Daily Energy (kWh) vs. Temperature Delta**, showing the "cost to condition" the home at various outdoor temperatures.

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

### 3.3 Energy Signature (Daily kWh vs Delta T)
For inverter systems, instantaneous power varies. A more robust metric is the **Energy Signature**, which correlates daily energy input with the daily thermal load (Delta T).

$$E_{hvac}(kWh) \approx E_{total} - (P_{base} \times 24h)$$

Plotting $E_{hvac}$ vs $(T_{out} - T_{in})$ reveals the system's efficiency curve and the building's aggregate thermal load.

---

## 4. Project Structure & Usage

### Files
* **`ecobee_viewer.py`**: The GUI entry point. Handles file loading, plotting, and user interaction.
* **`thermal_analysis.py`**: The physics engine. Handles resampling, filtering (washout/night logic), and regression analysis.

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install pandas matplotlib scipy numpy
    ```
3.  Run the viewer:
    ```bash
    python ecobee_viewer.py
    ```

### Workflow
1.  **Export Data:**
    * **Ecobee:** Log in to the web portal -> Home IQ -> System Monitor -> Download Data.
    * **Power (Optional):** Download "Green Button" XML (Daily resolution) from your utility provider.
2.  **Load:** Use the GUI buttons to load the CSVs and XMLs.
3.  **Configure:**
    * Set **Square Footage** (Default: 1500 sq ft).
    * Toggle **Night Only** to remove solar noise (Recommended).
    * Toggle **30m Buffer** to ignore lingering heat after HVAC cycles.
4.  **Analyze:** Click "Calculate Profile".
5.  **Results:** Check the `./Results` folder for generated plots (`.png`) and text reports (`.txt`).