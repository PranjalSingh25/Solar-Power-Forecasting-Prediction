import pvlib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from timezonefinder import TimezoneFinder # For timezone lookup
from typing import Dict # Ensure Dict is imported if using type hints

# --- Configuration ---

# 1. Input Weather File (Output from solar_data_pipeline_2.py - MUST BE HOURLY)
WEATHER_FILE = Path("data/nasa_power_hourly_raw.csv") # Using the HOURLY file now

# 2. Output File for Combined Data
OUTPUT_FILE = Path("data/processed/weather_and_simulated_hourly_power.csv") # Updated output name

# 3. Location Parameters (MUST match the location used for fetching weather data)
LATITUDE = 28.6139  # New Delhi example - CHANGE THIS
LONGITUDE = 77.2090 # New Delhi example - CHANGE THIS
ALTITUDE = 216     # Altitude in meters (optional but improves accuracy) - CHANGE THIS
# Try to automatically find timezone, otherwise set manually
try:
    tf = TimezoneFinder()
    TIMEZONE = tf.timezone_at(lng=LONGITUDE, lat=LATITUDE)
    if TIMEZONE is None:
        print("WARNING: Could not automatically determine timezone. Using 'UTC'. PLEASE SET MANUALLY for accuracy.")
        TIMEZONE = 'UTC' # <--- SET MANUALLY IF NEEDED (e.g., 'Asia/Kolkata')
except ImportError:
    print("WARNING: 'timezonefinder' not installed. Using 'UTC'. PLEASE SET MANUALLY for accuracy.")
    TIMEZONE = 'UTC' # <--- SET MANUALLY IF NEEDED
except Exception as e_tz:
    print(f"WARNING: Error finding timezone ({e_tz}). Using 'UTC'. PLEASE SET MANUALLY for accuracy.")
    TIMEZONE = 'UTC'

# 4. PV System Parameters (FILL THESE WITH YOUR SYSTEM DETAILS)
MODULE_DATABASE = 'CECMod'
MODULE_NAME = 'Canadian_Solar_Inc__CS6X_300M' # Example Module - CHANGE THIS

INVERTER_DATABASE = 'CECInverter'
INVERTER_NAME = 'SMA_America__SB7000TL_US__240V_' # Example Inverter - CHANGE THIS

SURFACE_TILT = 28          # Degrees from horizontal - CHANGE THIS
SURFACE_AZIMUTH = 180      # Degrees from North (180=South) - CHANGE THIS
MODULES_PER_STRING = 10    # - CHANGE THIS
STRINGS_PER_INVERTER = 2   # - CHANGE THIS

TEMPERATURE_MODEL_PARAMETERS = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

SYSTEM_LOSSES = 0.14 # Example: 14% total losses


# --- Logging Configuration ---
LOG_FILE = Path("logs/pv_simulation_hourly.log") # Updated log name
LOG_FILE.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PV_Simulation_Hourly") # Updated logger name

# --- Main Simulation Function ---

def simulate_pv_power(
    weather_file: Path,
    output_file: Path,
    latitude: float,
    longitude: float,
    altitude: float,
    timezone: str,
    module_db: str,
    module_name: str,
    inverter_db: str,
    inverter_name: str,
    tilt: float,
    azimuth: float,
    modules_per_string: int,
    strings_per_inverter: int,
    temp_model_params: dict,
    system_losses: float = 0.14
    ) -> bool:
    """
    Loads HOURLY weather data (inc. GHI, DNI, DHI), runs pvlib simulation, saves results.
    """
    logger.info(f"Starting PV simulation for location ({latitude}, {longitude}) using HOURLY data")
    mc = None # Initialize mc to None to handle potential errors gracefully

    # --- 1. Load and Prepare Weather Data ---
    try:
        logger.info(f"Loading weather data from: {weather_file}")
        if not weather_file.exists():
            logger.error(f"Weather data file not found: {weather_file}")
            return False

        if weather_file.suffix == ".parquet":
             # Assuming pyarrow or fastparquet is installed if this file exists
             weather_df = pd.read_parquet(weather_file)
        else:
             # *** Load using 'Timestamp' as index for hourly data ***
             weather_df = pd.read_csv(weather_file, index_col='Timestamp', parse_dates=True)

        logger.info(f"Loaded {len(weather_df)} data points.")

        # Ensure index is DatetimeIndex
        if not isinstance(weather_df.index, pd.DatetimeIndex):
            logger.error("Index could not be parsed as DatetimeIndex. Check CSV format.")
            # Attempt conversion again if needed, might depend on exact CSV format
            try:
                weather_df.index = pd.to_datetime(weather_df.index)
                logger.info("Index successfully converted to DatetimeIndex.")
            except Exception as e_idx:
                 logger.error(f"Failed to convert index to DatetimeIndex: {e_idx}")
                 return False

        # *** Ensure timezone localization (CRUCIAL) ***
        try:
             if weather_df.index.tz is None:
                  logger.info(f"Localizing timezone to {timezone}")
                  weather_df = weather_df.tz_localize(timezone) # Use localize if naive
             else:
                  logger.info(f"Converting timezone to {timezone}")
                  weather_df = weather_df.tz_convert(timezone) # Use convert if already localized
        except Exception as e_tz:
             logger.error(f"Error setting timezone '{timezone}': {e_tz}. Ensure timezone string is correct (e.g., 'Asia/Kolkata').", exc_info=True)
             return False


        # Rename columns for pvlib conventions
        # *** USING HOURLY DNI/DHI FROM NASA ***
        rename_map = {
            'ALLSKY_SFC_SW_DWN': 'ghi',        # Global Horizontal (W/m^2 average)
            'ALLSKY_SFC_SW_DNI': 'dni',        # Direct Normal (W/m^2 average)
            'ALLSKY_SFC_SW_DIFF': 'dhi',       # Diffuse Horizontal (W/m^2 average)
            'T2M': 'temp_air',                 # Celsius
            'WS10M': 'wind_speed'              # m/s
            # 'RH2M': 'relative_humidity'      # Optional: add if needed
        }
        pvlib_weather = weather_df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {list(rename_map.values())}")

        # Check for required columns after rename
        required_cols = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']
        missing_cols = [col for col in required_cols if col not in pvlib_weather.columns]
        if missing_cols:
            logger.error(f"Missing required columns after rename: {missing_cols}. Check input CSV and rename_map.")
            logger.error(f"Available columns: {pvlib_weather.columns.tolist()}")
            return False

        # Handle NaNs in essential columns (Drop rows missing ANY required input for simulation)
        initial_rows = len(pvlib_weather)
        pvlib_weather = pvlib_weather.dropna(subset=required_cols)
        dropped_rows = initial_rows - len(pvlib_weather)
        if dropped_rows > 0:
             logger.warning(f"Dropped {dropped_rows} rows due to missing values in required columns: {required_cols}")

        if pvlib_weather.empty:
             logger.error("Weather data is empty after dropping NaNs in required columns.")
             return False
        logger.info(f"Prepared weather data shape: {pvlib_weather.shape}")


    except FileNotFoundError: # Catch specific error
        logger.error(f"Weather data file not found: {weather_file}")
        return False
    except KeyError as e_col: # Catch specific error
         logger.error(f"Column error during loading/renaming: {e_col}. Check CSV header or index column name ('Timestamp'?).")
         return False
    except Exception as e:
        logger.error(f"Error loading or preparing weather data: {e}", exc_info=True)
        return False

    # --- 2. Define Location and PV System ---
    # *** THIS SECTION IS NOW UNCOMMENTED ***
    try:
        logger.info("Defining PV system location and parameters...")
        location = pvlib.location.Location(
            latitude=latitude, longitude=longitude, tz=timezone, altitude=altitude, name=f"SimLocation_{latitude}_{longitude}"
        )

        # Get module and inverter parameters from database
        module_params = pvlib.pvsystem.retrieve_sam(module_db)[module_name]
        inverter_params = pvlib.pvsystem.retrieve_sam(inverter_db)[inverter_name]
        logger.info(f"Using Module: {module_name} | Inverter: {inverter_name}")

        # Define the system
        # Using basic losses_parameters here, system_losses applied later if using ModelChain default
        pv_system = pvlib.pvsystem.PVSystem(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            module_parameters=module_params,
            inverter_parameters=inverter_params,
            temperature_model_parameters=temp_model_params,
            modules_per_string=modules_per_string,
            strings_per_inverter=strings_per_inverter
            # losses_parameters can be added for specific loss types if needed
        )
        logger.info("PVSystem object created.")

    except KeyError as e:
         logger.error(f"Model name error: '{e}' not found in the specified SAM database ({module_db} or {inverter_db}). "
                      f"Please check model names available at https://sam.nrel.gov/photovoltaic/pv-sub-page-2-performance-database.html")
         return False
    except Exception as e:
        logger.error(f"Error defining PV system: {e}", exc_info=True)
        return False

    # --- 3. Run Simulation using ModelChain ---
    # *** THIS SECTION IS NOW UNCOMMENTED ***
    try:
        logger.info("Initializing ModelChain...")
        # Define the ModelChain - using default losses for now
        # system_losses (passed as argument) is not directly used by default ModelChain setup,
        # but you could potentially integrate it via custom functions or specific loss parameters.
        mc = pvlib.modelchain.ModelChain(
            pv_system,
            location,
            aoi_model="physical",
            spectral_model="no_loss"
            # transducer=True # Include if you want DC results as well
        )
        logger.info("Running ModelChain simulation with provided GHI, DNI, DHI...")

        # *** NO DECOMPOSITION NEEDED HERE - run_model uses columns directly ***
        mc.run_model(pvlib_weather)
        logger.info("Simulation complete.")

    except AttributeError as e:
         logger.error(f"AttributeError during simulation (check objects like location, pv_system?): {e}", exc_info=True)
         return False
    except ValueError as e:
         # Catch potential issues during model run (e.g., incompatible data)
         logger.error(f"ValueError during ModelChain run: {e}", exc_info=True)
         return False
    except Exception as e:
        logger.error(f"Error during ModelChain simulation: {e}", exc_info=True)
        return False

    # --- 4. Process Results and Save ---
    try:
        # Check if mc object was successfully created and run
        if mc is None or mc.results is None:
             logger.error("ModelChain object or results are missing. Simulation likely failed earlier.")
             return False

        logger.info("Processing simulation results...")
        # Extract simulated AC power (in Watts)
        simulated_power_ac = mc.results.ac

        # Handle potential NaNs from simulation (e.g., clipping, thresholds) and ensure >= 0
        simulated_power_ac = simulated_power_ac.fillna(0)
        simulated_power_ac[simulated_power_ac < 0] = 0
        logger.info(f"Simulated AC power range (W): {simulated_power_ac.min():.2f} to {simulated_power_ac.max():.2f}")
        logger.info(f"Number of non-zero power hours: {(simulated_power_ac > 0).sum()}")


        # Rename for clarity
        simulated_power_ac = simulated_power_ac.rename('simulated_ac_power_W')

        # Combine with original relevant weather data
        # Re-load original dataframe to ensure all columns are available before join
        if weather_file.suffix == ".parquet":
             original_weather_df = pd.read_parquet(weather_file)
        else:
             original_weather_df = pd.read_csv(weather_file, index_col='Timestamp', parse_dates=True)

        # Localize original data index for safe joining
        if original_weather_df.index.tz is None:
             original_weather_df = original_weather_df.tz_localize(timezone)
        else:
             original_weather_df = original_weather_df.tz_convert(timezone)

        # Join based on the index (Timestamp)
        combined_df = original_weather_df.join(simulated_power_ac, how='inner')

        if combined_df.empty:
            logger.error("Combined DataFrame is empty after joining original weather and simulation results. Check index alignment.")
            return False
        if 'simulated_ac_power_W' not in combined_df.columns:
             logger.error("Simulated power column missing after join. Check index alignment and simulation results.")
             return False

        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving combined data ({len(combined_df)} rows) to: {output_file}")
        if output_file.suffix == ".parquet":
             combined_df.to_parquet(output_file)
        else:
             combined_df.to_csv(output_file)

        logger.info("Successfully saved combined weather and simulated power data.")
        return True

    except Exception as e:
        logger.error(f"Error processing or saving results: {e}", exc_info=True)
        return False


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("--- Starting PV Simulation Script (Hourly) ---") # Updated title

    # Make sure the WEATHER_FILE points to the HOURLY CSV generated by the updated pipeline
    if not WEATHER_FILE.exists():
         logger.error(f"Input weather file '{WEATHER_FILE}' not found. Please run the HOURLY data fetching script first.")
    else:
        success = simulate_pv_power(
            weather_file=WEATHER_FILE,
            output_file=OUTPUT_FILE,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            altitude=ALTITUDE,
            timezone=TIMEZONE,
            module_db=MODULE_DATABASE,
            module_name=MODULE_NAME,
            inverter_db=INVERTER_DATABASE,
            inverter_name=INVERTER_NAME,
            tilt=SURFACE_TILT,
            azimuth=SURFACE_AZIMUTH,
            modules_per_string=MODULES_PER_STRING,
            strings_per_inverter=STRINGS_PER_INVERTER,
            temp_model_params=TEMPERATURE_MODEL_PARAMETERS,
            system_losses=SYSTEM_LOSSES
        )

        if success:
            logger.info("--- PV Simulation Script Finished Successfully ---")
        else:
            logger.error("--- PV Simulation Script Failed ---")