import requests
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict # Ensure Dict is imported
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

# --- Constants ---
# Base directory relative to this script file
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
# CHANGED Filenames for HOURLY data
RAW_CSV_PATH = DATA_DIR / "nasa_power_hourly_raw.csv"
RAW_PARQUET_PATH = DATA_DIR / "nasa_power_hourly_raw.parquet"
LOG_FILE_PATH = LOG_DIR / "solar_data_fetching_hourly.log" # Changed log name

# Default parameters INCLUDING irradiance components
DEFAULT_NASA_PARAMS = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M", "WS10M", "RH2M"]
# REDUCED Max date range - CHECK NASA API DOCS FOR HOURLY LIMIT!
MAX_DATE_RANGE_DAYS = 9200 # Reduced significantly for hourly data

# --- Logging Configuration ---
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'), # Use new log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NASA_HourlyFetcher") # Updated logger name

# --- Parquet Support Check ---
try:
    import pyarrow
    PARQUET_SUPPORT = True
except ImportError:
    try:
        import fastparquet
        PARQUET_SUPPORT = True
    except ImportError:
        PARQUET_SUPPORT = False
        logger.warning(
            "Parquet file support is disabled. Install 'pyarrow' or 'fastparquet'."
        )

class NasaPowerHourlyFetcher: # Renamed class for clarity
    def __init__(self):
        self._create_directories()
        self.session = self._create_robust_session()
        logger.info("NasaPowerHourlyFetcher initialized.")

    def _create_directories(self):
        """Create necessary directories."""
        try:
            DATA_DIR.mkdir(exist_ok=True)
            logger.info(f"Ensured directory exists: {DATA_DIR}")
        except Exception as e:
            logger.error(f"Error creating directory {DATA_DIR}: {e}", exc_info=True)
            raise

    def _create_robust_session(self):
        """Create a robust requests session with retry mechanism."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        logger.debug("Robust requests session created.")
        return session

    def validate_and_prepare_inputs(self, latitude: float, longitude: float,
                                     start_date_str: str, end_date_str: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Validates inputs and adjusts end date if range exceeds limit.
        NOTE: MAX_DATE_RANGE_DAYS applies to the request, check API limits for HOURLY.
        """
        # --- Validation logic remains the same ---
        if not (-90 <= latitude <= 90):
            return False, None, None, f"Invalid latitude: {latitude}. Must be between -90 and 90."
        if not (-180 <= longitude <= 180):
            return False, None, None, f"Invalid longitude: {longitude}. Must be between -180 and 180."

        try:
            start_dt = datetime.strptime(start_date_str, "%Y%m%d")
            end_dt = datetime.strptime(end_date_str, "%Y%m%d")
        except ValueError:
            return False, None, None, f"Invalid date format. Use YYYYMMDD. Got: '{start_date_str}', '{end_date_str}'"

        if start_dt > end_dt:
            return False, None, None, f"Start date ({start_date_str}) must be before or same as end date ({end_date_str})."

        # NASA POWER hourly often available for recent ~20 years, daily goes back further
        # Adjust start year constraint if needed based on hourly availability. 1981 might be too early.
        # Let's use 2000 as a safer default starting point for hourly, but verify this.
        if start_dt.year < 2000:
            logger.warning(f"Start year {start_dt.year} might be too early for hourly data. API data often starts later (e.g., around 2000).")
            # return False, None, None, f"Start date ({start_date_str}) must be 2000 or later for reliable NASA POWER hourly data."

        # Check and adjust date range limit (Now using the reduced MAX_DATE_RANGE_DAYS)
        if (end_dt - start_dt).days > MAX_DATE_RANGE_DAYS:
            original_end_date = end_date_str
            end_dt = start_dt + timedelta(days=MAX_DATE_RANGE_DAYS)
            end_date_str = end_dt.strftime("%Y%m%d")
            logger.warning(
                f"Requested date range exceeds {MAX_DATE_RANGE_DAYS} days limit for hourly request. "
                f"End date adjusted from {original_end_date} to {end_date_str}."
            )
            logger.warning("You may need to make multiple requests to fetch the full desired period.")

        return True, start_dt.strftime("%Y%m%d"), end_date_str, None


    def fetch_data(self, latitude: float, longitude: float,
                   start_date: str, end_date: str,
                   parameters: List[str] = DEFAULT_NASA_PARAMS) -> Optional[pd.DataFrame]:
        """
        Fetches HOURLY data from NASA POWER API, validates inputs, and saves raw data.
        Assumes API returns irradiance in units representing average W/m^2 for the hour.
        """
        is_valid, valid_start, valid_end, error_msg = self.validate_and_prepare_inputs(
            latitude, longitude, start_date, end_date
        )

        if not is_valid:
            logger.error(f"Input validation failed: {error_msg}")
            return None

        # *** CHANGED API ENDPOINT ***
        api_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        api_params = {
            "parameters": ",".join(parameters),
            "community": "RE",
            "longitude": longitude,
            "latitude": latitude,
            "start": valid_start,
            "end": valid_end,
            "format": "JSON",
            "time-standard": "LST" # Using Local Solar Time is often best for PV simulation
        }

        logger.info(f"Fetching HOURLY data for Lat: {latitude}, Lon: {longitude}, "
                    f"Dates: {valid_start} to {valid_end}, Params: {parameters}")

        try:
            response = self.session.get(api_url, params=api_params, timeout=60) # Increased timeout for potentially larger response
            response.raise_for_status()

            data = response.json()
            if "messages" in data and data["messages"]:
                 logger.warning(f"API returned messages: {data['messages']}")

            parameter_data = data.get("properties", {}).get("parameter", {})

            if not parameter_data:
                logger.warning("No HOURLY data retrieved from NASA POWER API for the specified parameters/range.")
                return None

            # Robust DataFrame creation
            dfs = []
            valid_params_found = []
            for param in parameters:
                if param in parameter_data:
                    param_dict = parameter_data[param]
                    # Replace NASA POWER fill value -999 with NaN (common)
                    param_dict = {k: np.nan if v == -999 else v for k, v in param_dict.items()}
                    # Convert values to numeric, coercing errors
                    df_param = pd.DataFrame.from_dict(param_dict, orient='index', columns=[param])
                    df_param[param] = pd.to_numeric(df_param[param], errors='coerce')
                    dfs.append(df_param)
                    valid_params_found.append(param)
                else:
                    logger.warning(f"Parameter '{param}' not found in HOURLY API response.")

            if not dfs:
                logger.error("No valid parameters found in response. Cannot create DataFrame.")
                return None

            df = pd.concat(dfs, axis=1)

            # *** ADJUSTED DATETIME PARSING ***
            # Assumes API returns keys like 'YYYYMMDDHH' - VERIFY THIS FORMAT!
            try:
                 # Attempt common format YYYYMMDDHH
                 df.index = pd.to_datetime(df.index, format='%Y%m%d%H')
            except ValueError:
                 logger.warning("Could not parse index with format '%Y%m%d%H'. Trying ISO format.")
                 # Fallback try, maybe API returns ISO-like strings?
                 try:
                      df.index = pd.to_datetime(df.index)
                 except Exception as e_dt:
                      logger.error(f"Failed to parse datetime index from API response keys: {df.index[:5]}... Error: {e_dt}", exc_info=True)
                      logger.error("Check NASA POWER API documentation for the exact HOURLY timestamp format.")
                      return None

            df.index.name = 'Timestamp' # More appropriate name for hourly
            df.sort_index(inplace=True)

            logger.info(f"Successfully retrieved {len(df)} HOURLY data points for parameters: {valid_params_found}")
            # **UNIT ASSUMPTION NOTE:** Assuming ALLSKY_SFC_SW_* parameters are average W/m^2 for the hour. Verify this!

            # Save raw data
            try:
                df.to_csv(RAW_CSV_PATH) # Save to new hourly filename
                logger.info(f"Raw HOURLY data saved to {RAW_CSV_PATH}")
                if PARQUET_SUPPORT:
                    try:
                        df.to_parquet(RAW_PARQUET_PATH) # Save to new hourly filename
                        logger.info(f"Raw HOURLY data also saved as Parquet to {RAW_PARQUET_PATH}")
                    except Exception as e_parquet:
                        logger.warning(f"Could not save Parquet file: {e_parquet}")
            except IOError as e_io:
                 logger.error(f"Error saving data file: {e_io}", exc_info=True)
                 return None

            return df

        except requests.exceptions.Timeout:
             logger.error(f"NASA API request timed out.")
             return None
        except requests.exceptions.RequestException as e_req:
            logger.error(f"NASA API request failed: {e_req}", exc_info=True)
            return None
        except Exception as e_unexpected:
            logger.error(f"An unexpected error occurred during HOURLY data fetching: {e_unexpected}", exc_info=True)
            return None

    # --- interactive_input remains largely the same, just prompts apply to hourly fetch ---
    def interactive_input(self) -> Optional[Dict]:
        print("\n--- NASA POWER HOURLY Data Fetcher Configuration ---") # Updated title
        print(f"Data will be saved in: {DATA_DIR}")
        print(f"NOTE: Hourly data requests might be limited (e.g., {MAX_DATE_RANGE_DAYS} days per request).")
        print("      Check NASA POWER API documentation for exact limits.")
        print("      Hourly data availability often starts later than daily (e.g., ~year 2000).")

        while True:
            try:
                # Input prompts are the same
                print("\n Location Details:")
                lat_str = input("Enter Latitude (-90 to 90): ")
                lon_str = input("Enter Longitude (-180 to 180): ")
                latitude = float(lat_str)
                longitude = float(lon_str)

                print(f"\n Date Range (YYYYMMDD format):")
                start_date_str = input("Enter Start Date (e.g., 20210101): ")
                end_date_str = input("Enter End Date (e.g., 20211231): ")

                print("\n Weather Parameters (Optional):")
                param_input = input(f"Enter NASA Parameters (comma-separated, default includes GHI/DNI/DIFF):\n({','.join(DEFAULT_NASA_PARAMS)}): ")
                parameters = [p.strip().upper() for p in param_input.split(',')] if param_input else DEFAULT_NASA_PARAMS

                is_valid, _, _, error_msg = self.validate_and_prepare_inputs(
                    latitude, longitude, start_date_str, end_date_str
                )
                if not is_valid:
                    print(f" Input Error: {error_msg}")
                    continue

                return {
                    'latitude': latitude,
                    'longitude': longitude,
                    'start_date': start_date_str,
                    'end_date': end_date_str,
                    'parameters': parameters
                }

            except ValueError:
                print(" Invalid numerical input. Please enter numbers for latitude and longitude.")
            except Exception as e:
                 print(f" An unexpected error occurred during input: {e}")

# --- Main Execution ---
def main():
    print("NASA POWER HOURLY Data Fetcher ") # Updated title

    fetcher = NasaPowerHourlyFetcher() # Use renamed class

    while True:
        try:
            config = fetcher.interactive_input()
            if config is None:
                 print("Exiting due to input error.")
                 break

            print("\n Fetching HOURLY data...") # Updated message
            nasa_df = fetcher.fetch_data(
                config['latitude'],
                config['longitude'],
                config['start_date'],
                config['end_date'],
                config['parameters']
            )

            if nasa_df is not None and not nasa_df.empty:
                print("\n Data fetching completed successfully!")
                print("\n Fetched HOURLY Data Summary:") # Updated message
                print(f"  Data Points: {len(nasa_df)}")
                print(f"  Time Range: {nasa_df.index.min()} to {nasa_df.index.max()}")
                print(f"  Columns: {', '.join(nasa_df.columns)}")
                print(f"  Raw data saved in '{DATA_DIR}' as '{RAW_CSV_PATH.name}' (and optionally Parquet).")
                missing_vals = nasa_df.isnull().sum()
                if missing_vals.sum() > 0:
                     print("\n  Warning: Missing values detected:")
                     print(missing_vals[missing_vals > 0])
                else:
                     print("  No missing values detected in fetched hourly data.")

            else:
                print("\n Data fetching failed or returned no data. Check logs for details.")
                print(f"   Log file: {LOG_FILE_PATH}")

        except KeyboardInterrupt:
             print("\nOperation cancelled by user.")
             break
        except Exception as e:
             logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
             print(f" An unexpected error occurred. Check logs: {LOG_FILE_PATH}")

        # Ask to run again
        try:
            retry = input("\nFetch HOURLY data for another location/period? (yes/no): ").lower().strip()
            if retry != 'yes':
                break
        except EOFError:
             break

    print("\n Exiting NASA POWER HOURLY Data Fetcher.")

if __name__ == "__main__":
    main()