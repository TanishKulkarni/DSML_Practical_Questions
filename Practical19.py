import pandas as pd
import os

# --- Adjust the CSV path to your actual file ---
csv_path = r'venv\Datasets\IRIS.csv'   # <-- change if needed

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

# Load the data
df = pd.read_csv(csv_path)

# Normalize column name for species if necessary (case-insensitive)
species_col = None
for col in df.columns:
    if col.strip().lower() == "species":
        species_col = col
        break
if species_col is None:
    raise KeyError("No column named 'species' found. Check your column names: " + ", ".join(df.columns))

print("Using species column:", species_col)
print("\nColumns and dtypes before conversion:\n", df.dtypes, "\n")

# Attempt to convert all non-species object columns to numeric when possible
for col in df.columns:
    if col == species_col:
        continue
    # If object dtype, try to coerce to numeric (removes commas, extra spaces, etc.)
    if df[col].dtype == 'object':
        # make a cleaned version: strip spaces, remove common non-numeric characters
        cleaned = df[col].astype(str).str.strip().str.replace(r'[^\d\.\-eE]', '', regex=True)
        coerced = pd.to_numeric(cleaned, errors='coerce')
        # If a reasonable number of values converted (not all NaN), replace the column
        num_converted = coerced.notna().sum()
        if num_converted > 0:
            df[col] = coerced
            print(f"Converted column '{col}' to numeric (non-NaN values: {num_converted})")
        else:
            # leave as-is (likely truly categorical)
            pass

print("\nColumns and dtypes after conversion attempt:\n", df.dtypes, "\n")

# Helper to compute stats for a given species
def stats_for_species(species_name):
    print(f"\n---- Statistical Details for: {species_name} ----\n")
    species_df = df[df[species_col] == species_name]

    if species_df.empty:
        print(f"No rows found for species '{species_name}'")
        return

    # Select numeric columns only
    numeric_cols = species_df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        print("No numeric columns available for statistical summary.")
        return

    # Basic statistical summary for numeric columns
    print("Basic summary (describe) for numeric columns:\n")
    print(species_df[numeric_cols].describe().T)  # transpose for nicer layout

    # Additional percentiles (10th and 90th) and any other percentiles you want
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    print("\nSelected percentiles (10%,25%,50%,75%,90%):\n")
    print(species_df[numeric_cols].quantile(percentiles).T)  # transpose for nicer layout

# Call the function for the three species (use the names present in your file)
unique_species = df[species_col].unique()
print("Unique species in dataset:", unique_species, "\n")

# If exact names are present, call them; otherwise iterate all
for sp in unique_species:
    stats_for_species(sp)
