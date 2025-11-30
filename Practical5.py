
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import sys
from pandas.api import types as ptypes

# ------------ User settings ------------
CSV_FILE = "venv\Datasets\Lipstick.csv"   # <<-- change if your file has a different name
TARGET_COL = "Buys"          # <<-- set to your target column name if different; otherwise it will fall back to last column
DROP_COLS = ["Id"]           # columns to drop if present (non-features)
# ---------------------------------------

# ------------------ Helpers ------------------
def safe_load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: file '{path}' not found. Put your dataset in the same folder or update CSV_FILE.")
        sys.exit(1)

def ensure_target_in_df(df, target):
    if target not in df.columns:
        # fallback: use last column
        fallback = df.columns[-1]
        print(f"Warning: target '{target}' not found. Using last column '{fallback}' as target.")
        return fallback
    return target

def encode_dataframe(df, label_encoders):
    """
    - label_encoders: dict to fill with LabelEncoders for categorical columns
    - returns encoded copy of df
    """
    df_enc = df.copy()
    for col in df_enc.columns:
        if ptypes.is_object_dtype(df_enc[col]) or ptypes.is_categorical_dtype(df_enc[col]):
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            label_encoders[col] = le
        else:
            # numeric columns: keep as-is
            pass
    return df_enc

def encode_test_value(col, val, col_dtype, label_encoders):
    """
    Encodes a single test value according to the training column dtype / encoder.
    - If column is numeric, expect val to be numeric or a special token like "<21" which we map to a numeric example.
    - If column is categorical (LabelEncoder exists), encode with that encoder (error if unseen).
    """
    if col in label_encoders:
        le = label_encoders[col]
        val_str = str(val)
        if val_str not in le.classes_:
            # unseen label — try to handle simple case differences (strip whitespace)
            trimmed = val_str.strip()
            if trimmed in le.classes_:
                return int(le.transform([trimmed])[0])
            # can't encode unseen label reliably — raise informative error
            raise ValueError(f"Test value '{val}' for column '{col}' was not seen in training classes: {list(le.classes_)}")
        return int(le.transform([val_str])[0])
    else:
        # column treated as numeric in training
        if isinstance(val, str):
            # handle patterns like "<21", "<=21", ">30" by mapping to a representative numeric value
            s = val.strip()
            if s.startswith("<"):
                # choose a value one less than the boundary if numeric boundary present
                try:
                    bound = float(s[1:])
                    return bound - 1.0
                except:
                    # fallback numeric
                    return 0.0
            if s.startswith(">"):
                try:
                    bound = float(s[1:])
                    return bound + 1.0
                except:
                    return 100.0
            if s.isdigit():
                return float(s)
            # not parseable: try float conversion
            try:
                return float(s)
            except:
                raise ValueError(f"Cannot convert test value '{val}' for numeric column '{col}' to a number.")
        else:
            return float(val)

# ------------------ Main ------------------
def main():
    # 1) Load dataset
    df = safe_load_csv(CSV_FILE)
    print("Columns in dataset:", list(df.columns))
    # Determine target column
    target = TARGET_COL if TARGET_COL in df.columns else ensure_target_in_df(df, TARGET_COL)

    # 2) Drop unwanted columns if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
            print(f"Dropped column '{c}' from dataset (not used as feature).")

    # If target was dropped accidentally, ensure it's set correctly
    if target not in df.columns:
        target = ensure_target_in_df(df, target)

    print("Using target column:", target)

    # 3) Store original dtypes for columns (to handle test sample mapping)
    original_dtypes = df.dtypes.to_dict()

    # 4) Prepare training dataframe: encode categorical columns only (keep numeric)
    label_encoders = {}
    df_enc = encode_dataframe(df, label_encoders)

    # 5) Split features and target
    if target not in df_enc.columns:
        print(f"Error: target column '{target}' not found after processing. Columns: {list(df_enc.columns)}")
        sys.exit(1)
    X = df_enc.drop(columns=[target])
    y = df_enc[target]

    print("\nFinal feature columns (used for training):", list(X.columns))
    print("Number of training rows:", len(df_enc))

    # 6) Train the Decision Tree
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
    clf.fit(X, y)
    print("Decision Tree trained successfully.")

    # 7) Prepare the test sample as per assignment:
    # Test Data: [Age < 21, Income = Low, Gender = Female, MaritalStatus = Married]
    # IMPORTANT: the exact column names must match your dataset's column names.
    # Adjust the keys in test_sample below if your dataset uses different column names.
    test_sample_raw = {
        "Age": "<21",
        "Income": "Low",
        "Gender": "Female",
        "Ms": "Married"
    }

    # Verify that all keys exist (if not, try to find close matches or error)
    missing_cols = [c for c in test_sample_raw.keys() if c not in X.columns]
    if missing_cols:
        print(f"\nWarning: The following test columns are not present in training features: {missing_cols}")
        print("Training features are:", list(X.columns))
        # Try some common alternatives (case-insensitive) to map names
        lowered = {col.lower(): col for col in X.columns}
        mapping = {}
        for mc in missing_cols:
            if mc.lower() in lowered:
                mapping[mc] = lowered[mc.lower()]
        if mapping:
            print("Auto-mapping test columns:", mapping)
            for old, new in mapping.items():
                test_sample_raw[new] = test_sample_raw.pop(old)
            missing_cols = [c for c in test_sample_raw.keys() if c not in X.columns]
        if missing_cols:
            print("Cannot find matching columns for:", missing_cols)
            print("Please adjust test_sample_raw keys to match your dataset's feature names (exact match). Exiting.")
            sys.exit(1)

    # 8) Encode test sample values consistent with training
    test_encoded = {}
    for col in X.columns:
        if col in test_sample_raw:
            dt = original_dtypes.get(col)
            try:
                test_encoded[col] = encode_test_value(col, test_sample_raw[col], dt, label_encoders)
            except ValueError as e:
                print("Encoding error:", e)
                sys.exit(1)
        else:
            # If a feature was not provided in test_sample, we must supply a value.
            # Strategy: use the column's most frequent value from training (after encoding).
            # For numeric columns use mean; for categorical use mode.
            if col in label_encoders:
                # categorical: mode in original (unencoded) df
                mode_val = df[col].mode().iloc[0]
                test_encoded[col] = int(label_encoders[col].transform([str(mode_val)])[0])
                print(f"Note: feature '{col}' missing in test input — filled with training mode '{mode_val}'.")
            else:
                # numeric: fill with mean
                mean_val = X[col].mean()
                test_encoded[col] = float(mean_val)
                print(f"Note: numeric feature '{col}' missing in test input — filled with training mean {mean_val:.4f}.")

    # Create DataFrame for prediction
    test_df = pd.DataFrame([test_encoded], columns=X.columns)
    print("\nEncoded test sample (features in training order):")
    print(test_df)

    # 9) Predict
    pred_encoded = clf.predict(test_df)[0]

    # 10) Decode prediction back to original label (if target was categorical)
    if target in label_encoders:
        target_le = label_encoders[target]
        pred_label = target_le.inverse_transform([pred_encoded])[0]
    else:
        pred_label = pred_encoded

    print("\n=== FINAL PREDICTION ===")
    print("Test input (raw):", test_sample_raw)
    print("Predicted target (Buys?):", pred_label)

if __name__ == "__main__":
    main()
