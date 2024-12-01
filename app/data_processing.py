import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer



target_col = "label"  # "status"
ok_val = 1  # "OK"
important_columns = [
    "s8_sensor100_millimeter_step1",
    "shift",
    "weekday",
    "s4_sensor16_minuten (zeit)_step1",
    "s5_sensor0_sekunden (zeit)_step1",
    "s10_sensor2_gramm_step1",
    "s3_sensor0_km_step1",
    "s7_sensor26_mikroohm_step1",
    "s8_sensor32_millimeter_step1",
    "s10_sensor0_minuten (zeit)_step1",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
]



class SmartNormalizer:
    def __init__(self, two_col=False):
        self.quantile_transformer = None
        self.encoder = None
        self.two_col = two_col
        self.is_numeric = None

    def fit(self, data):
        assert type(data) == np.ndarray
        data = data.copy().ravel()
        self.is_numeric = data.dtype != np.dtype("O")
        if self.is_numeric:
            good_mask = np.isfinite(data)
            data = data[good_mask]
            if not len(data):
                data = np.zeros(1)
            data = data.reshape(-1, 1)
            self.quantile_transformer = QuantileTransformer(
                output_distribution="normal", n_quantiles=min(100, len(data))
            )
            self.quantile_transformer.fit(data)
        else:
            data = data.reshape(-1, 1)
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.encoder.fit(data)

    def transform(self, data):
        data = data.copy().ravel()
        if self.is_numeric:
            good_mask = np.isfinite(data)
            data[~good_mask] = 0
            data = data.reshape(-1, 1)
            first_col = self.quantile_transformer.transform(data)
        else:
            good_mask = data != None
            data = data.reshape(-1, 1)
            first_col = self.encoder.transform(data)
        if self.two_col:
            second_col = good_mask.astype(np.float32).reshape(-1, 1)
            return np.concatenate([first_col, second_col], axis=1)
        else:
            return first_col.reshape(-1, 1)


class SmartNormalizerDF:
    def __init__(self, two_col=False):
        self.two_col = two_col
        self.normalizers = {}

    def fit(self, df):
        for col in df.columns:
            self.normalizers[col] = SmartNormalizer(self.two_col)
            self.normalizers[col].fit(df[col].values)

    def transform(self, df):
        df = df.copy()
        for i, col in enumerate(df.columns):
            if self.two_col:
                df[[col, col + "_ok"]] = self.normalizers[col].transform(df[col].values)
            else:
                df[col] = self.normalizers[col].transform(df[col].values)
            if i % 10 == 0:
                df = df.copy()
        return df


def add_weather_data(df, weather_data_path = "weather.csv"):
    weather_df = pd.read_csv(weather_data_path)
    weather_df = weather_df.rename(columns={"date": "message_timestamp"})
    # round message_timestamp to nearest hour, remove timezone from both
    weather_df["message_timestamp"] = weather_df["message_timestamp"].apply(
        lambda x: pd.Timestamp(x).tz_localize(None).round("h")
    )
    df["message_timestamp"] = df["message_timestamp"].apply(
        lambda x: pd.Timestamp(x).tz_localize(None).round("h")
    )
    df = df.merge(weather_df, on="message_timestamp", how="inner")
    return df
