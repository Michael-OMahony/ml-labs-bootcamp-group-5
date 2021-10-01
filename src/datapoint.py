import math

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

plt.style.use("ggplot")


SENSOR_READING_COLUMNS = {
    "Bluetooth": ["Time", "Sensor", "Rssi"],
    "Gyroscope": ["Time", "Sensor", "X", "Y", "Z"],
    "Attitude": ["Time", "Sensor", "X", "Y", "Z"],
    "Accelerometer": ["Time", "Sensor", "X", "Y", "Z"],
    "Altitude": ["Time", "Sensor", "X", "Y"],
    "Magnetic-field": ["Time", "Sensor", "X", "Y", "Z", "Accuracy"],
    "Gravity": ["Time", "Sensor", "X", "Y", "Z"],
    "Heading": [
        "Time",
        "Sensor",
        "XTrue",
        "YTrue",
        "ZTrue",
        "XMagnetic",
        "YMagnetic",
        "ZMagnetic",
    ],
}


def read_csv(filepath, sensors=[]):
    filedump = open(filepath).read()
    lines = filedump.split("\n")[7:]
    if len(sensors) == 0:
        sensors = list(SENSOR_READING_COLUMNS.keys())
    data = {}
    # for sensor in sensors:
    for line in lines:
        if line == "" or "Activity" in line:
            continue
        readings = line.split(",")
        assert len(readings) >= 2, readings
        sensor = readings[1]
        assert sensor in SENSOR_READING_COLUMNS, sensor
        assert len(readings) == len(SENSOR_READING_COLUMNS[sensor]), (
            readings,
            SENSOR_READING_COLUMNS[sensor],
        )
        if sensor not in data:
            data[sensor] = []
        data[sensor].append(readings)

    for sensor, columns in SENSOR_READING_COLUMNS.items():
        if sensor not in data:
            continue
        _dtype_dict = {
            c: float if c not in ["Accuracy", "Sensor"] else "object" for c in columns
        }
        data[sensor] = pd.DataFrame(data[sensor], columns=columns)
        data[sensor] = data[sensor].astype(_dtype_dict)

    nsdata = read_non_sensor_data(filedump)
    return data, nsdata


def read_non_sensor_data(filedump):
    lines = filedump.split("\n")[:7]
    nsdata = {}
    for line in lines:
        key, value = line.split(",")
        nsdata[key] = value

    return nsdata


class DataPoint:
    def __init__(self, filepath, _type="train"):
        self.filepath = filepath
        self.data_dict, self.nsdata_dict = read_csv(filepath)

    def _get_sensor_data(self, sensor):
        return self.data_dict[sensor]

    def chirp_mean_sequence(self):
        bluetooth = self.bluetooth
        chirps, rssi = [], []
        for idx, t in bluetooth.iterrows():
            if idx > 0 and t.Time - bluetooth.Time.loc[idx - 1] > 1:
                chirps.append(np.array(rssi).mean())
                rssi = []
            else:
                rssi.append(-t.Rssi)
        return chirps + [np.array(rssi).mean()]

    def get_readings_by_time(
        self,
        t1,
        t2=None,
        position="at",
        sensors=[],
        reset_index=False,
        include_extremes=False,
    ):
        if len(sensors) == 0:
            data_dict = self.data_dict
        else:
            for sensor in sensors:
                assert sensor in self.data_dict, f"{sensor} not in data_dict"
            data_dict = {sensor: self.data_dict[sensor] for sensor in sensors}
        data = {}
        for sensor, df in data_dict.items():
            if position == "at":
                rows = df[df.Time == t1]
            elif position == "bw":
                assert not isinstance(t2, type(None))
                if include_extremes:
                    rows = df[(df.Time >= t1) & (df.Time <= t2)]
                else:
                    rows = df[(df.Time > t1) & (df.Time < t2)]
            elif position == "after":
                if include_extremes:
                    rows = df[df.Time >= t1]
                else:
                    rows = df[df.Time > t1]
            elif position == "before":
                if include_extremes:
                    rows = df[df.Time <= t1]
                else:
                    rows = df[df.Time < t1]
            if len(rows) > 0:
                if reset_index:
                    rows = rows.reset_index(drop=True)
                data[sensor] = rows
        if len(sensors) == 1 and sensors[0] in data:
            return data[sensors[0]]
        if len(data) == 0:
            return None
        return data

    def at(self, t, reset_index=False, sensors=[]):
        return self.get_readings_by_time(
            t, position="at", sensors=sensors, reset_index=reset_index
        )

    def bw(self, t1, t2, include_extremes=False, reset_index=False, sensors=[]):
        return self.get_readings_by_time(
            t1, t2, position="bw", sensors=sensors, reset_index=reset_index
        )

    def after(self, t, include_extremes=False, reset_index=False, sensors=[]):
        return self.get_readings_by_time(
            t, position="after", sensors=sensors, reset_index=reset_index
        )

    def before(self, t, include_extremes=False, reset_index=False, sensors=[]):
        return self.get_readings_by_time(
            t, position="before", sensors=sensors, reset_index=reset_index
        )

    def _plot(
        self, sensor="Bluetooth", df=None, t1=None, t2=None, include_extremes=False
    ):
        if t1 and t2:
            df = self.bw(t1, t2, include_extremes=include_extremes, sensors=[sensor])
        else:
            if isinstance(df, type(None)):
                df = self._get_sensor_data(sensor)

        self._plot_df(df)

    def _plot_df(self, df):
        readings = [
            col for col in df.columns if col not in ["Accuracy", "Sensor", "Time"]
        ]
        nrows = int(math.ceil(len(readings) / 2))
        fig, ax = plt.subplots(
            nrows=nrows, ncols=2, sharex=False, sharey=False, figsize=(16, 6 * nrows)
        )
        ax = ax.flatten()
        if len(ax) > len(readings):
            fig.delaxes(ax[-1])
        title = df.Sensor.sample().item()
        if len(readings) > 1:
            fig.suptitle(title)
        else:
            ax[0].set_title(title)
        for _ax, reading in zip(ax, readings):
            _ax.scatter(df.Time, df[reading])
            _ax.set_xlabel("Time")
            _ax.set_ylabel(reading)
        plt.show()

    @property
    def bluetooth(self):
        return self._get_sensor_data("Bluetooth")

    @property
    def gyroscope(self):
        return self._get_sensor_data("Gyroscope")

    @property
    def accelerometer(self):
        return self._get_sensor_data("Accelerometer")

    @property
    def gravity(self):
        return self._get_sensor_data("Gravity")

    @property
    def attitude(self):
        return self._get_sensor_data("Attitude")

    @property
    def altitude(self):
        return self._get_sensor_data("Altitude")

    @property
    def magnetic_field(self):
        return self._get_sensor_data("Magnetic-field")

    @property
    def heading(self):
        return self._get_sensor_data("Heading")

    @property
    def TXDevice(self):
        return self.nsdata_dict["TXDevice"]

    @property
    def RXDevice(self):
        return self.nsdata_dict["RXDevice"]

    @property
    def TXPower(self):
        return self.nsdata_dict["TXPower"]

    @property
    def TXCarry(self):
        return self.nsdata_dict["TXCarry"]

    @property
    def RXCarry(self):
        return self.nsdata_dict["RXCarry"]

    @property
    def RXPose(self):
        return self.nsdata_dict["RXPose"]

    @property
    def TXPose(self):
        return self.nsdata_dict["TXPose"]


if __name__ == "__main__":
    DataPoint("data/tc4tl_training_data_v1/tc4tl/data/train/bnqefvro_tc4tl20.csv")
