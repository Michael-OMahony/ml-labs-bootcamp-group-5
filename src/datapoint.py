import pandas as pd


SENSOR_READING_COLUMNS = {
    'Bluetooth': ['Time', 'Sensor', 'Rssi'],
    'Gyroscope': ['Time', 'Sensor', 'X', 'Y', 'Z'],
    'Attitude':  ['Time', 'Sensor', 'X', 'Y', 'Z'],
    'Accelerometer': ['Time', 'Sensor', 'X', 'Y', 'Z'],
    'Altitude': ['Time', 'Sensor', 'X', 'Y'],
    'Magnetic-field': ['Time', 'Sensor', 'X', 'Y', 'Z', 'Accuracy'],
    'Gravity': ['Time', 'Sensor', 'X', 'Y', 'Z'],
    'Heading': ['Time', 'Sensor', 'XTrue', 'YTrue', 'ZTrue', 'XMagnetic',
                'YMagnetic',  'ZMagnetic', ],
}


def read_csv(filepath, sensors=[]):
    lines = open(filepath).read().split('\n')[7:]
    if len(sensors) == 0:
        sensors = list(SENSOR_READING_COLUMNS.keys())
    data = {}
    # for sensor in sensors:
    for line in lines:
        if line == '':
            continue
        readings = line.split(',')
        assert len(readings) >= 2, readings
        sensor = readings[1]
        assert sensor in SENSOR_READING_COLUMNS, sensor
        assert len(readings) == len(
            SENSOR_READING_COLUMNS[sensor]), (readings, SENSOR_READING_COLUMNS[sensor])
        if sensor not in data:
            data[sensor] = []
        data[sensor].append(readings)

    for sensor, columns in SENSOR_READING_COLUMNS.items():
        _dtype_dict = {
            c: float if c not in ['Accuracy', 'Sensor'] else 'object'
            for c in columns

        }
        data[sensor] = pd.DataFrame(data[sensor], columns=columns)
        data[sensor] = data[sensor].astype(_dtype_dict)

    return data


class DataPoint:

    def __init__(self, filepath, _type="train"):
        self.filepath = filepath
        self.data_dict = read_csv(filepath)

    def _get_sensor_data(self, sensor):
        return self.data_dict[sensor]

    def at(self, t, reset_index=False, sensors=[]):
        if len(sensors) == 0:
            data_dict = self.data_dict
        else:
            for sensor in sensors:
                assert sensor in self.data_dict, f'{sensor} not in data_dict'
            data_dict = {
                sensor: self.data_dict[sensor] for sensor in sensors}
        data_at_t = {}
        for sensor, df in data_dict.items():
            rows = df[df.Time == t]
            if len(rows) > 0:
                if reset_index:
                    rows = rows.reset_index(drop=True)
                data_at_t[sensor] = rows
        if len(sensors) == 1 and sensors[0] in data_at_t:
            return data_at_t[sensors[0]]
        return data_at_t

    def bw(self, t1, t2, include_extremes=False, reset_index=False):
        data_bw = {}
        for sensor, df in self.data_dict.items():
            if include_extremes:
                rows = df[(df.Time >= t1) & (df.Time <= t2)]
            else:
                rows = df[(df.Time > t1) & (df.Time < t2)]
            if len(rows) > 0:
                if reset_index:
                    rows = rows.reset_index(drop=True)
                data_bw[sensor] = rows
        return data_bw

    def after(self, t, include_extremes=False, reset_index=False):
        data_after = {}
        for sensor, df in self.data_dict.items():
            if include_extremes:
                rows = df[df.Time >= t]
            else:
                rows = df[df.Time > t]
            if len(rows) > 0:
                if reset_index:
                    rows = rows.reset_index(drop=True)
                data_after[sensor] = rows
        return data_after

    def before(self, t, include_extremes=False, reset_index=False):
        data_before = {}
        for sensor, df in self.data_dict.items():
            if include_extremes:
                rows = df[df.Time <= t]
            else:
                rows = df[df.Time < t]
            if len(rows) > 0:
                if reset_index:
                    rows = rows.reset_index(drop=True)
                data_before[sensor] = rows
        return data_before

    @ property
    def bluetooth(self):
        return self._get_sensor_data('Bluetooth')

    @ property
    def gyroscope(self):
        return self._get_sensor_data('Gyroscope')

    @ property
    def accelerometer(self):
        return self._get_sensor_data('Accelerometer')

    @ property
    def gravity(self):
        return self._get_sensor_data('Gravity')

    @ property
    def attitude(self):
        return self._get_sensor_data('Attitude')

    @ property
    def altitude(self):
        return self._get_sensor_data('Altitude')

    @ property
    def magnetic_field(self):
        return self._get_sensor_data('Magnetic-field')

    @ property
    def heading(self):
        return self._get_sensor_data('Heading')
