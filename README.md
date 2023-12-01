# Neutrino Direction Detection

This is the repository for a final project at the Erdos Data Science Bootcamp.

### Description
The goal of our project was to determine an effective and efficient method for computing the direction of incoming neutrinos, using data collected at the IceCube Neutrino detector and provided via the associated Kaggle competition. The direction consists of two angles: an azimuth and zenith. Our goal was to minimize the mean angular error (a value between 0 and pi) between the true direction and predicted direction. 

### Background
The difficulty in this problem comes from the fact that a neutrino is never itself detected. Rather, when the neutrino hits a nucleus in the IceCube detector, the collision produces other particles which themselves produce a burst of light. These bursts are called Cherenkov Radiation and travel perpendicular to a “Cherenkov cone”. 

### Data Overview
The data provided by Kaggle consisted of 660 parquet files, each consisting of approximately two hundred thousand individual neutrino events. Each event has multiple pulses (often thousands), and each pulse detected on exactly one of the 5160 sensors. There are two kinds of files:
* **Batch files (Input data):** Each row represents a pulse and contains 5 columns:
  * ```event_id (int)```: the event ID. Saved as the index column in parquet.
  * ```time (int)```: the time of the pulse in nanoseconds in the current event time window. The absolute time of a pulse has no relevance, and only the relative time with respect to other pulses within an event is of relevance.
  * ```sensor_id (int)```: the ID of which of the 5160 IceCube photomultiplier sensors recorded this pulse.
  * ```charge (float32)```: An estimate of the amount of light in the pulse, in units of photoelectrons (p.e.). A physical photon does not exactly result in a measurement of 1 p.e. but rather can take values spread around 1 p.e. As an example, a pulse with charge 2.7 p.e. could quite likely be the result of two or three photons hitting the photomultiplier tube around the same time. This data has float16 precision but is stored as float32 due to limitations of the version of pyarrow the data was prepared with.
  * ```auxiliary (bool)```: If True, the pulse was not fully digitized, is of lower quality, and was more likely to originate from noise. If False, then this pulse was contributed to the trigger decision and the pulse was fully digitized.
* **Meta file (Label data):** Each row represents a neutrino event and contains 6 columns:
  * ```batch_id (int)```: the ID of the batch the event was placed into.
  * ```event_id (int)```: the event ID.
  * ```[first/last]_pulse_index (int)```: index of the first/last row in the features dataframe belonging to this event.
  * ```[azimuth/zenith] (float32)```: the [azimuth/zenith] angle in radians of the neutrino. A value between 0 and 2*pi for the azimuth and 0 and pi for zenith. These are the target columns. The direction vector represented by zenith and azimuth points to where the neutrino came from.
* **Sensor geometry:** A file containing x,y,z coordinates of all sensors.

### Approaches

1. **Sensor-Based Linear Regression:** In [linear_regression_sensor_data.ipynb](linear_regression_sensor_data.ipynb), we perform several linear regressions which we call "sensor-based". This means that the features included make essential use of all 5,160 IceCube sensors. In essence, each data point in X_train fed into the LinearRegression object will be a 5160-tuple where the $i^{th}$ entry provides information about the (in)activation of the $i^{th}$ sensor for that event. This has complications detailed in the notebook, due to which existing packages could not be used directly.
2. 
