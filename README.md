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
* **Data visualization:** In order to aid our understanding of the neutrino detection process and extract useful features for regression, we created a data visualization module in [event_plotting.ipynb](event_plotting.ipynb).

### Approaches

1. **Best-Fit Line using PCA:** In [Dataloader-SpaceBestFit.ipynb](Dataloader-SpaceBestFit.ipynb), we used PCA to get the best fit line interpolating the active sensors as an initial estimate of the neutrino's direction.
2. **Sensor-Based Linear Regression:** In [linear_regression_tensorflow.ipynb](linear_regression_tensorflow.ipynb), we make a custom loss funtion for tensor flow, then do a linear regression on our features for various optimizers (mostly Adam, after preliminary testing) in tensor flow models.
3. **Linear Regression With Extracted Features:** We extracted various features from the raw event data. In [linear-regression-testing.ipynb](linear-regression-testing.ipynb), we investigate which of our extracted features may be useful in an attempt to find a model. Unfortunately, linear regression did not appear to help.
4. **Physics-informed best-fit line:** In [Dataloader-TimeBestFit.ipynb](Dataloader-TimeBestFit.ipynb), we use a formula found in a physics paper to compute the best fit line of the points, taking into account time. The source is 'Direction Reconstruction of IceCube Neutrino Events using Millipede" by Alexander Wallace.
5. **Fully connected neural network:** In [](), we construct a primitve fully connected neural network to attempt to improve on the regression analysis. Linear regression doesn’t capture complex dependencies between pulses at different sensors, so some form of neural network should better capture these complexities. Specifically, we construct an net with 8 hidden layers, utilizing the Adam optimizer and tanh activation functions. Ultimately, this proved unsuccessful do to how much information had to be discarded to produce an efficient model. 

7. **Convolutional neural network:** In[](), we constuct a more sophisticated neural net. We chose a three dimensional network with four channels as it seemed to allow us to encode the spatial position of the detectors more effectively than our previous approach. Additionally, we felt that our problem was sufficiently similar to image processing that a convolutional network seemed appropriate. After the convolutional step, we reintroduced three additional features derived from the computations of the regression group: a regression based initial guess of both azimuth and position, along with an estimate of the number of spatial clusters. 
