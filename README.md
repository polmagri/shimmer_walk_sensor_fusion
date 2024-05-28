# shimmer_walk_sensor_fusion

This GitHub repository presents Python codes for obtaining the speed and displacement of a person walking, after two calibrated Shimmer sensors are applied to their foot.

The Shimmers are placed on the upper part of the foot and calibrated as follows:
- Accelerometer with a range of 16 g
- Gyroscope with maximum range

The inputs are CSV files (see example):
- Time in ms in UNIX
- Accelerometer in m/(s*s)
- Gyroscope in deg/sec

The first script (obtain-velocity.csv) uses the AHRS Algorithm for sensor fusion.
You can see all the optimization parameters in detail from this link.
It returns a .csv file to be saved in the same repository containing the time column and the speed column.
Select the start and end time (in seconds) of the data you want to analyze.
Perform this procedure twice in order to save the right foot speed and left foot speed. (remember to change the save name to avoid overwriting)

The second script (analysis_velocity) imports the two previously saved speed files, performs a plot, and synchronizes the data (select the desired time).
Then, it performs an interpolation of the speeds to transition from discrete data to continuous time and uses various methods and filters to merge them together:
- Moving average
- Savitzky-Golay filter
