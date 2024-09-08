Due to the large size of the [HEMEW<sup>S</sup>-3D database](https://doi.org/10.57745/LAI6YU), it cannot be entirely loaded on CPUs or GPUs. Therefore, the preprocessing step consists in writing individual files `sample_i.h5` that contain the material `a`, the source position `s`, the source angle `angle`, the equivalent moment tensor `moment`, and the three components of the velocity fields `uE`, `uN`, and `uZ`.

To reduce the computational load of machine learning applications, velocity fields are downsampled from 100 Hz to 50 Hz, and restricted to the time interval [0; 6.4s] (leading to 320 time steps).

To create the inputs, 
0. create a folder `data/raw` and a folder `data/formatted`
1. download the data from the [HEMEW<sup>S</sup>-3D repository](https://doi.org/10.57745/LAI6YU) and place the downloaded data inside the folder `data/raw/`
2. run `python3 create_data_geologies.py @Ntrain 27000 @Nval 3000 @Ntest 1000` (takes around 2 minutes)
3. run `python3 create_data_sources.py @Ntrain 27000 @Nval 3000 @Ntest 1000` (takes around 2 minutes)
4. run `python3 create_data_velocityfields.py @Ntrain 27000 @Nval 3000 @Ntest 1000` (takes around 1 hour)

Outputs of these codes are saved in `data/formatted/`.

To train the ML models, the mean and standard deviation of materials are also needed (to normalize inputs). They are created inside the function `create_data_geologies.py`.

Note: Samples of the HEMEW<sup>S</sup>-3D database are numbered 100000-129999 to avoid confusion with the HEMEW-3D database (with a fixed source), whose samples are numbered 0-29999. For ML applications, all samples are numbered 0-29999.
