# Time series classification for elevator failure prediction

In this Time Series Project I predict the failure of elevators using IoT sensor data, by treating it as a time series classification machine learning problem.

## Business Objective

Transportation is essential for individuals traveling from one location to another in this era of advanced technologies. The one significant vehicle that is created in virtually every tall building, the elevator, may also be seen everywhere. An elevator is a mode of transportation that moves cargo and people between building floors. As a result, it saves one's time and energy rather than requiring one to climb steps. The elevator, however, is still mechanical and can eventually fail. An elevator failure during transit would result in numerous inconveniences and possibly even disastrous accidents. Therefore, it is essential to do elevator maintenance to avoid the potential that it would fail and cause an accident.

Without maintenance, the equipment will eventually fail and potentially cause casualties. Preventive maintenance is a form of maintenance approach that is based on engineers' prior experience, similar machine data, and machine understanding. With the help of this data, one can choose a time frame for the machine to examine, such as daily, weekly, monthly, or annually. Preventive maintenance is a common practice for elevators. With this approach, scheduling time for service and maintenance results in work being done that is unneeded if the elevator is in good condition, and some parts are changed even though they can still support and function in the machine for a considerable amount of time.

Therefore, even when the elevator is in good condition, it results in the needless expenditure of money on services and replacement parts. With the installation of the predictive maintenance system, the system will notify the user if the elevator is about to break down, preventing the need for unforeseen or unscheduled maintenance. To record the elevator's current condition, real-time data will be gathered from it utilizing sensors.

In this project, we shall be analyzing the data captured each minute by IoT sensors for a month and developing a model that can predict if the elevator is broken or about to fail.

We shall be making use of the Time Series Classification techniques which can be said to be a combination of Classification modeling and Time Series Forecasting for this project.

## Data Description

The data is captured each minute by IoT sensors from 1/1/2020 to 31/1/2020. There is a total of 44640 data points with 11 features, and Status is the target variable.

## Sections

1. Initial steps:
   - Load data
   - general overview
   - check distribution
   - train/test split
   - select window size & lag
2. Exploratory data analysis:
   - Univariate EDA visualizations (density plots, time series plots)
   - correlation and dropping correlated features
   - prepare time-based new features (e.g. day of the week)
3. Signal processing:
   - Outlier removal
   - fill missing values
   - frequency filtering to remove noise
4. Classification:
   - Traditional ML using summary stats
   - 1-dimensional time series classifier framework
   - multi-dimensional time series classifier framework
   - evaluation of model results
   - check impact of window size & lag
