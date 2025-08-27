# Report

## Steps followed

### 1. Set up a project
- Created a requirements.txt file with necessary dependencies.
- Initialized a Git repository and created a `.gitignore` file to exclude unnecessary files.
- Set up a Python virtualenv.

### 2. Explore the data
- Loaded each dataset
- Computed summary statistics using pandas.describe(): mean, std, min, max, quartiles
- Used seaborn to plot a boxplot for each feature
- PV generation dataset contained negative kW values, as well as outliers
- Wind speed dataset contains a significant number of outliers, which were kept. They might represent valid readings, rather than noise.
- Distribution of air temperature, PV generation and global horizontal irradiance over the day matches expected patterns
- Similarly, distributions over the year also match seasonal patterns
- **Selection criteria**:
  - **Features**: weather measurements, PV generation, most recent irradiation forecast available before prediction time, time-based features (hour and day of year as sin and cos signals)
  - **Target**: PV generation 2 hour ahead

### 3. Prepare the data
- Filter out negative PV values
- Discarded PV outliers outside of `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`
- Merged datasets on timestamp. Cleaned `NAN` values and interpolated adjacent values.
- Created dataset class with sequence indexing (sliding window) 
- Split dataset indices into training (80%), validation (10%) and test (10%) sets, created Dataset Subsets and separate `DataLoaders` for train/val/test, thus allowing random sequence sampling during training.
- Scaled all features (except time-based) around 0 mean and unit variance of the training set using `StandardScaler` from `sklearn.preprocessing`. 

### 4. Train a model
- Used an LSTM -> Fully Connected Layer + tanh activation model.
- Sample data shape: (batch_size, sequence_length, num_features)
- Used MSE loss and Adam optimizer
- Save best model and scalers based on validation loss
- Implemented early stopping based on validation loss
- Adaptive lr based on validation loss
- Plotting training and validation loss curves over epochs

### 5. Evaluate the training and the model
- Evaluated model on test set using RMSE, R2 score and MAE metrics
- Plotted predicted vs actual values for all days in the test set
- **Improvements**:
  - Hyperparameter sweep (learning rate, batch size, sequence length, hidden layer size, number of layers)
  - Data cleaning systems from literature
  - Randomize train/val/test split. Currently the split is done chronologically, which leads to overfitting to a certain year interval.
  - Check train/test distribution skews

## Usage
- Install dependencies: `pip install -r requirements.txt`
- Dataset analysis: `python scripts/datasets.py`
- Train model: `python scripts/train.py`. Command line args are available
- Evaluate model: `python scripts/test.py`. Command line args are available
