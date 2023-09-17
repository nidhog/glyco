## How Glyco represents glucose data
Glyco represents glucose data as a pandas DataFrame.

From reading any glucose data Glyco generates a format that is as independent as possible from: the measurement device, the unit of measurement, the timestamp format used, the type of storage etc. 
Glyco does all the preprocessing and cleaning necessary to generate a glucose dataframe ready for analysis.

We will call this the **Generated Glucose Dataframe**, **Universal Glucose Dataframe** or for simplification we will sometimes refer to it as the **Glucose Dataframe**.

This is a pandas Dataframe that is generated either from: another pandas Dataframe or a glucose CSV file (such as Freestyle Libre data, or Dexcom data). The file or dataframe MUST contain the required columns:
* **A glucose column** see [the possible units here](./units.md).
* **A timestamp column** with both time and date.

For example a dataframe:

| Timestamp column | Glucose column | ... |
|-|-|-|
| "2023-01-15 16:25" | 6.2 |... | 
| "2023-01-15 16:40" | 6.4 |... | 
From the above, Glyco generates the glucose dataframe

| Timestamp column | Glucose column |... | Date | Hour | Glucose Derivative (dG/dT) | Glucose AUC  |...|
|-|-|-|-|-|- |-|-|
| 2023-01-15 16:25 | 6.2|... |2023-01-15 | 16 | ..| .. |...|
| 2023-01-15 16:40 | 6.4|... |2023-01-15 | 16| .. | .. | Derivative |...|

This resulting glucose dataframe will contain both the original columns and some new columns we will call the **Generated Glucose Properties**

## The Generated Glucose Dataframe
Here is a list of all the columns contained in a universal glucose dataframe:
* **The original columns** including the original glucose column and the original timestamp column.
* **Index column** by timestamp: *datetime.datetime*
* `tsp` timestamp (same as index), generated from the timestamp in the original data based on a given time format: *datetime.datetime*
*  . original format 
* `glucose` corrected glucose in mmol/L (see [the implemented units here](./units.md)) using interpolation and averaging to avoid outliers, errors and missing data. *float*
* **Time related** these are all derived from the timestamp:
  * `date`: It holds the date component. *datetime.date*  e.g.: `2019-12-31`
  * `date_str`: string representation of the date. *str*, e.g: `05-12-2019 (Thursday)`
  * `hour`: hour of measurement. *int*, e.g.: `1`
  * `weekday_number`: number of the day of the week 0 for Monday. *int* 
  * `weekday_name`: three letters of the name of the day of the week, the first letter is a capital letter. *str*, e.g.: `Thu`.
  * `is_weekend`: boolean, `True` if the day is a weekend day, `False` otherwise. *bool*
  * *Shifted time*: Shifted time is defined to be used to make plotting and data analysis easier. This is simply the timestamp shifted by a pre-defined amount of hours. For example this is used in glyco for plotting the nightime glucose more easily along with the daytime glucose. The fields generated are similar to the time fields with an added *"shifted_"* prefix: `shifted_t`, `shifted_date`,  `shifted_date_str`, `shifted_hour`, `shifted_weekday_number`, `shifted_weekday_name`,  `shifted_is_weekend`.
* `dt` Time interval between consecutive glucose measurements. `timedelta` 
* `dg` Rate of change of glucose (delta glucose) calculated from the glucose values. `float`
* `dg_dt` First order derivative of glucose. The rate of change of glucose divided by the time interval, representing glucose velocity. `float`
* Area under the curve:
  * `auc` The area under the curve (AUC) above the mean glucose. *float*
  * `auc_min` The area under the curve (AUC) above the smallest glucose value. *float*
  * `auc_lim` Area under the curve (AUC) calculated with a predefined glucose limit, given as input when reading the glucose data. *float*
## How to read glucose data and generate a glucose dataframe

When you read glucose data you generate a Generated Glucose Dataframe containing your data along with other useful properties. You can either read glucose data from: 
* a pandas Dataframe containing glucose data
* a glucose CSV file (such as Freestyle Libre data, or Dexcom data). The file or dataframe MUST contain the required columns: a **glucose column** (see [the possible units here](./units.md)) and a **timestamp column** with both time and date.
### Read a CSV file
You can read glucose data from a CSV file using the `read_csv` Function

The `read_csv` function in the glyco library reads a CSV file containing glucose data and generates the Generated Glucose DataFrame. You can specify the following parameters as input:

- `file_path`: Path to the glucose CSV file.
- `timestamp_col`: Name of the timestamp column in the CSV file.
- `timestamp_fmt`: The original format of the timestamp values in the timestamp column.
- `glucose_col`: Name of the glucose column in the CSV file.
- `calculate_glucose_properties`: Set to `True` to calculate and add the Generated Glucose Properties to the resulting DataFrame.
- `glucose_lim`: A lower glucose threshold used for some of the Generated Glucose Properties.
- `unit_autodetect`: If `True`, the glucose unit will be automatically inferred from the values in the CSV file.
More advanced inputs can be found in the `read_csv` function documentation.
### Read a pandas DataFrame
Alternatively, you can generate the Glucose DataFrame by directly passing a pre-existing pandas DataFrame that contains glucose data to the `read_df` function within glyco. 

- `df`: the pandas dataframe with glucose data.
- `timestamp_col`: Name of the timestamp column in the CSV file.
- `timestamp_fmt`: The original format of the timestamp values in the timestamp column.
- `glucose_col`: Name of the glucose column in the CSV file.
- `calculate_glucose_properties`: Set to `True` to calculate and add the Generated Glucose Properties to the resulting DataFrame.
- `glucose_lim`: A lower glucose threshold used for some of the Generated Glucose Properties.
- `unit_autodetect`: If `True`, the glucose unit will be automatically inferred from the values in the CSV file.
More advanced inputs, such as filtering rows, can be found in the `read_df` function documentation.

# Data from Specific Devices
Glyco makes it easy to read data from specific glucose monitoring devices such as FreeStyle Libre and Dexcom (to be improved in the next release). Here are some examples of what this data may look like:
## Freestyle Libre data 
[You can find a link to a sample of Freestyle Libre data here](../test/data/sample_glucose.csv)

The FreeStyle Libre sensor provides this data in the form of a csv file. 

The important fields in these files are:
* `Device`: same in all the dataset `FreeStyle LibreLink`
* `Serial Number`: same for all rows `b59d4499-1a07-462b-b7da-a179f2093996`
* `Device Timestamp`: time of the glucose measurement based on the device (phone).
* `Historic Glucose mmol/L`: glucose values in mmol/L.
* The above field can be different, in certain cases it is given in the `mg/dL` unit.
* `Record Type` this field is important, the main values to consider are:
  * 0: Glucose scans.
  * 1: Notes. Exercise/Meals. We can use these to generate events.
  * 6: Food carbs.
* `Notes` notes that the user enters manually, if any.

There are other fields in the data that are not relevant in this case.

This data is used by glyco to generate a Glucose Dataframe that corrects some of the errors due to the sensors or data collection.
