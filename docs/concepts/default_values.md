# Default values used in Glyco
Most of the default values used in glyco are based on FreeStyle Libre continuous glucose data. These however can always be changed to match other sources of glucose data.

## Units and Devices
Glyco primarily supports mmol/L as the default unit for glucose measurement, but it also facilitates conversion from mg/dL and g/L, which are common units used in various devices. 

Glyco includes an autodetection feature for glucose units, which can infer the unit from data values if not specified.

### Automatic Unit Detection
The automatic unit detection in Glyco is designed to simplify the data import process by identifying the glucose measurement units from the dataset without user intervention. This feature examines a sample of glucose values and uses statistical methods to determine the most likely unit:

- **mmol/L**: Typically used outside the United States and has values generally between 0.5 and 33.0.
- **mg/dL**: Commonly used in the United States with values typically between 10 and 600.
- **g/L**: Less common but used in some medical contexts, with values significantly lower than those of mg/dL.

The detection algorithm calculates the average of a sample; if the average is above 33, the unit is likely mg/dL. If the average is below 6 but higher than typical g/L measurements, it is inferred as mmol/L. The standard deviation is also considered to improve accuracy, especially distinguishing between mg/dL and g/L in borderline cases.

### Supported Devices
Glyco is built to support data regardless of which device it came from. However, it also offers some extra features for specific devices.
For the moment, implemented devices in Glyco are defined under the `Devices` enum, with `abbott` (representing FreeStyle Libre) currently implemented. Users can extend this to other devices as needed, but additional manual adjustments might be required for devices not predefined in Glyco.

## Naming of glucose dataframe columns
The naming conventions for columns in the glucose dataframe are standardized to ensure consistency across different datasets and analyses. Below are the default names used:

- **Timestamp Column**: `'tsp'` (Generated from the original timestamp column)
- **Glucose Column**: `'glucose'` (Converted and standardized glucose measurements)
- **Date Column**: `'date'`
- **Hour Column**: `'hour'`
- **Weekday Number Column**: `'weekday_number'`
- **Weekday Name Column**: `'weekday_name'`
- **Weekend Indicator Column**: `'is_weekend'`

Additionally, for data manipulation and feature generation, other columns like `'dt'` (time interval), `'dg'` (change in glucose), and `'dg_dt'` (rate of change of glucose) are generated. For more you can read about [the columns in the generated glucose dataframe here](./glucose.md).

## Naming of event/meal sessions dataframe columns
For dataframes that handle events or meals, which can affect glucose readings, Glyco uses the following column naming conventions:

- **Timestamp Column**: `'tsp'` (Aligns with the glucose dataframe's timestamp for consistency)
- **Meal Reference Column**: `'Reference'` (Used to link meal data with glucose readings)
- **Notes Column**: `'Notes'` (For additional user-entered information regarding the meal or event)

These names are designed to facilitate easy merging and analysis of glucose and event data, helping researchers and users to correlate dietary habits with glucose fluctuations effectively.