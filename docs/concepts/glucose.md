## How Glyco represents glucose data

To represent glucose data, Glyco uses a format that is as independent as possible from: the measurement device, the unit of measurement, the timestamp format used, the type of storage etc. 
This will be based on top of pandas dataframes, since those are a goto tool for all forms of data processing and analysis.

We will call this a **universal glucose dataframe**.

This is a pandas Dataframe that is generated either from: another pandas Dataframe or a glucose CSV file (such as Freestyle Libre data, or Dexcom data). The file or dataframe MUST contain the required columns:
* **A glucose column** see [the possible units here](./units.md).
* **A timestamp column** with both time and date.

From the above, Glyco generates the universal glucose dataframe

## The Glucose Dataframe columns
Here is a list of all the columns contained in a universal glucose dataframe:
* **Index column** timestamp: datetime.Time **REQUIRED**
* timestamp (other than the index): datetime.Time **REQUIRED**
* original glucose in original unit (see [the implemented units here](./units.md)): G_orig. original format **REQUIRED**
* corrected glucose in mmol/L: G. float **REQUIRED**
* **Time related** these are derived from the timestamp:
  * `date`: datetime.date  e.g.: `2019-12-31`
  * `date_str`: string of the date, e.g: `05-12-2019 (Thursday)`
  * `hour`: hour of measurement e.g.: `1`
  * `weekday_number`: number of the day of the week 0 for Monday, 
  * `weekday_name`: three letters of the name of the day of the week, the first letter is a capital letter. E.g.: `Thu`.
  * `is_weekend`: boolean, `True` if the day is a weekend day, `False` otherwise.
  * *Shifted time*: Shifted time is defined to be used to make plotting and data analysis easier. This is simply the timestamp shifted by a pre-defined amount of hours. For example this can be used for plotting the nightime glucose more easily. For more information read the [logic documentation](logic.md). The fields generated are similar to the time fields with an added *"shifted_"* prefix: `shifted_t`, `shifted_date`,  `shifted_date_str`, `shifted_hour`, `shifted_weekday_number`, `shifted_weekday_name`,  `shifted_is_weekend`.
* Difference in time dt :
* Difference in glucose dg:
* First order derivative of glucose dg_dt.
* Area under the curve:
  * `auc`
  * `auc_min`
  * `auc_lim`
# Device specific properties
### Freestyle Libre data 
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

