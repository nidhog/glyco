## The Unified glucose dataframe
This is generated either from a pandas Dataframe or Freestyle Libre data, Dexcom data or any other data that contains the following required columns:
* A glucose column (see [the possible units here](./units.md)).
* A timestamp column (with both time and date).

From the above, glyco generates the Unified glucose dataframe with the following columns:
* **Index column** timestamp: datetime.Time **REQUIRED**
* timestamp (other than the index): datetime.Time **REQUIRED**
* original glucose in original unit (see [the implemented units here](./units.md)): G_orig. original format **REQUIRED**
* corrected glucose in mmol/L: G. float **REQUIRED**
* Time related:
  * date, hour, week, year
  * shifted date, hour, week, year
* Difference in time dt :
* Difference in glucose dg:
* First order derivative of glucose dg_dt.
* Area under the curve:
  * auc
  * auc_min
  * auc_lim
* 



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

