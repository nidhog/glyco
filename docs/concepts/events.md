## The Unified events dataframe
This is generated from either Freestyle Libre data, Dexcom data or any other data that contains:
* A glucose column.
* A timestamp column (with both time and date).



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
