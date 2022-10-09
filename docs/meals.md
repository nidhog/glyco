## The unified Meals dataframe
This is generated from either:
* A dataframe with meal times that contains: 
  * *Required* A timestamp column (datetime.Timestamp) (with both time and date)
  * *Optional* Notes column (str).
  * *Optional* Reference column (str) refers to a file or something that describes the meal in more details.
* A folder with meal pictures.

To generate the unified meals dataframe, glyco also requires: [**a unified Glucose dataframe**](glucose.md), if a folder is used a time delta must be provided in hours to correct the timezone if there is a timezone difference (This time delta is substracted: `timedelta = glucose device timezone - meal timezone`).

As a result `glyco` will generate a **Meals dataframe**, with:

* _original timestamp_
* _nearest glucose timestamp_
* Meal id
* Meal session
  * _estimated meal start timestamp_
  * _estimated meal end timestamp_
  * Meal session id
* *Optional*:
  * References if provided in the input (if a folder is used)
  * Notes if provided in the input
  * timezone difference
* Glucose stats:
  * Glucose max
  * Glucose min
  * Glucose average
  * Glucose standard deviation
  * Glucose 68th percentile
  * Glucose 95th percentile
  * Glucose 5th percentile
* Meal features:
  * Area Under the Curve
    * Above minimum
    * Above threshold
    * Above mean
  * Ingestion speed
  * Clearance speed

