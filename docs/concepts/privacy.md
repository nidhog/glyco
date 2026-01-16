# Handling Private and Sensitive Information
Glyco is **designed to preserve privacy and de-identify glucose data analytics**.  

This document explains **how privacy works internally** and **how to use it correctly** when sharing, publishing, or collaborating on glucose data.

## Why privacy matters in glucose data
Continuous glucose monitoring (CGM) data is **highly identifying**:
- Timestamp patterns may reveal daily routines.
- Notes may contain personal information.
- Serial numbers uniquely identify devices.
- Glucose values and timestamps can be re-identified when combined with other datasets.

Glyco’s privacy features are designed to **reduce the risk of identification** while preserving:
- Relative glucose dynamics.
- The temporal structure.
- Analytical usefulness for statistics, plots, and sessions.

This is why we recommend using the privacy feature whenever you are planning to share the data.
You can always verify the results with the original data if you add noise to the glucose values.
## What _glyco_’s privacy layer does and how to use it

Privacy handling is implemented in:

```python
from glyco.privacy import mask_private_information
```
It can be enabled automatically when reading data using:
```
read_csv(..., mask_private_info=True)
read_df(..., mask_private_info=True)
```
Or called manually on an existing dataframe.

### Core privacy actions
Glyco can:
* Replace sensitive column values (e.g. notes) using hashing
* Remove sensitive columns entirely
* Shift timestamps to a new start date
* Add statistical noise to glucose values

Each action is optional and configurable.
### How to use _glyco_’s privacy layer
To de-identify and make glucose data more private you can use _glyco_'s `mask_private_information` function.
This function is used by default in `read_csv` and `read_df` if you enable the argument `mask_private_info=True`.


```python
from glyco.privacy import mask_private_information

masked_df, added_noise, replaced_values = mask_private_information(
    gdf=df,
    glucose_col="glucose",
    tsp_col="tsp",
    tsp_fmt="%d-%m-%Y %H:%M",
    set_start_date="01-01-2023 00:00",
    remove_columns=["Serial Number"],
    replace_columns=["Notes"],
)
```
* This replaces values in selected columns (e.g. notes) using a hash
* Removes selected identifying columns
* Shifts all timestamps so the data starts at a new date (time of day is preserved)
* Adds Gaussian noise to glucose values

Amd returns:
* the masked dataframe
* the noise added to glucose (for optional local reversal if needed)
* the original values of replaced columns (if any, for local reversal if needed)

**To customise the behaviour you can use `PrivateInfoKwargs`**
```
from glyco.glucose import PrivateInfoKwargs

privacy_cfg = PrivateInfoKwargs(
    set_start_date="01-01-2022 00:00",
    remove_columns=["Serial Number", "User ID"],
    replace_columns=["Notes", "Reference"],
    noise_std=0.3,
)

gdf = read_csv(
    "glucose.csv",
    mask_private_info=True,
    private_info_kwargs=privacy_cfg,
)
```
This lets you decide on:
* Which columns ro remove or anonymise (with a hash)
* How much noise is added
* What date the data is shifted to
