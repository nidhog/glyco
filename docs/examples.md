# Examples

**|** &nbsp; [Overview](../README.md) &nbsp; **|** **Examples** **|**
## Table of Contents
- [Examples](#examples)
  - [Table of Contents](#table-of-contents)
  - [Read Glucose Data and Plot trends](#read-glucose-data-and-plot-trends)
  - [Read Glucose Data in a different unit or a different format](#read-glucose-data-in-a-different-unit-or-a-different-format)
  - [Read Meals and Notes from FreeStyle Libre Data](#read-meals-and-notes-from-freestyle-libre-data)
  - [Automatically detect meals (infered from glucose)](#automatically-detect-meals-infered-from-glucose)
  - [Plot the Glucose response to a Meal](#plot-the-glucose-response-to-a-meal)
## Read Glucose Data and Plot trends
In this example, you will:
* Read data from a glucose CSV file.
* This will generate a [glucose dataframe](concepts/glucose.md) that you will use to:
  * Plot the glucose curve.
  * Plot the hourly trend (with percentiles).
  * Plot the daily trend (comparison of days).
## Read Glucose Data in a different unit or a different format
In this example, you will:
* Read data from a glucose file with different formats:
  * Different glucose units.
  * Different column names.
  * and more.
* This will generate a [glucose dataframe with the generated glucose properties](concepts/glucose.md).
## Read Meals and Notes from FreeStyle Libre Data
FreeStyle Libre glucose data contains meals and notes. In this example, you will:
* Read data from a glucose FreeStyle Libre CSV file.
* Read Meals based on the FreeStyle Libre notes in the same CSV file (this will generate a [meals dataframe](concepts/meals.md)).
## Automatically detect meals (infered from glucose)
Glyco has the possibility to detect meals based on variations in glucose. In this example, you will:
* Read data from a glucose CSV file.
* Detect meals based on variations in the glucose column (this will generate a [meals dataframe](concepts/meals.md))
## Plot the Glucose response to a Meal
In this example, you will:
* Read data from a glucose CSV file (This will generate a [glucose dataframe with the generated glucose properties](concepts/glucose.md)).
* Read data about meals from a CSV file (this will generate a [meals dataframe](concepts/meals.md)).
* Plot the glucose response to a meal based on these two dataframes.