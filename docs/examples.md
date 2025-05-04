# Examples

**|** &nbsp; [Overview](../README.md) &nbsp; **|** **Examples** **|**
## Table of Contents
- [Examples](#examples)
  - [Table of Contents](#table-of-contents)
  - [Getting started: Read glucose data and plot meals/activities](#getting-started-read-glucose-data-and-plot-mealsactivities)
  - [Read Meals and Notes from FreeStyle Libre Data](#read-meals-and-notes-from-freestyle-libre-data)
  - [Automatically detect meals (infered from glucose)](#automatically-detect-meals-infered-from-glucose)
## Getting started: Read glucose data and plot meals/activities
In this [Get Started](https://github.com/nidhog/glyco/blob/main/examples/Get%20started.ipynb) example, you will:
* **Read glucose data** from a CSV file. This could be from FreeStyle Libre, DexCom or any other device.
* This will generate a glucose dataframe (independent on the device/unit [more details here](concepts/glucose.md)) that you will use to:
  * **Plot the glucose curve**.
  * Plot the hourly trend (with percentiles).
  * Plot the daily trend (comparison of days).
  * Show summary statistics and more.
* You will also learn to **read meal (or activity) data** and show the response to a meal or any other type of event.

[To get a hands on go to the notebook for this example here: Get started.ipynb](https://github.com/nidhog/glyco/blob/main/examples/Get%20started.ipynb)

In addition to the above, you will see how to:
* Read data from a glucose file with different formats, Different glucose units, Different column names, and more.

## Read Meals and Notes from FreeStyle Libre Data
**This example will be provided soon, for now refer to the code documentation**
FreeStyle Libre glucose data contains meals and notes. In this example, you will:
* Read data from a glucose FreeStyle Libre CSV file.
* Read Meals based on the FreeStyle Libre notes in the same CSV file (this will generate a [meals dataframe](concepts/meals.md)).
## Automatically detect meals (infered from glucose)
Glyco has the possibility to detect meals based on variations in glucose. In this example, you will:
* Read data from a glucose CSV file.
* Detect meals based on variations in the glucose column (this will generate a [meals dataframe](concepts/meals.md))
**This example will be provided soon, for now refer to the code documentation**
