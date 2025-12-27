# Issues and Plans for next releases
## What is planned for the coming releases
The next release will contain:
* Advanced meal and event analysis: generating features for meals, such as how much glucose a meal contained and comparing meals with each other in terms of glucose.
* Simplify the Generated Glucose DataFrame by reducing unused columns.

## Issues to fix
* Move constants from glucose.py to constants.py
* Do not use the defaults for constants in meals.py (if they change in the user input this will not work)