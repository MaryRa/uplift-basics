# Uplift basics

The main purposes of this repo are 
 1. showing how different uplift models work on the basic models where it’s 
possible, 
 2. giving you an instrument which generate synthetic datasets in order to give 
you an opportunity to play with your uplift models


## How to use
1. Clone the repo: `https://github.com/MaryRa/uplift-basics.git` <br/>
2. Go through ipynb-files:
   - `1.1.communicate_or_not (binary problem).ipynb` - it shows how to work 
   in situation when we have binary target. For example if there were 
   purchases or not.
   - `1.2.discount_value (regression problem).ipynb` - it shows how to work 
   in situation when we have non-binary target - for example the value of purchases.
   - `2.metrics.ipynb` - it contains some metrics to check quality of your model
 
## Additional information
   Synthetic data is used in all these modules. Algorithm generating synthetic 
   data `synthetic_data.py`:
1. Creates users with purchase probability distributed as uniform
2. Randomly chooses characteristics (age, gender, …) for users which will 
generate more purchases if they get the discount and increase their purchase 
probability if they get a discount.

Since the pattern is known, there is an opportunity to run 'AB test' as we 
would do it in real life, that's why you can see AB-test sections in ipynb-files
