# Data Cleaning

## Generic
- datatypes checking
- drop duplicate rows
- drop NAN or null value rows

## Holiday Data
- standardize holiday names (all string and in lowercase or title case)
- strip whitespaces

## Weather Data
- drop rows if conditions is not clear, clouds, light_rain or heavy_rain
- temperatures between -60 and 60
- check humidity between 0 and 100 (inclusive)
- check windspeed greater than equal to 0

# Dagster

# RustFS