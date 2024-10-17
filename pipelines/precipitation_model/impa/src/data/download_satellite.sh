#!/bin/sh

YEAR="$1"
DAY="$2"
PRODUCTS="ABI-L2-RRQPEF "ABI-L2-ACHAF""

umask 000

for product in $PRODUCTS
do
  path="data/raw/satellite/"$product"/"$YEAR"/"$DAY"/"
  poetry run aws s3 sync s3://noaa-goes16/"$product"/"$YEAR"/"$DAY" "$path" --no-sign-request
done
