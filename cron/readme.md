#Cron

STATUS: stopped line 190 in cron.py. The API call is not working.

This directory is a work-in-progress exercise in automation.

The high-level goal is to run a nightly "arbitrage" report which scrapes a random number of ungraded cards from ebay, grades them, and logs the cards.

Todo:
1) Make SQL table to hold the results of the nightly automation report.
    Same script as below.

2) Write a python script to 
    A) Scrape 100 random cards
    B) Grade them
    C) Record the results in a SQL table
        url     grade    title      price       date

3) Use cron to automate this collection every night.
