#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pgeocode
import numpy as np
import pandas as pd
import requests
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from feature_engine.creation import CyclicalFeatures


# ### **Feature engineering process**

# ##### Create new features:"DURATION", the difference between delivery time and pick up time; Create new features:"DURATION_DAY", "DURATION_MINUTE",  convert DURATION to days and minutes; Drop zero and negative DURATION

# In[ ]:


def add_duration(df):
    df["DURATION"] = df["DL_APPT_DATETIME"] - df["PU_APPT_DATETIME"]

    # drop negative duration
    df = df[df["DURATION"] > pd.Timedelta(0)]

    # add duration days, hours and minute
    df["DURATION_DAY"] = df["DURATION"].dt.total_seconds() / 60 / 60 / 24
    df["DURATION_HOUR"] = df["DURATION"].dt.total_seconds() / 60 / 60
    df["DURATION_MINUTE"] = df["DURATION"].dt.total_seconds() / 60

    # return dataframe
    return df


# #####Create new features: "ORIGIN_COUNTRY" and "DEST_COUNTRY" based on origin_state and dest_state; Drop 0s and nulls in ORIGIN_COUNTRY and DEST_COUNTRY

# In[ ]:


def add_country(df):
    us_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
    canada_states = ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB',
                     'SK', 'AB', 'BC', 'YT', 'NT', 'NU']

    df_us = df["ORIGIN_STATE"].isin(us_states)
    df_can = df["ORIGIN_STATE"].isin(canada_states)
    df_us = df_us.replace(
        to_replace=[True],
        value="us")
    df_us = df_us.replace(
        to_replace=[False],
        value="")
    df_can = df_can.replace(
        to_replace=[True],
        value="ca")
    df_can = df_can.replace(
        to_replace=[False],
        value="")

    df_origin_con = df_us + df_can

    df_us = df["DEST_STATE"].isin(us_states)
    df_can = df["DEST_STATE"].isin(canada_states)
    df_us = df_us.replace(
        to_replace=[True],
        value="us")
    df_us = df_us.replace(
        to_replace=[False],
        value="")
    df_can = df_can.replace(
        to_replace=[True],
        value="ca")
    df_can = df_can.replace(
        to_replace=[False],
        value="")

    df_dest_con = df_us + df_can

    df["ORIGIN_COUNTRY"] = df_origin_con
    df["DEST_COUNTRY"] = df_dest_con

    # Drop 0s and nulls ORIGIN_COUNTRY and DEST_COUNTRY
    df = df[df["ORIGIN_COUNTRY"] != ""]
    df = df[df["DEST_COUNTRY"] != ""]

    return df


# #####Create new features: "ORIGIN_LAT", "ORIGIN_LONG", "DEST_LAT", "DEST_LONG"; Latitude and longtitude for origin and destination based on zip code

# In[ ]:


# input type: pandas dataframe
def zip_to_LONG_LAT(df):
    us_nomi = pgeocode.Nominatim("us")
    ca_nomi = pgeocode.Nominatim("ca")

    def us_origin_zip_decode(zip):
        return {zip: {"ORIGIN_LAT": us_nomi.query_postal_code(zip)["latitude"],
                      "ORIGIN_LONG": us_nomi.query_postal_code(zip)["longitude"]}}

    def us_dest_zip_decode(zip):
        return {zip: {"DEST_LAT": us_nomi.query_postal_code(zip)["latitude"],
                      "DEST_LONG": us_nomi.query_postal_code(zip)["longitude"]}}

    def ca_origin_zip_decode(zip):
        return {zip: {"ORIGIN_LAT": ca_nomi.query_postal_code(zip)["latitude"],
                      "ORIGIN_LONG": ca_nomi.query_postal_code(zip)["longitude"]}}

    def ca_dest_zip_decode(zip):
        return {zip: {"DEST_LAT": ca_nomi.query_postal_code(zip)["latitude"],
                      "DEST_LONG": ca_nomi.query_postal_code(zip)["longitude"]}}

    ca_origin = df[df["ORIGIN_COUNTRY"] == "ca"]
    ca_origin = ca_origin["ORIGIN_ZIP"].unique()

    ca_dest = df[df["DEST_COUNTRY"] == "ca"]
    ca_dest = ca_dest["DEST_ZIP"].unique()

    origin_info = {}
    for i in ca_origin:
        origin_info.update(ca_origin_zip_decode(i))

    dest_info = {}
    for i in ca_dest:
        dest_info.update(ca_dest_zip_decode(i))

    us_origin = df[df["ORIGIN_COUNTRY"] == "us"]
    us_origin = us_origin["ORIGIN_ZIP"].unique()

    us_dest = df[df["DEST_COUNTRY"] == "us"]
    us_dest = us_dest["DEST_ZIP"].unique()

    for i in us_origin:
        origin_info.update(us_origin_zip_decode(i))

    for i in us_dest:
        dest_info.update(us_dest_zip_decode(i))

    origin_key = list(origin_info.keys())
    dest_key = list(dest_info.keys())

    def assign_dest_lat(zipcode):
        if zipcode in dest_key:
            return dest_info.get(zipcode).get("DEST_LAT")
        else:
            return np.nan

    def assign_dest_long(zipcode):
        if zipcode in dest_key:
            return dest_info.get(zipcode).get("DEST_LONG")
        else:
            return np.nan

    def assign_origin_lat(zipcode):
        if zipcode in origin_key:
            return origin_info.get(zipcode).get("ORIGIN_LAT")
        else:
            return np.nan

    def assign_origin_long(zipcode):
        if zipcode in origin_key:
            return origin_info.get(zipcode).get("ORIGIN_LONG")
        else:
            return np.nan

    df["ORIGIN_LAT"] = df.ORIGIN_ZIP.apply(lambda x: assign_origin_lat(x))
    df["ORIGIN_LONG"] = df.ORIGIN_ZIP.apply(lambda x: assign_origin_long(x))

    df["DEST_LAT"] = df.DEST_ZIP.apply(lambda x: assign_dest_lat(x))
    df["DEST_LONG"] = df.DEST_ZIP.apply(lambda x: assign_dest_long(x))

    df = df.dropna(subset=["ORIGIN_LAT", "DEST_LAT"])

    # output type: pandas dataframe
    return df


# #####Create new features: "DELTA_LAT","DELTA_LONG", the difference of latitude and longtitude

# In[ ]:

# Add DELTA_LAT and DELTA_LONG
def delta_lat_and_long(df):
    df["DELTA_LAT"] = df["DEST_LAT"] - df["ORIGIN_LAT"]
    df["DELTA_LONG"] = df["DEST_LONG"] - df["ORIGIN_LONG"]

    return df


# ####Create new features: "PU_YEAR","DL_YEAR","PU_MONTH","DL_MONTH","PU_WEEKDAY","DL_WEEKDAY","PU_HOUR","DL_HOUR","PU_MINUTE","DL_MINUTE" the difference of latitude and longtitude

# In[ ]:

def add_time_info(df):
    df["PU_YEAR"] = df["PU_APPT_DATETIME"].dt.year
    df["DL_YEAR"] = df["DL_APPT_DATETIME"].dt.year

    df["PU_MONTH"] = df["PU_APPT_DATETIME"].dt.month
    df["DL_MONTH"] = df["DL_APPT_DATETIME"].dt.month

    df["PU_WEEKDAY"] = df["PU_APPT_DATETIME"].dt.weekday
    df["DL_WEEKDAY"] = df["DL_APPT_DATETIME"].dt.weekday

    df["PU_HOUR"] = df["PU_APPT_DATETIME"].dt.hour
    df["DL_HOUR"] = df["DL_APPT_DATETIME"].dt.hour

    df["PU_MINUTE"] = df["PU_APPT_DATETIME"].dt.minute
    df["DL_MINUTE"] = df["DL_APPT_DATETIME"].dt.minute

    return df


# #####Create new feature: "FUEL_PRICE", add fuel price of the month in which that pick up happened 

# In[ ]:


def add_fuel(df):
    # Reading the dataframe
    url = 'https://fred.stlouisfed.org/graph/fredgraph.xls?id=CHXRSA'
    r = requests.get(url)
    open('temp.xls', 'wb').write(r.content)
    fuel_df = pd.read_excel('temp.xls', sheet_name="FRED Graph", skiprows=9)

    fuel_df["Frequency: Monthly"] = pd.to_datetime(fuel_df["Frequency: Monthly"], errors='coerce')
    fuel_df = fuel_df.dropna()
    fuel_df["PU_MONTH"] = fuel_df["Frequency: Monthly"].dt.month
    fuel_df["PU_YEAR"] = fuel_df["Frequency: Monthly"].dt.year

    fuel_df = fuel_df.rename(columns={'Unnamed: 1': 'FUEL_PRICE'})
    fuel_df = fuel_df.drop(columns=["Frequency: Monthly"])
    df = pd.merge(df, fuel_df, on=["PU_MONTH", "PU_YEAR"])

    return df


# #####Create new feature: "CROSS_COUNTRY", CROSS_COUNTRY is 1 if the shipment didn't cross crountry

# In[ ]:


def cross_country(df):
    df["CROSS_COUNTRY"] = df.apply(lambda x: 1 if x["ORIGIN_COUNTRY"] == x["DEST_COUNTRY"] else 0, axis=1)

    return df


# ##### Use target encoding on the pick up time and delivery time

# In[ ]:


def encoding(df):
    cyclical = CyclicalFeatures(variables=None, drop_original=True)
    pu_month = cyclical.fit_transform(df["PU_MONTH"].to_frame())
    dl_month = cyclical.fit_transform(df["DL_MONTH"].to_frame())
    df = df.join(pu_month)
    df = df.join(dl_month)
    df = df.drop(columns=["PU_MONTH", "DL_MONTH"])

    pu_weekday = cyclical.fit_transform(df["PU_WEEKDAY"].to_frame())
    dl_weekday = cyclical.fit_transform(df["DL_WEEKDAY"].to_frame())
    df = df.join(pu_weekday)
    df = df.join(dl_weekday)
    df = df.drop(columns=["PU_WEEKDAY", "DL_WEEKDAY"])

    pu_hour = cyclical.fit_transform(df["PU_HOUR"].to_frame())
    dl_hour = cyclical.fit_transform(df["DL_HOUR"].to_frame())
    df = df.join(pu_hour)
    df = df.join(dl_hour)
    df = df.drop(columns=["PU_HOUR", "DL_HOUR"])

    pu_minute = cyclical.fit_transform(df["PU_MINUTE"].to_frame())
    dl_minute = cyclical.fit_transform(df["DL_MINUTE"].to_frame())
    df = df.join(pu_minute)
    df = df.join(dl_minute)
    df = df.drop(columns=["PU_MINUTE", "DL_MINUTE"])

    one_hot_encoder = OneHotEncoder()
    encoder_df = pd.DataFrame(one_hot_encoder.fit_transform(df[["ACTUAL_MODE"]]).toarray())
    encoder_df.columns = one_hot_encoder.get_feature_names()

    df = df.join(encoder_df)
    df = df.drop(columns=["SHIPMENT_ID", "ACTUAL_MODE"])

    return df

