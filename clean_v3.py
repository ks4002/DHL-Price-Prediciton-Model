#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime


# In[ ]:


# ### Data Cleaning Process

# #### Drop 0s, negatives, and nulls values from the dataset

# In[ ]:


# input type: pandas dataframe
def remove_zero_null(df):
    # Drop columns "CUSTOMER", "SOURCE_LOCATION_ID", "ORIGIN_NAME", "DEST_LOCATION_ID", "ACTUAL_CARRIER", "CONSIGNEE_NAME", "PU_ARRIVAL_(X3)", "PU_DEPARTED_(AF)", "DL_ARRIVAL_(X1)", "DL_DEPARTED_(D1)"
    df = df.drop(columns=["CUSTOMER", "SOURCE_LOCATION_ID", "ORIGIN_NAME", "DEST_LOCATION_ID", 
                          "CONSIGNEE_NAME", "ACTUAL_CARRIER", "PU_ARRIVAL_(X3)", "PU_DEPARTED_(AF)", "DL_ARRIVAL_(X1)",
                          "DL_DEPARTED_(D1)"])

    # Drop negative values
    df = df[df["CASES"] >= 0]
    df = df[df["LINEHAUL_COSTS"] >= 0]
    df = df[df["FUEL_COSTS"] >= 0]
    df = df[df["ACC._COSTS"] >= 0]
    df = df[df["TOTAL_ACTUAL_COST"] >= 0]

    # Drop 0s and nulls Total_Actual_Cost
    df = df[df["TOTAL_ACTUAL_COST"] != 0]
    df = df.dropna(subset=["TOTAL_ACTUAL_COST"])

    # Drop 0s and nulls DISTANCE
    df = df[df["DISTANCE"] > 0]
    df = df.dropna(subset=["DISTANCE"])

    # Drop 0s and nulls VOLUME
    df = df[df["VOLUME"] > 0]
    df = df.dropna(subset=["VOLUME"])

    # Drop 0s and nulls WEIGHT
    df = df[df["WEIGHT"] > 0]
    df = df.dropna(subset=["WEIGHT"])

    # Drop nulls in Zip, STATE, and CITY
    df = df.dropna(subset=["ORIGIN_ZIP", "DEST_ZIP", "ORIGIN_STATE", "DEST_STATE",
                           "ORIGIN_CITY", "DEST_CITY"])

    # Drop Null in PU_APPT and DL_APPT
    df = df.dropna(subset=["PU_APPT", "DL_APPT"])

    # Keep last row of all SHIPMENT_ID
    df = df.sort_values("Insert_Date").groupby("SHIPMENT_ID").tail(1)

    # output type: pandas dataframe
    return df


# #### Summarize ACTUAL_MODE in the data into TL, LTL, and intermodal 

# In[ ]:


# input type: pandas dataframe
def mode_clean(df):
    # Add "." in front of all entries for ACTUAL_MODE in order to split the string. Some entries are "LTL", "TL"
    df["ACTUAL_MODE"] = "." + df["ACTUAL_MODE"].astype(str)

    # Split string and keep the last element of string
    df["ACTUAL_MODE"] = df["ACTUAL_MODE"].str.split(".").str.get(-1)

    # Only keep entries with "LTL", "TL", and "INTERMODAL" in ACTUAL_MODE
    mode_list = ["LTL", "TL", "INTERMODAL"]
    df = df[df["ACTUAL_MODE"].isin(mode_list)]

    # output type: pandas dataframe
    return df


# #### Keep only US states and Canada states for ORIGIN_STATE and DEST_STATE

# In[ ]:


# input type: pandas dataframe
def state_clean(df):
    # Remove all symbols from 'ORIGIN_STATE' and 'DEST_STATE'
    df["ORIGIN_STATE"] = df["ORIGIN_STATE"].map(lambda x: re.sub(r"\W+", "", x))
    df["DEST_STATE"] = df["DEST_STATE"].map(lambda x: re.sub(r"\W+", "", x))

    # Change all states into upper cases
    df["ORIGIN_STATE"] = df["ORIGIN_STATE"].str.upper()
    df["DEST_STATE"] = df["DEST_STATE"].str.upper()

    # Create dictionary with {key:value} pairs for us states and ca states. Example: {"OHIO":"OH", "FLORIDA":"FL"...}
    us_states = {"ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
                 "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
                 "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
                 "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA",
                 "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT",
                 "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
                 "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
                 "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
                 "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT",
                 "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
                 "DISTRICT OF COLUMBIA": "DC", "AMERICAN SAMOA": "AS", "GUAM": "GU", "NORTHERN MARIANA ISLANDS": "MP",
                 "PUERTO RICO": "PR", "UNITED STATES MINOR OUTLYING ISLANDS": "UM", "U.S. VIRGIN ISLANDS": "VI"}
    ca_states = {"ALBERTA": "AB", "BRITISH COLUMBIA": "BC", "MANITOBA": "MB",
                 "NEW BRUNSWICK": "NB",
                 "NEWFOUNDLAND AND LABRADOR": "NL", "NORTHWEST TERRITORIES": "NT",
                 "NOVA SCOTIA": "NS",
                 "NUNAVUT": "NU", "ONTARIO": "ON", "PRINCE EDWARD ISLAND": "PE", "QUEBEC": "QC",
                 "SASKATCHEWAN": "SK", "YUKON": "YT"}
    states = us_states | ca_states

    def state_to_abbr(x):
        if x in states.keys():
            states.get(states.values())
        elif x in states.values():
            return x
        else:
            return np.NAN

    df.ORIGIN_STATE = df.ORIGIN_STATE.apply(lambda x: state_to_abbr(x))
    df.DEST_STATE = df.DEST_STATE.apply(lambda x: state_to_abbr(x))

    df = df.dropna(subset=["ORIGIN_STATE", "DEST_STATE"])

    # output type: pandas dataframe
    return df


# #### Extract useful information from PU_APPT and DL_APPT columns and convert data type into datetime

# In[ ]:


# input type: pandas dataframe
def appointment_clean(df):
    df["PU_APPT"] = df["PU_APPT"] + " "
    df["DL_APPT"] = df["DL_APPT"] + " "

    df["PU_APPT_DATETIME"] = df["PU_APPT"].str.split(" ").str.get(0) + " " + df["PU_APPT"].str.split(" ").str.get(1)
    df["PU_APPT_TIMEZONE"] = df["PU_APPT"].str.split(" ").str.get(2)

    df["DL_APPT_DATETIME"] = df["DL_APPT"].str.split(" ").str.get(0) + " " + df["DL_APPT"].str.split(" ").str.get(1)
    df["DL_APPT_TIMEZONE"] = df["DL_APPT"].str.split(" ").str.get(2)

    df["PU_APPT_DATETIME"] = pd.to_datetime(df["PU_APPT_DATETIME"], errors='coerce')
    df["DL_APPT_DATETIME"] = pd.to_datetime(df["DL_APPT_DATETIME"], errors='coerce')

    df["PU_APPT_TIMEZONE"][df["PU_APPT_TIMEZONE"] != 'AMERICA/CHICAGO'] = timedelta(hours=0)
    df["PU_APPT_TIMEZONE"][df["PU_APPT_TIMEZONE"] == 'AMERICA/CHICAGO'] = timedelta(hours=1)

    df["DL_APPT_TIMEZONE"][df["DL_APPT_TIMEZONE"] != 'AMERICA/CHICAGO'] = timedelta(hours=0)
    df["DL_APPT_TIMEZONE"][df["DL_APPT_TIMEZONE"] == 'AMERICA/CHICAGO'] = timedelta(hours=1)

    df["PU_APPT_DATETIME"] = df["PU_APPT_DATETIME"] + df["PU_APPT_TIMEZONE"]
    df["DL_APPT_DATETIME"] = df["DL_APPT_DATETIME"] + df["DL_APPT_TIMEZONE"]

    df = df.dropna(subset=["PU_APPT_DATETIME", "DL_APPT_DATETIME"])

    df = df.drop(columns=["DL_APPT_TIMEZONE", "PU_APPT_TIMEZONE", "PU_APPT", "DL_APPT"])

    # output type: pandas dataframe
    return df


# #### Change the format of ORIGIN_ZIP and DEST_ZIP

# In[ ]:


# input type: pandas dataframe
def zip_clean(df):

    df = df.dropna(subset=["ORIGIN_ZIP", "DEST_ZIP"])
    
    df["ORIGIN_ZIP"] = df["ORIGIN_ZIP"].astype(str)
    df["DEST_ZIP"] = df["DEST_ZIP"].astype(str)
    
    # Add "0" for 4 digit zip
    df.DEST_ZIP = df.DEST_ZIP.apply(lambda x: x if len(str(x)) != 4 else "0" + x)
    df.ORIGIN_ZIP = df.ORIGIN_ZIP.apply(lambda x: x if len(str(x)) != 4 else "0" + x)

    # Change 3 digit zip to null
    df.DEST_ZIP = df.DEST_ZIP.apply(lambda x: x if len(str(x)) >= 5 else np.nan)
    df.ORIGIN_ZIP = df.ORIGIN_ZIP.apply(lambda x: x if len(str(x)) >= 5 else np.nan)

    # Change format from XXXXX-XXXX to XXXXX
    df.DEST_ZIP = df.DEST_ZIP.apply(lambda x: x if len(str(x)) != 10 else x[:5])
    df.ORIGIN_ZIP = df.ORIGIN_ZIP.apply(lambda x: x if len(str(x)) != 10 else x[:5])

    # Split CA zip
    df.DEST_ZIP = df.DEST_ZIP.apply(lambda x: x if len(str(x)) != 6 else x[:3] + " " + x[3:])
    df.ORIGIN_ZIP = df.ORIGIN_ZIP.apply(lambda x: x if len(str(x)) != 6 else x[:3] + " " + x[3:])
    df.DEST_ZIP = df.DEST_ZIP.apply(lambda x: x if len(str(x)) != 7 else x[:3] + " " + x[4:])
    df.ORIGIN_ZIP = df.ORIGIN_ZIP.apply(lambda x: x if len(str(x)) != 7 else x[:3] + " " + x[4:])

    df = df.dropna(subset=["ORIGIN_ZIP", "DEST_ZIP"])

    # output type: pandas dataframe
    return df


# #####Remove extreme values

# In[ ]:


# input type: pandas dataframe
def remove_extreme(df):
    # Exclude Extreme Value in cases
    q_cases_99 = df["CASES"].quantile(0.99)
    EXTREME = df["CASES"] > q_cases_99

    # Exclude Extreme Value in distance
    q_distance_99 = df["DISTANCE"].quantile(0.99)
    q_distance_01 = df["DISTANCE"].quantile(0.01)
    EXTREME = EXTREME + (df["DISTANCE"] > q_distance_99)
    EXTREME = EXTREME + (df["DISTANCE"] < q_distance_01)

    # Exclude Extreme Value in weight
    q_weight_95 = df["WEIGHT"].quantile(0.99)
    q_weight_05 = df["WEIGHT"].quantile(0.01)
    EXTREME = EXTREME + (df["WEIGHT"] > q_weight_95)
    EXTREME = EXTREME + (df["WEIGHT"] < q_weight_05)

    # Exclude Extreme Value in volume
    q_volume_95 = df["VOLUME"].quantile(0.99)
    q_volume_05 = df["VOLUME"].quantile(0.01)
    EXTREME = EXTREME + (df["VOLUME"] > q_weight_95)
    EXTREME = EXTREME + (df["VOLUME"] < q_weight_05)

    # Exclude Extreme Value in LINEHAUL_COSTS
    q_linehaul_95 = df["LINEHAUL_COSTS"].quantile(0.99)
    q_linehaul_05 = df["LINEHAUL_COSTS"].quantile(0.01)
    EXTREME = EXTREME + (df["LINEHAUL_COSTS"] > q_linehaul_95)
    EXTREME = EXTREME + (df["LINEHAUL_COSTS"] < q_linehaul_05)

    # Exclude Extreme Value in FUEL_COSTS
    q_linehaul_95 = df["FUEL_COSTS"].quantile(0.99)
    EXTREME = EXTREME + (df["FUEL_COSTS"] > q_linehaul_95)

    # Exclude Extreme Value in ACC._COSTS
    q_acc_95 = df["ACC._COSTS"].quantile(0.99)
    EXTREME = EXTREME + (df["ACC._COSTS"] > q_acc_95)

    # Exclude Extreme Value in "DURATION_MINUTE"
    q_acc_99 = df["DURATION_MINUTE"].quantile(0.99)
    EXTREME = EXTREME + (df["DURATION_MINUTE"] > q_acc_99)

    df = df[~EXTREME]

    # output type: pandas dataframe
    return df

