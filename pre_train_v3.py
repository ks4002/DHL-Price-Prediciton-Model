import Clean
import FE



def main(df):

    df = clean.remove_zero_null(df)
    df = clean.mode_clean(df)
    df = clean.state_clean(df)
    df = clean.zip_clean(df)
    df = clean.appointment_clean(df)

    df = FE.add_duration(df)
    df = FE.add_country(df)
    df = FE.zip_to_LONG_LAT(df)
    df = FE.add_delta(df)
    df = FE.add_time_info(df)
    df = FE.add_fuel(df)
    df = FE.cross_country(df)
    df = FE.encoding(df)


    df = clean.remove_extreme(df)

    df = df.drop(columns=["ORIGIN_COUNTRY", "ORIGIN_CITY", "ORIGIN_STATE",
                          "ORIGIN_ZIP", "DEST_COUNTRY", "DEST_CITY", "DEST_STATE",
                          "DEST_ZIP", "ACTUAL_EQUIP", "LINEHAUL_COSTS", "FUEL_COSTS",
                          "ACC._COSTS", "Insert_Date", "DURATION"])

    return df


if __name__ == "main":
    main(df)
