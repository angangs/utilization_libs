import pandas as pd
import GeoMapsService as gms


def speed_limit(df):
    api_token = gms.js_r('../credentials/azure.json')['value']
    sl = gms.GetSpeedLimit(api_token)
    df[['Speed_Limit', 'Road_Use']] = df.apply(
        lambda row: pd.Series(sl.get_speed_limit(row['LATITUDE'], row['LONGITUDE'])), axis=1)
    return df


def traffic(df):
    api_token = gms.js_r('../credentials/azure.json')['value']
    sl = gms.GetTraffic(api_token)
    df['Traffic'] = df.apply(lambda row: sl.get_traffic(row['LATITUDE'], row['LONGITUDE']), axis=1)
    return df


def pois(df):
    api_token = gms.js_r('../credentials/azure.json')['value']
    gpfc = gms.GetPoisFromCoordinates(api_token)

    # categories of poi
    categories = pd.read_csv('categories.csv')

    # format name columns to pass to the package below
    df.rename(columns={'LONGITUDE': 'Longitude', 'LATITUDE': 'Latitude'}, inplace=True)

    # get max number of poi for each category
    poi = gpfc.get_google_max_pois(categories, loc[['Longitude', 'Latitude']], radius=1000)

    return poi
