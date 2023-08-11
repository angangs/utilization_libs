import json
from functools import lru_cache
import requests
from geopy import Nominatim
from haversine import haversine
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

LRU_SIZE = 2 ** 15


def js_r(filename):
    with open(filename) as f_in:
        return json.load(f_in)['value']


class GetCoordinatesPostalFromAddress:
    def __init__(self, api_service, country_code='GR'):
        self.api_service = api_service
        self.country_code = country_code

    @lru_cache(maxsize=LRU_SIZE)
    def get_osm_coordinates(self, addr):
        try:
            geolocator = Nominatim(user_agent=self.api_service)
            response = geolocator.geocode(addr, country_codes=self.country_code)
            lati, long = response.latitude, response.longitude
            response = geolocator.reverse(str(lati) + ',' + str(long))
            po_code = response.raw['address']['postcode']
            return lati, long, po_code
        except Exception as e:
            print('Address {} not found, {}'.format(addr, e))
            return None, None, None

    @lru_cache(maxsize=LRU_SIZE)
    def get_azure_coordinates(self, addr):
        try:
            response = requests.get('https://atlas.microsoft.com/search/address/json?'
                                    'subscription-key={subscription_key}&api-version=1.0&query={qry}&'
                                    'countrySet={country}'.format(subscription_key=self.api_service, qry=addr,
                                                                  country=self.country_code)).json()
            return response['results'][0]['position']['lat'], response['results'][0]['position']['lon'], response[
                'results'][0]['address']['postalCode']
        except Exception as e:
            print('Address {} not found, {}'.format(addr, e))
            return None, None, None

    @lru_cache(maxsize=LRU_SIZE)
    def get_google_coord(self, addr, language='el'):
        try:
            response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}&'
                                    'language={}'.format(self.api_service, addr, language))

            route = response.json()
            s = route['results'][0]['formatted_address']
            return route['results'][0]['geometry']['location']['lat'], route['results'][0]['geometry']['location'][
                'lng'], s
        except Exception as e:
            print('Address {} not found, {}'.format(addr, e))
            return None, None, None


class GetCoordinatesFromPostalCode:
    def __init__(self, api_service, country_code='GR'):
        self.api_service = api_service
        self.country_code = country_code

    @lru_cache(maxsize=LRU_SIZE)
    def get_coordinates_osm(self, postal_code):
        try:
            geolocator = Nominatim(user_agent=self.api_service)
            response = geolocator.geocode(str(postal_code), country_codes=self.country_code)
            return response.latitude, response.longitude
        except Exception as e:
            print('Zipcode {} not found, {}'.format(int(postal_code), e))
            return None, None

    @lru_cache(maxsize=LRU_SIZE)
    def get_coordinates_azure(self, postal_code):
        try:
            response = requests.get('https://atlas.microsoft.com/search/address/json?'
                                    'subscription-key={subscription_key}&api-version=1.0&query={qry}&'
                                    'countrySet={country}'.format(subscription_key=self.api_service,
                                                                  qry=str(postal_code), country=self.country_code)
                                    ).json()
            return response['results'][0]['position']['lat'], response['results'][0]['position']['lon']
        except Exception as e:
            print('Zipcode {} not found, {}'.format(int(postal_code), e))
            return None, None


class GetPostalCodeFromCoordinates:
    def __init__(self, api_service):
        self.api_service = api_service

    @lru_cache(maxsize=LRU_SIZE)
    def get_postal_code_osm(self, lat, long):
        try:
            geolocator = Nominatim(user_agent=self.api_service)
            response = geolocator.reverse(str(lat) + ',' + str(long))
            return response.raw['address']['postcode']
        except Exception as e:
            print('Lat/Lon {},{} not found, {}'.format(lat, long, e))
            return None

    @lru_cache(maxsize=LRU_SIZE)
    def get_postal_code_azure(self, lat, long):
        try:
            response = requests.get('https://atlas.microsoft.com/search/address/reverse/json?'
                                    'subscription-key={subscription_key}&api-version=1.0&query={qry}'.
                                    format(subscription_key=self.api_service, qry=str(lat) + ',' + str(long))).json()
            return response['addresses'][0]['address']['postalCode']
        except Exception as e:
            print('Lat/Lon {},{} not found, {}'.format(lat, long, e))
            return None


class GetDistance:
    def __init__(self, api_service):
        self.api_service = api_service

    @lru_cache(maxsize=LRU_SIZE)
    def get_route(self, origin_lat, origin_lon, dest_lat, dest_lon, haversine_mode=True):
        if haversine_mode:
            travel_dist = haversine((origin_lat, origin_lon), (dest_lat, dest_lon))
            travel_time, lat, lon = None, None, None
        else:
            base_url = "https://atlas.microsoft.com/route/directions/json"
            response = requests.get(
                '{}?api-version=1.0&query={},{}:{},{}&subscription-key={}'.format(
                    base_url, origin_lat, origin_lon, dest_lat, dest_lon, self.api_service
                )
            )
            print(response.status_code)
            route = response.json()
            if response.status_code == 400:
                return None, None, None, None
            summary = route["routes"][0]["summary"]
            points = route["routes"][0]["legs"][0]['points']
            lat = list(map(lambda x: x["latitude"], points))
            lon = list(map(lambda x: x["longitude"], points))
            travel_dist = summary["lengthInMeters"] / 1000
            travel_time = summary["travelTimeInSeconds"] / (60 * 60)

        return travel_dist, travel_time, lat, lon


class GetPoisFromCoordinates:
    def __init__(self, api_service, country_code='GR'):
        self.api_service = api_service
        self.country_code = country_code

    @lru_cache(maxsize=2 ** 15)
    def get_azure_pois(self, query, lati=None, long=None, radius=1000, limit=100, offset=0, language='el-GR'):
        response = requests.get('https://atlas.microsoft.com/search/poi/category/json?subscription-key='
                                '{subscription_key}&api-version=1.0&query={qry}&limit={lim}&countrySet='
                                '{countrySet}&language={language}&lat={lat}&lon={lon}&radius={rad}&ofs='
                                '{ofs}'.format(subscription_key=self.api_service, qry=query, lim=limit,
                                               countrySet=self.country_code, rad=radius, lat=lati, lon=long, ofs=offset,
                                               language=language))
        route = response.json()
        return route

    def get_azure_categories(self, in_categories):
        # POIS CATEGORIES
        response = requests.get(
            'https://atlas.microsoft.com/search/poi/category/tree/json?subscription-key={subscription_key}'
            '&api-version=1.0'.format(subscription_key=self.api_service))
        pois_category = response.json()
        pois_category_list = []
        for poi_category in pois_category['poiCategories']:
            pois_category_list.extend([c for c in poi_category['synonyms']])

        # INPUT CATEGORIES, CHECK IS EXISTS
        output_category = []
        for c in in_categories:
            if any(c in s for s in pois_category_list):
                output_category.append(c)

        return output_category

    def get_azure_max_pois(self, input_categories, loc, radius, threshold_pois=1900):
        poi_list = []
        try:
            for latitude, longitude in zip(loc['Latitude'], loc['Longitude']):
                print("Lat,Lon: {} {}".format(latitude, longitude))

                for category in input_categories['Categories']:
                    pois = self.get_azure_pois(category, lati=latitude, long=longitude, radius=radius)

                    for p in pois['results']:
                        poi_list.append({'Main Category': category,
                                         'Category': p['poi']['categories'][0],
                                         'Name': p['poi']['name'],
                                         'Latitude': p['position']['lat'],
                                         'Longitude': p['position']['lon'],
                                         })

                    if pois['summary']['totalResults'] > 100:
                        if pois['summary']['totalResults'] > 1900:
                            end = threshold_pois
                        else:
                            end = pois['summary']['totalResults']
                        for i in range(100, end, 100):
                            pois['results'].extend(
                                self.get_azure_pois(pois['summary']['query'], lati=latitude, long=longitude, offset=i,
                                                    radius=radius)['results'])
                        for p in pois['results']:
                            poi_list.append({'Main Category': category,
                                             'Category': p['poi']['categories'][0],
                                             'Name': p['poi']['name'],
                                             'Latitude': p['position']['lat'],
                                             'Longitude': p['position']['lon'],
                                             })
                        poi_list.extend(pois['results'])
                return poi_list
            return pd.DataFrame(poi_list)
        except Exception as e:
            print(e)
            return pd.DataFrame(poi_list)

    @lru_cache(maxsize=2 ** 15)
    def get_google_pois(self, query, lati=None, long=None, radius=1000, pagetoken=None):
        if pagetoken is None:
            response = requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json?key={}&'
                                    'language=el&location={},{}&radius={}&type={}'.format(self.api_service, lati, long,
                                                                                          radius, query))
        else:
            response = requests.get('https://maps.googleapis.com/maps/api/place/nearbysearch/json?key={}&'
                                    'language=el&location={},{}&radius={}&type={}&'
                                    'pagetoken={}'.format(self.api_service, lati, long, radius, query, pagetoken))
        route = response.json()
        return route

    def get_google_max_pois(self, input_categories, loc, radius, threshold_poi=60):
        poi_list = []
        try:
            for latitude, longitude in zip(loc['Latitude'], loc['Longitude']):
                print("Lat,Lon: {} {}".format(latitude, longitude))

                for category in input_categories['Categories']:
                    pois = self.get_google_pois(category, lati=latitude, long=longitude,
                                                radius=radius)

                    for p in pois['results']:
                        poi_list.append({'Main Category': category,
                                         'Category': p['types'][0],
                                         'Name': p['name'],
                                         'Latitude': p['geometry']['location']['lat'],
                                         'Longitude': p['geometry']['location']['lng'],
                                         'Origin_Latitude': latitude,
                                         'Origin_Longitude': longitude,
                                         })

                    while 'next_page_token' in pois.keys() and len(poi_list) < threshold_poi:
                        pagetoken = pois['next_page_token']
                        pois = self.get_google_pois(category, lati=latitude, long=longitude, radius=radius,
                                                    pagetoken=pagetoken)

                        for p in pois['results']:
                            poi_list.append({'Main Category': category,
                                             'Category': p['types'][0],
                                             'Name': p['name'],
                                             'Latitude': p['geometry']['location']['lat'],
                                             'Longitude': p['geometry']['location']['lng'],
                                             })

            return pd.DataFrame(poi_list)
        except Exception as e:
            print(e)
            return pd.DataFrame(poi_list)


class GetPoisFromPolygons:
    def __init__(self, api_service, polygons, country_code='GR'):
        self.api_service = api_service
        self.country_code = country_code
        self.polygons = polygons

    def get_osm_pois(self, input_categories, municipality_name='kalnameGR', region_name='Name_GR',
                     geometry='geometry_proj'):
        input_categories['Boolean'] = True
        categories = input_categories.set_index(['Category']).to_dict()['Boolean']
        pois_list = []
        for i, row in self.polygons.iterrows():
            municipality = row[municipality_name]
            region = row[region_name]
            try:
                gdf = ox.geometries_from_polygon(row[geometry], categories)
            except Exception as e:
                print(e, i, row)
                continue

            for index, subrow in gdf.iterrows():
                if 'addr:postcode' in subrow.keys():
                    postcode = subrow['addr:postcode']
                else:
                    postcode = None

                if 'population' in subrow.keys():
                    population = subrow['population']
                else:
                    population = None

                if 'name' in subrow.keys():
                    name = subrow['name']
                else:
                    name = None

                pois_list.append({
                    'Name': name,
                    'Latitude': subrow['geometry'].centroid.bounds[1],
                    'Longitude': subrow['geometry'].centroid.bounds[0],
                    'Category': [subrow[x] for x in categories.keys() if x in gdf.columns and
                                 not (subrow[x] != subrow[x])][0],
                    'Main Category': [x for x in categories.keys() if x in gdf.columns and
                                      not (subrow[x] != subrow[x])][0],
                    'Municipality': municipality,
                    'Region': region,
                    'Population': population,
                    'Postal Code': postcode})
        return pd.DataFrame(pois_list)


class GetMunicipality:
    def __init__(self, df, polygons):
        self.df = df
        self.polygons = polygons

    def get_municipality(self, municipality_name='kalnameGR', geometry='geometry_proj'):
        lat = []
        lon = []
        munic = []
        for index, row in self.df.iterrows():
            flag_found = False

            p = Point(row['Longitude'], row['Latitude'])

            polygon = None

            for _, polygon in self.polygons.iterrows():
                if polygon[geometry].contains(p):
                    flag_found = True
                    break

            if flag_found:
                munic.append(polygon[municipality_name])
                lat.append(row['Latitude'])
                lon.append(row['Longitude'])
            else:
                munic.append(None)
                lat.append(row['Latitude'])
                lon.append(row['Longitude'])

        self.df['Municipality'] = munic


class GetPopulation:
    def __init__(self, api_token):
        self.api_token = api_token
        
    def get_total_population(self, polygons):
        polygon_list = [[x, y] for x, y in zip(polygons.exterior.coords.xy[0], polygons.exterior.coords.xy[1])]
        data = [{
            'geometry': {
                'rings': [polygon_list],
                'spatialReference': {'wkid': 102100, 'latestWkid': 4326}
            }
        }]

        url = "https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/GeoEnrichment/enrich?" \
              "studyAreas=" + json.dumps(data)

        payload = 'f=json&token='+self.api_token+'&inSR=4326&outSR=4326&returnGeometry=false'

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        response = response.json()
        try:
            totalpop = response['results'][0]['value']['FeatureSet'][0]['features'][0]['attributes']['TOTPOP']
        except Exception as e:
            if 'error' in response.keys():
                print(e, response['error']['message'])
                totalpop = 'missing permissions'
            else:
                print(e)
                totalpop = None
        return totalpop


class GetSpeedLimit:
    def __init__(self, api_token):
        self.api_token = api_token

    @lru_cache(maxsize=LRU_SIZE)
    def get_speed_limit(self, lat, long):
        try:
            response = requests.get('https://atlas.microsoft.com/search/address/reverse/json?'
                                    'subscription-key={subscription_key}&api-version=1.0&query={qry}&'
                                    'returnSpeedLimit=True&'
                                    'returnRoadUse=True'.
                                    format(subscription_key=self.api_token, qry=str(lat) + ',' + str(long))).json()
            speed_limit = None
            road_use = None

            if 'speedLimit' in response['addresses'][0]['address']:
                speed_limit = response['addresses'][0]['address']['speedLimit'].split('KPH')[0]
            else:
                speed_limit = None

            if 'roadUse' in response['addresses'][0]:
                road_use = response['addresses'][0]['roadUse'][0]
            else:
                road_use = None

            return speed_limit, road_use

        except Exception as e:
            print('Lat/Lon {},{} not found, {}'.format(lat, long, e))
            return None


class GetTraffic:
    def __init__(self, api_token):
        self.api_token = api_token

    @lru_cache(maxsize=LRU_SIZE)
    def get_traffic(self, lat, long):
        try:
            response = requests.get('https://atlas.microsoft.com/traffic/flow/segment/json?'
                                    'subscription-key={subscription_key}&api-version=1.0&query={qry}&'
                                    'style=relative-delay&zoom=22'.
                                    format(subscription_key=self.api_token, qry=str(lat) + ',' + str(long))).json()
            return response['flowSegmentData']['currentSpeed']
        except Exception as e:
            print('Lat/Lon {},{} not found, {}'.format(lat, long, e))
            return None
