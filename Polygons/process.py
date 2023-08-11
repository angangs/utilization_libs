import geopandas as gpd
from shapely.wkt import loads
from geopandas import GeoDataFrame
from shapely.geometry import Point
import unicodedata


# remove 'tonous' from greek strings
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def read_geo_dataframe(filename):
    geo_df = gpd.read_file(filename, encoding='ANSI')
    geo_df['geometry_proj'] = geo_df['geometry'].to_crs(epsg=4326)
    geo_df = GeoDataFrame(geo_df, geometry='geometry_proj')
    return geo_df


def fix_hellenic_municipalities(municipality_shp):
    municipality_shp['NAME'] = municipality_shp['NAME'].str.upper()
    municipality_shp['NAME'] = municipality_shp['NAME'].apply(lambda row: strip_accents(row))
    # the following two lines must be included if shapefiles were downloaded from hellenic government site
    municipality_shp['NAME'] = municipality_shp['NAME'].apply(lambda row: row.replace('¶', 'Α'))
    municipality_shp.at[113, 'NAME'] = 'ΗΡΑΚΛΕΙΟΥ ΑΤΤΙΚΗΣ'
    return municipality_shp


def geo_dataframe_to_json(geo_df, json_file):
    return geo_df.to_file(json_file, driver="GeoJSON")


def geo_dataframe_to_shape(geo_df, shape_file):
    return geo_df.to_file(shape_file, driver='ESRI Shapefile', encoding='utf-8')


def map_locations_to_polygons(polygons, locations):
    polyg_list = []
    for index, row in locations.iterrows():
        flag_found = False

        point = Point(row['Latitude'], row['Longitude'])
        polyg = None
        polyg_min = None
        p_distance_min = 0.5 # in KM
        for _, polyg in polygons.iterrows():
            polygon = polyg['geometry'].convex_hull
            if polygon.contains(point) or polygon.touches(point):
                flag_found = True
                break
            elif haversine(polygon.centroid.coords[0], point.coords[0]) < p_distance_min:
                flag_found = True
                polyg_min = polyg
                break
            else:
                pass

        if flag_found:
            polyg_list.append(polyg['Name'])
        else:
            polyg_list.append(None)

        print("{} / {}: {}".format(index, len(locations), flag_found))

    return polyg_list


def locations_to_geodataframe(locations, group_by_column='label'):
    geo_df = GeoDataFrame()
    geo_df['Name'] = logins[group_by_column].drop_duplicates()
    pol_lst = []
    for i, p in geo_df.iterrows():
        if len(locations[locations[group_by_column] == p['Name']]) > 2:
            pol_lst.append(Polygon([(p['Latitude'], p['Longitude']) for i, p in logins[logins['label'] == p['Name']].iterrows()]))
        else:
            pol_lst.append(None)
    
    geo_df['geometry'] = pol_lst
    geo_df.dropna(inplace=True)
    geo_df.set_crs(epsg=4326, inplace=True)
    geo_df = GeoDataFrame(geo_df, geometry='geometry')
    return geo_df