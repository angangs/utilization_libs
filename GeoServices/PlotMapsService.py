import numpy as np
from geopandas import GeoDataFrame
import folium
import os
from datetime import datetime


class PlotMapsService:
    @staticmethod
    def choropleth_locations(df, df_columns_list, output_folder=None):
        # plot using choropleth
        location = (np.mean(df.iloc[:, 0]), np.mean(df.iloc[:, 1]))
        folium_map = folium.Map(location=location, tiles="cartodbpositron", zoom_start=11)
        my_geo_df = GeoDataFrame(df, geometry='geometry_proj')['geometry_proj'].__geo_interface__
        ch = folium.Choropleth(
            geo_data=my_geo_df,
            data=df,
            columns=df_columns_list,
            key_on='feature.id',
            fill_color='RdYlGn',
            nan_fill_color='black',
            fill_opacity=.6,
            line_opacity=1,
            legend_name='Prefecture with DC',
            smooth_factor=0,
            bins=[0, 30, 60, 90, 120, 150, 180, 210],
            show=False).add_to(folium_map)
        # save the map
        ch.geojson.add_child(folium.features.GeoJsonTooltip(['tooltip'], labels=False))

        DC_pref_coords = df.groupby(['DC', 'DC Latitude', 'DC Longitude']).size().reset_index()
        d1 = {}
        for i, row in DC_pref_coords.iterrows():
            d1[row['DC']] = [row['DC Latitude'], row['DC Longitude']]

        for i in d1.keys():
            icon = folium.features.CustomIcon(
                'https://www.pngitem.com/pimgs/m/237-2374894_warehouse-es-warehouse-icons-png-transparent-png.png',
                icon_size=(25, 25))
            folium.Marker(d1[i], icon=icon).add_to(ch)


        now = datetime.now()
        dt = now.strftime("%Y_%m_%d-%H_%M_%S")
        file_name = dt + ".html"

        if output_folder:
            if not os.path.exists(output_folder + "/html"):
                os.makedirs(output_folder + "/html")
            folder_name = output_folder + "/html"
        else:
            folder_name = "./html"

        folium_map.save('/'.join([folder_name, file_name]))

    @staticmethod
    def plot_locations(df1, df2=None, output_folder=None):
        location = (np.mean(df1.iloc[:, 0]), np.mean(df1.iloc[:, 1]))
        folium_map = folium.Map(location=location, tiles="cartodbpositron", zoom_start=11)
        occurences = folium.map.FeatureGroup()

        for index, row in df1.iterrows():
            occurences.add_child(
                folium.vector_layers.CircleMarker(
                    (row[0], row[1]),
                    radius=5,  # define how big you want the circle markers to be
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6,
                )
            )

        if df2 is not None:
            for index, row in df2.iterrows():
                occurences.add_child(
                    folium.vector_layers.CircleMarker(
                        (row[0], row[1]),
                        radius=5,  # define how big you want the circle markers to be
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.6,
                    )
                )

        folium_map.add_child(occurences)

        now = datetime.now()
        dt = now.strftime("%Y_%m_%d-%H_%M_%S")
        file_name = dt + ".html"

        if output_folder:
            if not os.path.exists(output_folder + "/html"):
                os.makedirs(output_folder + "/html")
            folder_name = output_folder + "/html"
        else:
            folder_name = "./html"

        folium_map.save('/'.join([folder_name, file_name]))
        return map

    @staticmethod
    def plot_cluster_locations(df, output_folder=None):
        location = (np.mean(df['LATITUDE']), np.mean(df['LONGITUDE']))
        data = df.copy()
        lst_colors = [
            'Red', 'Purple', 'Indigo', 'Light Blue', 'Aqua', 'Green', 'Lime', 'Khaki', 'Amber', 'Deep Orange', 'Brown',
            'Gray', 'Pale Red', 'Pale Green', 'Pink', 'Deep Purple', 'Blue', 'Cyan', 'Teal', 'Light Green', 'Sand',
            'Yellow', 'Orange', 'Blue Gray', 'Light Gray', 'Dark Gray', 'Pale Yellow', 'Pale - Blue',
        ]

        factor = int(len(data) / len(lst_colors)) + 1

        lst_colors *= factor

        lst_elements = sorted(list(df["Label"].unique()))

        data["color"] = data["Label"].apply(lambda t: lst_colors[lst_elements.index(t)])

        # initialize the map with the starting location
        folium_map = folium.Map(location=location, tiles="cartodbpositron", zoom_start=11)
        occurences = folium.map.FeatureGroup()

        for index, row in data.iterrows():
            occurences.add_child(
                folium.vector_layers.CircleMarker(
                    (row['LATITUDE'], row['LONGITUDE']),
                    radius=10,  # define how big you want the circle markers to be
                    color=row['color'],
                    fill=True,
                    fill_color=row['color'],
                    fill_opacity=0.7,
                )
            )

        folium_map.add_child(occurences)
        now = datetime.now()
        dt = now.strftime("%Y_%m_%d-%H_%M_%S")
        file_name = dt + ".html"

        if output_folder:
            if not os.path.exists(output_folder + "/html"):
                os.makedirs(output_folder + "/html")
            folder_name = output_folder + "/html"
        else:
            folder_name = "./html"

        folium_map.save('/'.join([folder_name, file_name]))
        return map