import os
import pandas as pd
import matplotlib.pyplot as plt
import shapely.wkt
from ortools.linear_solver import pywraplp
from GeoServices import GeoMapsService as gms

dir_path = os.path.dirname(os.path.abspath("__file__"))
AZURE_API_KEY = gms.js_r(dir_path+'/../../credentials/azure.json')['value']
gd = gms.GetDistance(AZURE_API_KEY)
counter = [0]


def geodist(lat1, lon1, lat2, lon2):
    counter.append(counter[-1] + 1)
    print(counter[-1])
    travel_dist, travel_time, lat, lon = gd.get_route(lat1, lon1, lat2, lon2, haversine_mode=True)
    return travel_dist


hu = pd.read_csv(dir_path+'/Post_MILP_vaccination_dc_asis.csv', sep=';', index_col=None, encoding='utf-8-sig')

hu_top_dim_index = pd.read_csv(dir_path+'/hu_top_dim_indices.csv', sep=';', index_col=None)
hu_top_dim_index.drop(['Unnamed: 0'], axis=1, inplace=True)
hu_top_dim_distances = pd.read_csv(dir_path+'/hu_top_dim_distances.csv', sep=';', index_col=None)
hu_top_dim_distances.drop(['Unnamed: 0'], axis=1, inplace=True)

top_dim = pd.read_csv(dir_path+'/KoinotitesMainlandPopulation.csv', sep=';', index_col=None, encoding='utf-8-sig')
top_dim['LON'] = top_dim.apply(lambda row: shapely.wkt.loads(row['geometry']).x, axis=1)
top_dim['LAT'] = top_dim.apply(lambda row: shapely.wkt.loads(row['geometry']).y, axis=1)
top_dim_population = top_dim[['Population']]

mean_hu_top_dim_distances = hu_top_dim_distances.sum(axis=1) / hu_top_dim_index.shape[1]
assign_to_1 = list(mean_hu_top_dim_distances[mean_hu_top_dim_distances >= 80].index.values)
avg = top_dim['Population'].sum() // hu.shape[0]

data = {
    'population': top_dim_population['Population'].tolist(),
    'distance': hu_top_dim_distances.to_numpy(),
    'items': list(range(hu_top_dim_index.shape[0])),
    'bins': list(range(hu_top_dim_index.shape[1])),
    'population_capacity': (hu['ΩΡΑΡΙΟ ΛΕΙΤΟΥΡΓΙΑΣ'].fillna(1)*hu['Εμβολιαστικά Ιατρεία']).astype(str).apply(
        lambda row: row.split(' ')[0]).astype(int) * (8 * 60 / 10) * 365 * 2,
    'distance_threshold': 80,
    'avg': top_dim['Population'].sum() // hu.shape[0],
    'total_df_pop_threshold': (top_dim['Population'].sum() // hu.shape[0]) * 1.5
}

# Create the mip solver with the SCIP backend.
solver = pywraplp.Solver.CreateSolver('GLOP')

# VARIABLES i Topikes Dhmotikes Koinotita, j Health Unit
x = {}
for i in data['items']:
    for j in data['bins']:
        if i not in assign_to_1:
            x[i, hu_top_dim_index.iloc[i, j]] = solver.NumVar(0, data['population'][i],
                                                              'x_%i_%i' % (i, hu_top_dim_index.iloc[i, j]))

    if i in assign_to_1:
        x[i, hu_top_dim_index.iloc[i, 0]] = solver.NumVar(data['population'][i], data['population'][i],
                                                          'x_%i_%i' % (i, hu_top_dim_index.iloc[i, 0]))

# VARIABLES Delta Population
delta = solver.NumVar(0, data['total_df_pop_threshold'], 'delta')

# CONSTRAINTS For each TOP Sum of population for each HU assigned to this TOP must be equal to the total population of the TOP
for i in data['items']:
    if i not in assign_to_1:
        solver.Add(sum(x[i, hu_top_dim_index.iloc[i, j]] for j in data['bins'] if (i, hu_top_dim_index.iloc[i, j])
                       in x.keys()) == data['population'][i])
    else:
        solver.Add(x[i, hu_top_dim_index.iloc[i, 0]] == data['population'][i])

# CONSTRAINTS Distance between HU and TOP must be lower than a threshold
for j in data['bins']:
    for i in data['items']:
        if (i, hu_top_dim_index.iloc[i, j]) in x.keys():
            solver.Add(
                x[i, hu_top_dim_index.iloc[i, j]] * data['distance'][i][j] <=
                x[i, hu_top_dim_index.iloc[i, j]] * data['distance_threshold']
            )

# CONSTRAINTS For each HU respect its population capacity
for j in range(hu.shape[0]):
    if j in [y[1] for y in x.keys()]:
        solver.Add(sum(x[i, j] for i in data['items'] if (i, j) in x.keys()) <= data['population_capacity'][j])

# CONSTRAINTS for uniform distribution |X|-avg<delta
for j in range(hu.shape[0]):
    solver.Add(sum(x[i, j] for i in data['items'] if (i, j) in x.keys()) >= avg - delta)
    solver.Add(sum(x[i, j] for i in data['items'] if (i, j) in x.keys()) <= avg + delta)

# OBJECTIVE "approximately uniform" distribution: minimize sum( abs diff tou utilization apo to mean )
solver.Minimize(delta)
solver.set_time_limit(1 * 60 * 1000)
status = solver.Solve()

if status != 2:
    print('SOLVED')
else:
    print('UNSOLVED')

if status != 2:
    lon_list_hu = []
    lat_list_hu = []
    lon_list_top = []
    lat_list_top = []
    pop_list = []
    distance_list = []
    index_list = []
    index_koin_list = []
    for i, j in x.keys():
        if x[i, j].solution_value() != 0:
            index_list.append(j)
            lon_list_hu.append(hu.iloc[j]['LON'])
            lat_list_hu.append(hu.iloc[j]['LAT'])
            lon_list_top.append(top_dim.iloc[i]['LON'])
            lat_list_top.append(top_dim.iloc[i]['LAT'])
            pop_list.append(x[i, j].solution_value())
            index_koin_list.append(i)
            distance_list.append(
                geodist(hu.iloc[j]['LAT'], hu.iloc[j]['LON'], top_dim.iloc[i]['LAT'], top_dim.iloc[i]['LON']))

    df = pd.DataFrame({'Index_HU': index_list,
                       'Index_Koin': index_koin_list,
                       'HU_LON': lon_list_hu,
                       'HU_LAT': lat_list_hu,
                       'TOP_LON': lon_list_top,
                       'TOP_LAT': lat_list_top,
                       'POPULATION': pop_list,
                       'DISTANCE': distance_list})

    df = df.merge(pd.DataFrame(data['population_capacity'], columns=['POPULATION_CAPACITY']), left_on='Index_HU',
                  right_index=True)

    df['UTILIZATION'] = df['POPULATION'] / df['POPULATION_CAPACITY']

    df.to_csv('map_population2hu.csv', index=False, sep=';')

    print("Used HU: {}".format(len(df.drop_duplicates(subset=['HU_LON', 'HU_LAT']))))

    df.groupby(by=['HU_LON', 'HU_LAT']).sum()['POPULATION'].hist(bins=30)
    plt.title('Distribution of Population across capacitated Health Units')
    plt.show()

    df.groupby(by=['HU_LON', 'HU_LAT']).mean()['DISTANCE'].hist(bins=30)
    plt.title('Distribution of Mean Distance across capacitated Health Units')
    plt.show()



    # cms.plot_locations(df[['HU_LAT', 'HU_LON']], df2=df[['TOP_LAT', 'TOP_LON']], output_folder=dir_path)
    # df['Label'] = df['Index_HU']
    # df.rename(columns={
    #     'TOP_LAT': 'Latitude',
    #     'TOP_LON': 'Longitude',
    #     'HU_LAT': 'Latitude_marked',
    #     'HU_LON': 'Longitude_marked'
    # }, inplace=True)
    # cms.plot_cluster_locations(df, output_folder=dir_path)
