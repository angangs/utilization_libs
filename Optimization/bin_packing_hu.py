from __future__ import print_function
from ortools.linear_solver import pywraplp
import pandas as pd

from pathlib import Path
import sys
import os

sys.path.append(str(Path.cwd())+'/Vaccination-c19')
os.chdir(str(Path.cwd())+'/Vaccination-c19')


def main():
    centroids = pd.read_csv('final_centroids.csv', sep=';', index_col=None)
    centroids.drop(['Unnamed: 0', 'Count'], axis=1, inplace=True)
    vaccination = pd.read_csv('GM_Vaccination_Locations_v7.csv', sep=';', index_col=None)
    vaccination_locations = vaccination[['LAT', 'LON']]
    vaccination_population = vaccination['Πληθυσμός']
    vaccination_centres = vaccination['Εμβολιαστικά Ιατρεία']

    dc_distances = pd.read_csv('distances.csv', index_col=None)
    centroids.drop([int(i) for i in dc_distances.columns[dc_distances.isna().any()].tolist()], axis=0, inplace=True)
    dc_distances.dropna(axis=1, inplace=True)

    # brown
    extra_loc = [
        (38.13885188, 23.83734748),
        (40.54674427, 23.01833977),
        (39.34818091, 21.92071828),
        (35.33555501, 25.15816988),
        (38.22235254, 21.76233236),
        (39.64147795, 20.82811431)
    ]
    # centroids = centroids.append(pd.DataFrame(extra_loc, columns=['Latitude', 'Longitude'])).reset_index(drop=True)
    # as is
    centroids = centroids.iloc[-len(extra_loc):, :].reset_index(drop=True)
    dc_distances = dc_distances.iloc[:, -len(extra_loc):].reset_index(drop=True)
    dc_distances.columns = list(range(len(extra_loc)))

    data = {
        'population': vaccination_population.tolist(),
        'centre': vaccination_centres.tolist(),
        'distance': dc_distances.to_numpy(),
        'items': list(range(len(vaccination_locations))),
        'bins': list(range(len(centroids))),
        # brown
        # 'distance_capacity': 120,
        # 'population_capacity': 1200000,
        # 'health_unit_capacity': 140
        # green
        # 'distance_capacity': 120,
        # 'population_capacity': 1200000,
        # 'health_unit_capacity': 140
        # fixed 6
        'distance_capacity': 380,
        'population_capacity': 17000000,
        'health_unit_capacity': 1700
    }

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data['items']:
        for j in data['bins']:
            x[i, j] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # y[j] = 1 if bin j is used.
    y = {}
    for j in data['bins']:
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)


    # Constraints

    # Open bins for brown and as is scenarios
    solver.Add(sum(y[j] for j in data['bins'][-len(extra_loc):]) == len(extra_loc))

    # Each item must be in exactly one bin.
    for i in data['items']:
        solver.Add(sum(x[i, j] for j in data['bins']) == 1)

    # The amount packed in each bin cannot exceed its capacity.
    for j in data['bins']:
        solver.Add(
            sum([x[i, j] * data['population'][i] for i in data['items']]) <= y[j] * data['population_capacity'])

        for i in data['items']:
            solver.Add(
                x[i, j] * data['distance'][i][j] <= y[j] * data['distance_capacity'])

        solver.Add(
            sum([x[i, j] * data['centre'][i] for i in data['items']]) <= y[j] * data['health_unit_capacity'])

    # Objective

    # solver.Minimize(
    #     solver.Sum([x[(i, j)] * data['distance'][i][j] for i in data['items'] for j in data['bins']]))

    # solver.Minimize(solver.Sum([y[j] for j in data['bins']]))

    solver.Minimize(
        sum([99 * y[j] + x[i, j] * data['distance'][i][j] for j in data['bins'] for i in data['items']])
    )

    solver.set_time_limit(60*1000)
    status = solver.Solve()

    results = []
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        num_bins = 0.
        for j in data['bins']:
            if y[j].solution_value() == 1:
                bin_items = []
                population_weight = 0
                distance_weight = []
                for i in data['items']:
                    if x[i, j].solution_value() > 0:
                        bin_items.append(i)
                        population_weight += data['population'][i]
                        distance_weight.append(data['distance'][i, j])
                        dict_results = {
                            'Prefecture_Index': i,
                            'Latitude':  vaccination_locations.loc[i]['LAT'],
                            'Longitude': vaccination_locations.loc[i]['LON'],
                            'Distance': round(data['distance'][i, j], 2),
                            'Vaccination Centre': int(data['centre'][i]),
                            'Population': int(data['population'][i]),
                            'Centroid Lat': centroids.loc[j]['Latitude'],
                            'Centroid Lon': centroids.loc[j]['Longitude']
                        }
                        results.append(dict_results)
                if len(bin_items) > 0:
                    num_bins += 1
                    print('Bin number', j)
                    print('   Items packed:', bin_items)
                    print('   Total population:', population_weight)
                    print('   Max distance:', round(max(distance_weight), 2))
                    print('   Total health units:', len(bin_items))
        print('Number of bins used:', num_bins)
        print('Time = ', solver.WallTime(), ' milliseconds')
    else:
        print('The problem is infeasible.')

    df = pd.DataFrame(results)
    df.to_csv('vaccination_dc_green.csv', sep=';', index=False)


if __name__ == '__main__':
    main()
    os.chdir(str(Path.cwd()) + '/..')
