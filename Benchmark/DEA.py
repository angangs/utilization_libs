import copy
from pulp import *
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull


class DEABenchmarking:
    def __init__(self, input_data, input_names, output_names, dmu_identifier):
        self.input_data = input_data
        self.input_names = input_names
        self.output_names = output_names
        self.dmu_identifier = dmu_identifier

    def pre_processing(self):
        # for names_temp in self.input_names + self.output_names:
        for names_temp in self.output_names:
            self.input_data = self.input_data[self.input_data[names_temp] > 0]

        self.inputs = self.input_data[[self.dmu_identifier] + self.input_names].set_index(self.dmu_identifier)
        self.input_dict = self.inputs.to_dict(orient='index')
        self.outputs = self.input_data[[self.dmu_identifier] + self.output_names].set_index(self.dmu_identifier)
        self.output_dict = self.outputs.to_dict(orient='index')

        self.DMUnames = self.input_data[self.dmu_identifier].unique().tolist().copy()
        self.DMUnames_temp = self.input_data[self.dmu_identifier].unique().tolist().copy()

    def convex_hull_estimation(self):
        def append_to_convex_hull(names_list, input_df, descriptives, list_to_append_to):
            for name_temp in names_list:
                list_to_append_to.append(
                    input_df[input_df[name_temp] == descriptives[name_temp]['min']].index.tolist())
                list_to_append_to.append(
                    input_df[input_df[name_temp] == descriptives[name_temp]['max']].index.tolist())

            return list_to_append_to

        points_to_find_convex_hull = np.array([[self.input_dict[dmu][input_temp] for input_temp in self.input_names] + [
            self.output_dict[dmu][output_temp] for output_temp in self.output_names] for dmu in self.DMUnames])
        hull = ConvexHull(points_to_find_convex_hull)

        potentially_efficient_DMUs = [DMU for i, DMU in enumerate(self.DMUnames) if i in list(hull.vertices)]

        input_descriptives = self.inputs.describe().to_dict()
        output_descriptives = self.outputs.describe().to_dict()

        potentially_efficient_DMUs_extra = []

        append_to_convex_hull(self.input_names, self.inputs, input_descriptives, potentially_efficient_DMUs_extra)
        append_to_convex_hull(self.output_names, self.outputs, output_descriptives, potentially_efficient_DMUs_extra)

        potentially_efficient_DMUs.extend([elem1 for elem in potentially_efficient_DMUs_extra for elem1 in elem])
        potentially_efficient_DMUs = list(set(potentially_efficient_DMUs))

        non_efficient_DMUs = [DMU for i, DMU in enumerate(self.DMUnames) if DMU not in potentially_efficient_DMUs]

        non_efficient_dmus_in_potentially_efficient_dmus = [1]
        while len(non_efficient_dmus_in_potentially_efficient_dmus) > 0:
            non_efficient_dmus_in_potentially_efficient_dmus = []
            for DMU in potentially_efficient_DMUs:
                results = self.lp_solution(potentially_efficient_DMUs, DMU)
                if results < 0.999999:
                    non_efficient_dmus_in_potentially_efficient_dmus.append(DMU)
                    non_efficient_DMUs.append(DMU)
                    potentially_efficient_DMUs.remove(DMU)

            self.efficient_DMUs = copy.deepcopy(potentially_efficient_DMUs)

    def testing_convex_hull_results(self):
        for DMU in self.efficient_DMUs:
            results = self.lp_solution(self.DMUnames, DMU)
            if results < 1: print(DMU, results)

    def dea_convex_hull(self, to_print_dmu_list_len=False):
        self.pre_processing()
        self.convex_hull_estimation()
        self.efficient_level_input_output = pd.DataFrame(index=self.DMUnames, columns=self.input_names + self.output_names + ['Theta'])
        self.most_efficient_DMUs = []
        self.dmu_theta_list = []
        for i, DMU in enumerate(self.DMUnames):
            if i % 100 == 0:
                print(i, "                     ", end="\r")
            DMU_list_dea = self.efficient_DMUs + [DMU] if DMU not in self.efficient_DMUs else copy.deepcopy(self.efficient_DMUs)
            if to_print_dmu_list_len: print(len(DMU_list_dea))
            theta, self.most_efficient_DMUs = self.lp_solution(DMU_list_dea, DMU, self.most_efficient_DMUs, self.efficient_level_input_output)

            self.dmu_theta_list.append([DMU, theta])
            self.efficient_level_input_output.loc[DMU, 'Theta'] = theta

    def dea_reducted_sample(self):
        self.pre_processing()
        self.efficient_level_input_output = pd.DataFrame(index=self.DMUnames, columns=self.input_names + self.output_names + ['Theta'])
        self.most_efficient_DMUs = []
        self.dmu_theta_list = []
        for i, DMU in enumerate(self.DMUnames):
            if i % 100 == 0:
                print(i, "                     ", end="\r")
            theta, self.most_efficient_DMUs = self.lp_solution(self.DMUnames_temp, DMU, self.most_efficient_DMUs, self.efficient_level_input_output)

            self.dmu_theta_list.append([DMU, theta])
            self.efficient_level_input_output.loc[DMU, 'Theta'] = theta

            if theta <= 0.999999:
                self.DMUnames_temp.remove(DMU)

    def lp_solution(self, dmu_names, reference_dmu, most_efficient_dmus=None, df_efficient_level_input_output=None):
        dmu_names = [str(dmu) for dmu in dmu_names]

        prob = LpProblem('Store efficiency problem', LpMinimize)
        Theta = LpVariable('Theta', lowBound=0, upBound=1, cat=LpContinuous)
        Lamdas = {'L_' + dmu: LpVariable('L_' + dmu, lowBound=0, upBound=1, cat=LpContinuous) for dmu in dmu_names}
        prob += Theta

        for name in self.input_names:
            LHS_inputs = lpSum(Lamdas['L_' + dmu] * self.input_dict[dmu][name] for dmu in dmu_names)
            RHS_inputs = Theta * self.input_dict[reference_dmu][name]
            prob += LHS_inputs <= RHS_inputs

        for name in self.output_names:
            LHS_outputs = lpSum(Lamdas['L_' + dmu] * self.output_dict[dmu][name] for dmu in dmu_names)
            RHS_outputs = self.output_dict[reference_dmu][name]
            prob += LHS_outputs >= RHS_outputs

        prob += lpSum(Lamdas['L_' + dmu] for dmu in dmu_names) == 1

        optimization_result = prob.solve(pulp.PULP_CBC_CMD(msg=False))

        assert optimization_result == LpStatusOptimal

        if most_efficient_dmus is None:
            return Theta.value()
        else:
            peers = []

            if df_efficient_level_input_output is not None:
                for name in self.output_names:
                    df_efficient_level_input_output.loc[reference_dmu, name] = self.output_dict[reference_dmu][name] / Theta.value()

            for name in self.input_names:
                temp_sum = 0
                for var in Lamdas:
                    if Lamdas[var].value() > 0:
                        temp_sum += Lamdas[var].value() * self.input_dict[Lamdas[var].name[2:]][name]

                        peers.extend([Lamdas[var].name[2:]])

                if df_efficient_level_input_output is not None:
                    df_efficient_level_input_output.loc[reference_dmu, name] = temp_sum

            peers = list(set(peers))

            most_efficient_dmus += peers

            return Theta.value(), most_efficient_dmus



# dea_benchmarking_results = dea_benchmarking(df_input, df_target, input_names)