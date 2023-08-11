import pandas as pd
import copy
import time


def objective_function(df):
    pass


class DualAnnealingAlgorithm:

    @staticmethod
    def optimize(data_input):
        objective_values_list = []
        iteration_counter = [0]
        data = copy.deepcopy(data_input)
        number_of_vars = data['raw_vector']['Variable'].tolist().index('BasePrice')
        print("Optimizing first {} variables".format(number_of_vars))
        start_time_optimizer = time.time()

        def plot_convergence(objective_values):
            import matplotlib.pyplot as plt
            plt.plot(objective_values)
            plt.title("Objective Value vs Iterations")
            plt.xlabel('Iterations')
            plt.ylabel('Objective Value')
            plt.show()

        def fit_function(x):
            res = objective_function(x)
            return -res

        def round_to_nearest_integer(df):
            for i in range(len(df)):
                if data['raw_vector'].VectorConstrainType.iloc[i] == 'int':
                    df.Vector.iloc[i] = int(round(df.Vector.iloc[i]))
                else:
                    pass
            return df

        def define_space(df):
            print("Defining space")
            df_tmp = df[:number_of_vars]
            bounds = []
            for v_min, v_max in zip(df_tmp['VectorConstrainMin'], df_tmp['VectorConstrainMax']):
                bounds.append((float(v_min), float(v_max)))
            return bounds

        def build_init_points(init_method, df=None, xlsx_name=None):
            print("Building initial points")
            init_vec = None
            if init_method == 1:
                init_vec = None
            elif init_method == 2:
                init_vec = df['Vector'].tolist()[:number_of_vars]
            elif init_method == 3:
                min_val = df['VectorConstrainMin']
                max_val = df['VectorConstrainMax']
                add_factor = ((max_val-min_val)*0.5).astype(int)
                init_vec = (min_val+add_factor).tolist()[:number_of_vars]
            elif init_method == 4:
                init_vec = pd.read_excel(xlsx_name, index_col=None)['Vector'].tolist()[:number_of_vars]
            else:
                pass
            return init_vec


        def callbackf(_, e, __):
            end_time = time.time()
            print(
                "---------------Found maximum {:.1f} after {:.1f} s".format(-e, end_time - start_time_optimizer))

        def dualanneal_optimizer(df):
            from scipy.optimize import dual_annealing
            import math

            bounds = define_space(df)
            init_vec = build_init_points(2, df=df)

            maxglobaliter = 12
            res = dual_annealing(fit_function,
                                 x0=init_vec,
                                 bounds=bounds,
                                 seed=1234,
                                 maxiter=maxglobaliter,
                                 maxfun=maxglobaliter * number_of_vars * 2,
                                 initial_temp=-1 / math.log(0.9999),
                                 restart_temp_ratio=2e-5,
                                 visit=2.62,
                                 accept=-5,
                                 no_local_search=True,
                                 callback=callbackf
                                 )

            print("Global minimum: xmin = {}, fit_function(xmin) = {}".format(res.x, res.fun))


            plot_convergence(objective_values_list)

            return res.x, res.fun

        result = dualanneal_optimizer(data['raw_vector'])
        return result
