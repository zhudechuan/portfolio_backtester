import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def estimate_para(data):
    sigma = data.cov(ddof=1).values
    beta = data.mean().values
    p = sigma.shape[0]
    A = np.ones((1, p))
    # A=np.array([[i+1 for i in range(p)]])
    b, k = 100, 1

    beta = 0*beta
    return sigma, beta, p, A, b, k

from docplex.mp.model import Model


def find_lambda_max_cplex(sigma, beta, A, b, p, k, factor):
    # lp1
    model = Model('lp1')
    model.parameters.read.scale = -1
    # model.parameters.barrier.convergetol = 1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4

    # define variables
    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))

    # set objective
    expr = model.sum([model.abs(w[i]) for i in range(p)])
    model.minimize(expr)

    # add constraints
    for row in range(k):
        expr = model.dot(w, A[row])
        model.add_constraint(expr == b)

    # solve and get the results
    solution = model.solve()

    if model.solve_status.value == 2:
        lp1_norm = solution.get_objective_value()
        model.clear()
    else:
        model.clear()
        print('Infeasible lp1')

    # lp2
    # variables
    model = Model('lp2')
    model.parameters.read.scale = -1
    #
    # model.parameters.barrier.convergetol = 1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4

    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))
    _lambda_scaled = model.continuous_var(name='lambda')

    # objective
    model.minimize(_lambda_scaled)

    # constraints
    for row in range(p):
        expr = factor * (model.dot(w, sigma[row]) +
                         model.dot(gamma, A.T[row])) - _lambda_scaled
        model.add_constraint(expr <= 0)

        expr = factor * (model.dot(w, sigma[row]) +
                         model.dot(gamma, A.T[row])) + _lambda_scaled
        model.add_constraint(expr >= 0)

    for row in range(k):
        expr = model.dot(w, A[row])
        model.add_constraint(expr == b)

    expr = model.sum([model.abs(w[i]) for i in range(p)])
    model.add_constraint(expr == lp1_norm)

    # solve and get results
    solution = model.solve()

    if model.solve_status.value == 2:
        lambda_max = solution.get_value(_lambda_scaled)
        if lambda_max != 0:
            new_factor = 1 / lambda_max * factor
        else:
            new_factor = factor

        model.clear()
    else:
        model.clear()
        return 'infeasible lp2'

    return lambda_max, new_factor


def CDE_DOcplex_phase1(sigma, beta, A, b, p, k, factor, _lambda_scaled):
    # Create DOcplex model
    model = Model(name='phase 1')

    model.parameters.read.scale = -1
    # model.parameters.barrier.convergetol=1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4

    # Define variables
    # w_plus = np.array([model.continuous_var(name=f'w_plus{i}', lb=0) for i in range(p)])
    # w_minus = np.array([model.continuous_var(name=f'w_minus{i}', lb=0) for i in range(p)])
    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))

    # Set objective
    expr = model.sum(model.abs(w[i]) for i in range(p))
    model.minimize(expr)

    # Add constraints
    for row in range(p):
        expr = factor * (model.dot(w, sigma[row]) +
                         model.dot(gamma, A.T[row])) - _lambda_scaled
        model.add_constraint(expr <= 0)

        expr = factor * (model.dot(w, sigma[row]) +
                         model.dot(gamma, A.T[row])) + _lambda_scaled
        model.add_constraint(expr >= 0)

    for row in range(k):
        expr = model.dot(w, A[row])
        model.add_constraint(expr == b)

    # Solve the problem
    solution = model.solve()

    if model.solve_status.value == 2:
        initial_w = np.array(solution.get_values(w))
        model.clear()
    else:
        model.clear()
        return 'Infeasible CDE_phase 1'

    return initial_w


def CDE_DOcplex_phase2_with_control(sigma,beta,A,b,p,k,factor,_lambda_scaled,target_norm, benchmark_w):
    model = Model(name='phase 2')

    # Increase tolerance on feasilibity
    model.parameters.read.scale = -1
    # model.parameters.barrier.convergetol = 1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4
    # model.parameters.preprocessing.qtolin = 0

    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))

    # perturb objective functions
    # w = w_p - w_m
    l1_norm=model.sum(model.abs(benchmark_w[i] - w[i]) for i in range(p))
    model.minimize(l1_norm)

    # Add constraints
    for row in range(p):
        expr = factor * (model.dot(w, sigma[row]) +
                         model.dot(gamma, A.T[row])) - _lambda_scaled
        model.add_constraint(expr <= 0)

        expr = factor * (model.dot(w, sigma[row]) +
                         model.dot(gamma, A.T[row])) + _lambda_scaled
        model.add_constraint(expr >= 0)

    for row in range(k):
        expr = model.dot(w, A[row])
        model.add_constraint(expr == b)

    expr = model.sum(model.abs(w[i]) for i in range(p))
    model.add_constraint(expr == target_norm)

    # Solve the problem
    solution = model.solve()

    if model.solve_status.value == 2:
        enumerate_w = np.array(solution.get_values(w))
        model.clear()
    else:
        # print(model.solve_status)
        # raise Exception('Infeasbiel phase 2 (quadratic)')
        # print('Infeasible phase 2 (enumeration)')
        print(model.solve_status)
        model.clear()
        return 'Infeasible CDE_phase 2'

    return enumerate_w


def CDE_DOcplex_with_control(sigma, beta, A, b, p, k, factor, _lambda_scaled, benchmark_w):
    # phase 1 (default)
    initial_w = CDE_DOcplex_phase1(sigma, beta, A, b, p, k, factor, _lambda_scaled)
    if type(initial_w) == str:
        raise Exception('Infeasible CDE phase 1')

    # phase 2, change c to var and minimize
    target_norm = np.sum(np.abs(initial_w))
    temp_w = CDE_DOcplex_phase2_with_control(sigma, beta, A, b, p, k, factor, _lambda_scaled, target_norm,
                                                 benchmark_w)
    if type(temp_w) !=str:
        return ('feasible', temp_w)
    else:
        print('infeasible phase 2 (control), return initial w')
        return ('feasible', initial_w)


def CDE_DOcplex_simulation_with_control(list_df, _lambda, benchmark_w):   #no cross-validation, for simulation only
    # Start of the strategy
    data = list_df[0]
    position_nan = data.isna().any().values
    portfolio = np.zeros(data.shape[1])
    data = data[data.columns[position_nan == False]]

    # scale each dataset such that lambda_max is always 1
    sigma, beta, p, A, b, k = estimate_para(data)
    factor = 1 / sigma.diagonal().min()
    lambda_max, original_factor = find_lambda_max_cplex(sigma, beta, A, b, p, k, factor)
    # original_factor scales lambda_max to 1
    lambda_max = 1

    flag, w = CDE_DOcplex_with_control(sigma, beta, A, b, p, k, original_factor, _lambda, benchmark_w)

    portfolio[position_nan == False] = w

    return portfolio



if __name__ == '__main__':
    p = 100
    model = '1'
    result_dic = {}
    portfolio_dic = {}
    lambda_list = [i / 20 for i in range(1, 21)]
    for _lambda in lambda_list:
        portfolio_dic[_lambda] = []
    for index, _lambda in enumerate(lambda_list):
        print(f'Optimizing {index}:{_lambda}')
        return_list = np.empty(100)
        for counter in range(100):
            data = pd.read_csv(f'simulation datasets/{model} simulation data 127x{p} {counter + 1}.csv',
                               index_col='Date',
                               parse_dates=True)
            if index == 0:
                benchmark_w = 1 / p * np.ones(p)
            else:
                benchmark_w = portfolio_dic[lambda_list[index - 1]][counter]
            w = CDE_DOcplex_simulation_with_control([data.iloc[:-1]], _lambda, benchmark_w)
            portfolio_dic[_lambda].append(w)
            return_list[counter] = data.iloc[-1].values @ w
            print(f'finished data {counter + 1}')
        result_dic[_lambda] = return_list
    result = pd.DataFrame.from_dict(result_dic, orient='index')
    result.to_csv(f'simulation datasets/results/{model} simulation data 127x{p}_extended_CDE_fixed_lambda_l1_control_b=100_default_tolerance_results.csv')
    portfolios=pd.DataFrame.from_dict(portfolio_dic,orient='index')
    portfolios.to_csv(f'simulation datasets/results/{model} simulation data 127x{p}_extended_CDE_fixed_lambda_l1_control_b=100_default_tolerance_portfolios.csv')