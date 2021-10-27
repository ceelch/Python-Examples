import pygad
import numpy

from sklearn.metrics import mean_squared_error
#%%
function_inputs = [4,-2,3.5,5]
desired_output = 75
#%%
def fitness_func(solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    # fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    fitness = 1.0 / (numpy.square(output - desired_output) + 0.000001)
    
    return fitness
#%%
ga_instance = pygad.GA(num_generations=100,
                       sol_per_pop=5,
                       num_genes=4,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       mutation_type="random",
                       mutation_probability=0.6)
#%%
ga_instance.run()
#%%
ga_instance.plot_result()
#%%
valSolucion=ga_instance.best_solution()

print(numpy.sum(valSolucion[0]*function_inputs))