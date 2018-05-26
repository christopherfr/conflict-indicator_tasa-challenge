from feature_extraction import get_features
import numpy as np
import random

# Global variables
df = 0
n_speakers = 0

def find_best_table(pop,pop_ci):
    best_ci = np.min(pop_ci)
    best_ci_index = np.where(pop_ci == best_ci)[0][0]
    best_table = pop[best_ci_index]
    print_table(best_table, best_ci)

def print_table(table, ci):
    string = '('
    for speaker in table:
        string += df['a'].cat.categories[speaker] + ','
    string = string[:-1] + '): ' + str(ci)
    print(string)

def right(pos):
    # If pos is the last index, then return the first one (0)
    if pos == n_speakers-1:
        return 0
    else:
        return pos + 1

def left(pos):
    # If pos is the first index (0), then return the last one
    if pos == 0:
        return n_speakers-1
    else:
        return pos - 1

def population_ci(pop):
    pop_ci = np.array([], dtype = np.int16)
    for table in pop:
        pop_ci = np.append(pop_ci,table_ci(table))

    return pop_ci

def table_ci(table):
    ci = 0
    # Sum the CI of every speaker in the table
    for speaker in range(len(table)):
        ci_speaker_to_right = ci_from_a_to_b(table[speaker],table[right(speaker)])
        ci_speaker_to_left = ci_from_a_to_b(table[speaker],table[left(speaker)])
        ci_speaker = ci_speaker_to_right + ci_speaker_to_left
        ci = ci + ci_speaker

    return ci

def ci_from_a_to_b(a_speaker,b_speaker):
    # Get the boolean array where the a_speaker is found in the a column
    a_found = (df['a'] == df['a'].cat.categories[a_speaker])
    df_a = df[a_found]
    # Get the boolean array where the b_speaker is found in the b column
    b_found = (df_a['b'] == df_a['b'].cat.categories[b_speaker])
    df_ab = df_a[b_found]
    # Match both boolean arrays in the DataFrame and get the CI
    return df_ab['ci'].iloc[0]


def adjacency_matrix(table):
    lefts = np.array([], dtype=np.uint8)
    rights = np.array([], dtype=np.uint8)

    for speaker in range(len(table)):
        index_of_speaker = np.where(table == speaker)[0][0]
        lefts = np.append(lefts,table[left(index_of_speaker)])
        rights = np.append(rights,table[right(index_of_speaker)])

    return np.transpose(np.vstack((lefts,rights)))

def crossover(father,mother):
    # Edge Recombination Crossover Operator https://en.wikipedia.org/wiki/Edge_recombination_operator
    father_am = adjacency_matrix(father)
    mother_am = adjacency_matrix(mother)
    union_am = np.hstack((father_am,mother_am))
    unique_am = []
    # Conver to list since each row will surely have different lengths
    for i in range(union_am.shape[0]):
        unique_am.append(np.unique(union_am[i]).tolist())
    # Choose a random parent to set the first node of the child
    random_parent = random.randint(0,1)
    node = [father[0],mother[0]][random_parent]
    child = np.array([node])
    while child.shape[0] < n_speakers:
        min_length = n_speakers
        # The list of indexes in unique_am that matches the min_length
        min_adjacents = []
        # Remove node added to son on every unique_am row (list)
        index = -1
        for nodes_list in unique_am:
            index += 1
            if node in nodes_list:
                nodes_list.remove(node)
                # We also use this for loop to get the min_length between the nodes_lists adjacent to node
                if len(nodes_list) < min_length and index not in child:
                    min_length = len(nodes_list)
        # Append the index of every node_adjacents list that matches the min_length so we can later choose one of them randomly
        for node_adjacents in unique_am[node]:
            if len(unique_am[node_adjacents]) == min_length:
                min_adjacents.append(node_adjacents)
        # Select one of the node_adjacents lists with minimum length
        if len(min_adjacents) > 0:
            node = min_adjacents[random.randint(0,len(min_adjacents)-1)]
        else:
            # This should always return the last node pending to be added to child
            node = unique_am[node][0]
        child = np.append(child,node)
    return child

def mutate(table):
    # Randomly select one speaker
    speaker_position = random.randint(0,n_speakers-1)
    # Randomly select the swap direction
    swap_direction = random.randint(0,1)
    # Swap two speakers
    speaker = table[speaker_position]
    if swap_direction == 0:
        table[speaker_position] = table[left(speaker_position)]
        table[left(speaker_position)] = speaker
    elif swap_direction == 1:
        table[speaker_position] = table[right(speaker_position)]
        table[right(speaker_position)] = speaker

    return table

def create_initial_population(POPULATION):
    for i in range(POPULATION):
        if i == 0:
            pop = np.random.choice(n_speakers, n_speakers, replace = False)
        else:
            pop = np.vstack((pop,np.random.choice(n_speakers, n_speakers, replace = False)))

    return pop

def main():
    POPULATION = 50
    MAX_ITER = 15
    MUTATION_PROB = 0.1

    global df
    df = get_features()
    global n_speakers
    n_speakers = df['a'].cat.categories.size

    # Initial population
    pop = create_initial_population(POPULATION)
    pop_ci = population_ci(pop)
    print('-------------------------\tInitial population\t-------------------------')
    find_best_table(pop,pop_ci)

    for iteration in range(MAX_ITER):
        print('-------------------------\tIteration ' + str(iteration+1) + ' of ' + str(MAX_ITER) + '\t-------------------------')
        # Initialization
        crossover_bag = np.zeros((1,8), dtype = np.int16)
        # Compute fitness for each table in the population
        pop_fitness = pop_ci * -1
        leveler = np.min(pop_fitness)
        pop_fitness = pop_fitness - leveler
        total_fitness = np.sum(pop_fitness)
        # Generate the crossover_bag
        for i in range(POPULATION):
            perc = pop_fitness[i]/total_fitness
            n = int(round(perc*POPULATION))
            for j in range(n):
                crossover_bag = np.vstack((crossover_bag,pop[i]))
        crossover_bag = crossover_bag[1:]
        # Crossover
        for i in range(POPULATION):
            # Randomly select 2 parents (tables)
            couple = random.sample(range(crossover_bag.shape[0]),2)
            father = crossover_bag[couple[0]]
            mother = crossover_bag[couple[1]]
            child = crossover(father,mother)
            # Mutation
            if random.random() <= MUTATION_PROB:
                child = mutate(child)
            if i == 0:
                new_pop = child
            else:
                new_pop = np.vstack((new_pop,child))
        pop = new_pop
        # Compute CI for each table in the new population
        pop_ci = population_ci(pop)
        find_best_table(pop,pop_ci)

if __name__ == '__main__':
    main()