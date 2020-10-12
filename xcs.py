import numpy
import numpy.random
import itertools
from copy import deepcopy
from parameters import Parameters
from classifier import Classifier

"""
The main XCS class
"""

class XCS:
    """
        Initializes an instance of the X classifier system
        @param parameters - A parameters instance (See parameters.py), containing the parameters for this system
        @param state_function - A function which returns the current state of the system, as a string
        @param reward_function - A function which takes a state and an action, performs the action and returns the reward
        @param eop_function - A function which returns whether the state is at the end of the problem
    """
    def __init__(self, parameters, state_function, reward_function, eop_function):
        self.parameters = parameters
        self.state_function = state_function
        self.reward_function = reward_function
        self.eop_function = eop_function
        self.population = []
        self.time_stamp = 0

        self.previous_action_set = None #[A]_{-1}
        self.previous_reward = 0        #rho_{-1}
        self.previous_state = None      #sigma_{-1}

    """
        Prints the current population to stdout
    """
    def print_population(self):
        for i in self.population:
            print(i)


    """
       Classifies the given state, returning the class
       @param state - the state to classify
    """
    def classify(self, state):
        match_set = self._generate_match_set(state)
        prediction_array = self._generate_prediction_array(match_set)
        action = numpy.argmax(prediction_array)
        return action



    """
    RUN EXPERIMENT (3.3 The main loop)
        Runs a single iteration of the learning algorithm for this XCS instance
    """
    def run_experiment(self):
        curr_state = self.state_function()  #sigma
        match_set = self._generate_match_set(curr_state)
        prediction_array = self._generate_prediction_array(match_set)
        action = self._select_action(prediction_array)
        action_set = _generate_action_set(match_set, action) 
        reward = self.reward_function(curr_state, action)

        #Update the previous set
        if self.previous_action_set:
            P = self.previous_reward + self.parameters.gamma * max(prediction_array)
            self._update_set(self.previous_action_set, P)
            self._run_ga(self.previous_action_set, self.previous_state)

        if self.eop_function():
            self._update_set(action_set, reward)
            self._run_ga(action_set, curr_state)
            self.previous_action_set = None
        else:
            self.previous_action_set = action_set
            self.previous_reward = reward
            self.previous_state = curr_state
        self.time_stamp = self.time_stamp + 1

  
    """
    GENERATE MATCH SET (3.4 Formation of the match set)
        Generates the match set for the given state, covering as necessary
        @param state - the state to generate a match set 
    """
    def _generate_match_set(self, state):
        set_m = []
        while len(set_m) == 0:
            set_m = [clas for clas in self.population if does_match(clas.condition, state)]
            if len(set_m) < self.parameters.theta_mna:#Cover
                clas = self._generate_covering_classifier(state, set_m)
                self._insert_in_population(clas)
                self._delete_from_population()
                set_m = []
        return set_m

    """
    GENERATE COVERING CLASSIFIER (3.4 Formation of the match set ~Covering~)
        Generates a classifier that conforms to the given state, and has an unused action from
        the given match set
        @param state - The state to make the classifier conform to
        @param match_set - The set of current matches
    """
    def _generate_covering_classifier(self, state, match_set):
        clas = Classifier(self.parameters, state)
        used_actions = [classifier.action for classifier in match_set]
        available_actions = list(set(range(self.parameters.num_actions)) - set(used_actions))
        clas.action = numpy.random.choice(available_actions)
        clas.time_stamp = self.time_stamp
        return clas

    """
    GENERATE PREDICTION ARRAY (3.5 The prediction array)
        Generates a prediction array for the given match set
        @param match_set - The match set to generate predictions
    """
    def _generate_prediction_array(self, match_set):
        PA = [0] * self.parameters.num_actions
        FSA = [0] * self.parameters.num_actions
        for clas in match_set:
            PA[clas.action] += clas.prediction * clas.fitness
            FSA[clas.action] += clas.fitness

        normal = [PA[i] if FSA[i] == 0 else PA[i]/FSA[i] for i in range(self.parameters.num_actions)]

        return normal

    """
    SELECT ACTION (3.6 Choosing an action)
        Selects the action to run from the given prediction array. Takes into account exploration
        vs exploitation
        @param prediction_array - The prediction array to generate an action from
    """
    def _select_action(self, prediction_array):
        valid_actions = [action for action in range(self.parameters.num_actions) if prediction_array[action] != 0]
        if len(valid_actions) == 0:
            return numpy.random.randint(0, self.parameters.num_actions)

        if numpy.random.rand() < self.parameters.p_explr:
            return numpy.random.choice(valid_actions)
        else:
            return numpy.argmax(prediction_array)

    """
    UPDATE SET (3.8 Updating classifier parameters)
       Updates the given action set's prediction, error, average size and fitness using the given decayed performance
       @param action_set - The set to update
       @param P - The reward to use
    """
    def _update_set(self, action_set, P):
        set_numerosity = sum([clas.numerosity for clas in action_set])
        for clas in action_set:
            clas.experience = clas.experience + 1
            if clas.experience < 1. / self.parameters.beta:
                clas.prediction = clas.prediction + (P - clas.prediction) / clas.experience
                clas.error = clas.error + (abs(P - clas.prediction) - clas.error) / clas.experience
                clas.average_size = clas.average_size + (set_numerosity - clas.numerosity) / clas.experience
            else:
                clas.prediction = clas.prediction + (P - clas.prediction) * self.parameters.beta
                clas.error = clas.error + (abs(P - clas.prediction) - clas.error) * self.parameters.beta
                clas.average_size = clas.average_size + (set_numerosity - clas.numerosity) * self.parameters.beta

        self._update_fitness(action_set)

        if self.parameters.do_action_set_subsumption:
            self._do_action_set_subsumption(action_set)

    """
    UPDATE FITNESS (3.8 Updating classifier parameters ~Fitness update~)
        Updates the given action set's fitness
        @param action_set - The set to update
    """
    def _update_fitness(self, action_set):
        #UPDATE FITNESS in set [A]
        kappa = {clas: 1 if clas.error < self.parameters.e0 else self.parameters.alpha * (clas.error / self.parameters.e0) ** -self.parameters.nu for clas in action_set}
        accuracy_sum = sum([kappa[clas] * clas.numerosity for clas in action_set])
        for clas in action_set:
            clas.fitness = clas.fitness + self.parameters.beta * (kappa[clas] * clas.numerosity / accuracy_sum - clas.fitness)

    """
    RUN GA (3.9 The genetic algorithm in XCS)
        Runs the genetic algorithm on the given set, generating two new classifers
        to be inserted into the population
        @param action_set - the action set to choose parents from
        @param state - The state mutate with
    """
    def _run_ga(self, action_set, state):
        if len(action_set) == 0:
            return

        if self.time_stamp - numpy.average([clas.time_stamp for clas in action_set], weights=[clas.numerosity for clas in action_set]) > self.parameters.theta_GA:
            for clas in action_set:
                clas.time_stamp = self.time_stamp
           
            parent_1 = self._select_offspring(action_set, selection_method = 'Tournament')
            parent_2 = self._select_offspring(action_set, selection_method = 'Tournament')
            child_1 = deepcopy(parent_1)
            child_2 = deepcopy(parent_2)
            child_1.id = Classifier.global_id
            child_2.id = Classifier.global_id + 1
            Classifier.global_id = Classifier.global_id + 2
            child_1.numerosity = 1
            child_2.numerosity = 1
            child_1.experience = 0
            child_2.experience = 0

            if numpy.random.rand() < self.parameters.chi:
                _apply_crossover(child_1, child_2)
                child_1.prediction = child_2.prediction = numpy.average([parent_1.prediction, parent_2.prediction])
                child_1.error = child_2.error = numpy.average([parent_1.error, parent_2.error])
                child_1.fitness = child_2.fitness = numpy.average([parent_1.fitness, parent_2.fitness])

            child_1.fitness = child_1.fitness * 0.1
            child_2.fitness = child_2.fitness * 0.1

            for child in [child_1, child_2]:
                child._apply_mutation(state, self.parameters.mu, self.parameters.num_actions)
                if self.parameters.do_GA_subsumption == True:
                    if parent_1._does_subsume(child, self.parameters.theta_sub, self.parameters.e0):
                        parent_1.numerosity = parent_1.numerosity + 1
                    elif parent_2._does_subsume(child, self.parameters.theta_sub, self.parameters.e0):
                        parent_2.numerosity = parent_2.numerosity + 1
                    else:
                        self._insert_in_population(child)
                else:
                    self._insert_in_population(child)

                self._delete_from_population()

    """
    SELECT OFFSPRING (3.9 The genetic algorithm in XCS ~Roulette-wheel selection~)
        Makes parent selection in GA by Roulette-wheel selection
        (Tournament selection is also supported.)
        @param action_set - the set to run GA
    """
    def _select_offspring(self, action_set, selection_method = 'Roulette'):

        if selection_method == 'Roulette':
            fitness_sum = sum([clas.fitness for clas in action_set])
            choice_point = numpy.random.rand()

            fitness_sum = 0.
            for clas in action_set:
                fitness_sum = fitness_sum + clas.fitness
                if fitness_sum > choice_point:
                    break
            return clas

        elif selection_method == 'Tournament':
            parent = None
            for clas in action_set:
                if parent == None or parent.fitness / parent.numerosity < clas.fitness / clas.numerosity:
                    for i in range(1, clas.numerosity):
                        if numpy.random.rand() < self.parameters.tau:
                            parent = clas
                            break
            if parent == None:
                parent = numpy.random.choice(action_set)
            return parent

        else:
            return

    """
    INSERT IN POPULATION (3.10 Insertion in the population)
        Inserts the given classifier into the population, if it isn't able to be
        subsumed by some other classifier in the population
        @param clas - the classifier to insert
    """
    def _insert_in_population(self, clas):
        same = [c for c in self.population if (c.action, c.condition) == (clas.action, clas.condition)]
        if same:
            same[0].numerosity = same[0].numerosity + 1
            return
        self.population.append(clas)

    """
    DELETE FROM POPULATION (3.11 Deletion from the population ~Roulette-wheel deletion~)
        Deletes a classifier from the population, if necessary
    """
    def _delete_from_population(self):
        numerosity_sum = sum([clas.numerosity for clas in self.population])
        if numerosity_sum <= self.parameters.N:
            return

        average_fitness = sum([clas.fitness for clas in self.population]) / numerosity_sum
        vote_sum = 0.
        
        for clas in self.population:
            vote_sum = vote_sum + clas._deletion_vote(average_fitness, self.parameters.theta_del, self.parameters.delta)

        choice_point = numpy.random.rand() * vote_sum
        vote_sum = 0.

        for clas in self.population:
            vote_sum = vote_sum + clas._deletion_vote(average_fitness, self.parameters.theta_del, self.parameters.delta)
            if(clas.numerosity > choice_point):
                if clas.numerosity > 1:
                    clas.numerosity -= 1
                else:
                    self.population.remove(clas)
                return

    """
    DO ACTION SET SUBSUMPTION (3.12 Subsumption)
        Does subsumption inside the action set, finding the most general classifier
        and merging things into it
        @param action_set - the set to perform subsumption on
    """
    def _do_action_set_subsumption(self, action_set):
        cl = None
        for clas in action_set:
            if clas._could_subsume(self.parameters.theta_sub, self.parameters.e0):
                if cl == None or len([i for i in clas.condition if i == '#']) > len([i for i in cl.condition if i == '#']) or numpy.random.rand() > 0.5:
                    cl = clas

        if cl:
            for clas in action_set:
                if cl._is_more_general(clas):
                    cl.numerosity = cl.numerosity + clas.numerosity
                    action_set.remove(clas)
                    self.population.remove(clas)




"""
DOES MATCH (3.4 Formation of the match set)
    Returns whether the given state matches the given condition
    @param condition - The condition to match against
    @param state - The state to match against
"""
def does_match(condition, state):
    return all([c == '#' or c == s for c, s in zip(condition, state)])

"""
GENERATE ACTION SET (3.7 Formation of the action set)
    Forms action set from match set
    @param match_set - The match set for generating action set
"""
def _generate_action_set(match_set, action):
    return [clas for clas in match_set if clas.action == action]


"""
APPLY CROSSOVER (3.9 The genetic algorithm in XCS ~Crossover~)
    Cross's over the given children, modifying their conditions
    @param child_1 - The first child to crossover
    @param child_2 - The second child to crossover
"""
def _apply_crossover(child_1, child_2):
    x = numpy.random.randint(0, len(child_1.condition))
    y = numpy.random.randint(0, len(child_1.condition))

    if x > y:
        x, y = y, x
    i = 0

    child_1_condition = list(child_1.condition)
    child_2_condition = list(child_2.condition)

    while True:
        if x <= i < y:
            child_1_condition[i], child_2_condition[i] = child_2_condition[i], child_1_condition[i]
        i += 1
        if i >= y:
            break

    child_1.condition = ''.join(child_1_condition)
    child_2.condition = ''.join(child_2_condition)