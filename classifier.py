import numpy
import numpy.random

"""
A classifier in XCS
"""


class Classifier:
    """
        Initializes an instance of Classifier
        @param parameters - A parameters instance (See parameters.py), containing the parameters for this classifier
        @param state - The state of the system to generate classifier
    """
    global_id = 0  # A Globally unique identifier

    def __init__(self, parameters, state=None):
        self.id = Classifier.global_id
        Classifier.global_id = Classifier.global_id + 1
        self.action = numpy.random.randint(0, parameters.num_actions)
        self.prediction = parameters.p_I
        self.error = parameters.e_I
        self.fitness = parameters.F_I
        self.experience = 0
        self.time_stamp = 0
        self.action_set_size = 1
        self.numerosity = 1
        self.kappa = 1

        if state is None:
            self.condition = ''.join(
                ['#' if numpy.random.rand() < parameters.p_hash else '0' if numpy.random.rand() > 0.5 else '1' for i in
                 [0] * parameters.state_length])
        else:
            # Generate the condition from the state (if supplied)
            self.condition = ''.join(
                ['#' if numpy.random.rand() < parameters.p_hash else state[i] for i in range(parameters.state_length)])

    def __str__(self):
        return "Classifier " + str(self.id) + ": " + self.condition + " = " + str(self.action) + " Fitness: " + str(
            self.fitness) + " Prediction: " + str(self.prediction) + " Error: " + str(
            self.error) + " Experience: " + str(self.experience)

    """
    APPLY MUTATION (3.9 The genetic algorithm in XCS ~Mutation~)
       Mutates this classifier, changing the condition and action
       @param state - The state of the system to mutate around
       @param mu - The probability with which to mutate
       @param num_actions - The number of actions in the system
    """

    def _apply_mutation(self, state, mu, num_actions):
        self.condition = ''.join(
            [self.condition[i] if numpy.random.rand() > mu else state[i] if self.condition[i] == '#' else '#' for i in
             range(len(self.condition))])

        if numpy.random.rand() < mu:
            used_actions = [self.action]
            available_actions = list(set(range(num_actions)) - set(used_actions))
            self.action = numpy.random.choice(available_actions)

    """
    DELETION VOTE (3.11 Deletion from the population ~The deletion vote~)
       Calculates the deletion vote for this classifier, that is, how much it thinks it should be deleted
       @param average_fitness - The average fitness in the current action set
       @param theta_del - See parameters.py
       @param delta - See parameters.py
    """

    def _deletion_vote(self, average_fitness, theta_del, delta):
        vote = self.action_set_size * self.numerosity
        if self.experience > theta_del and self.fitness / self.numerosity < delta * average_fitness:
            vote = vote * average_fitness / (self.fitness / self.numerosity)

        return vote

    """
    COULD SUBSUME (3.12 Subsumption ~Subsumption of a classifier~)
        Returns whether this classifier can subsume others
        @param theta_sub - See parameters.py
        @param e0 - See parameters.py
    """

    def _could_subsume(self, theta_sub, e0):
        return self.experience > theta_sub and self.error < e0

    """
    IS MORE GENERAL (3.12 Subsumption ~Subsumption of a classifier~)
        Returns whether this classifier is more general than another
        @param spec - the classifier to check against
    """

    def _is_more_general(self, spec):
        if len([i for i in self.condition if i == '#']) <= len([i for i in spec.condition if i == '#']):
            return False

        i = 0
        while True:
            if self.condition[i] != '#' and self.condition[i] != spec.condition[i]:
                return False
            i += 1
            if i >= len(self.condition):
                break
        return True

    """
    DOES SUBSUME (3.12 Subsumption ~Subsumption of a classifier~)
        Returns whether this classifier subsumes another
        @param tos - the classifier to check against
        @param theta_sub - See parameters.py
        @param e0 - See parameters.py
    """

    def _does_subsume(self, tos, theta_sub, e0):
        return self.action == tos.action and self._could_subsume(theta_sub, e0) and self._is_more_general(tos)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if other is None:
            return False
        return self.id == other.id
