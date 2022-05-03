# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

from captureAgents import CaptureAgent
import random, util
import game
from game import Directions
from util import nearestPoint
import json

#################
# Team creation #
#################
arguments = {}
training = False


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent', **args):
    if 'numTraining' in args:
        arguments['numTraining'] = args['numTraining']
        global_vars = globals()
        global_vars['training'] = True
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)


class PacAttack(CaptureAgent):
    def assignAgents(self, gameState):
        """
    will look at game state and assign agents to be offensive or defensive

    possibly send some sort of value to determine how offensive (this should
    be over-rideable when specific scenarios occur, like them losing an agent,
    when an agent might otherwise be trapped, etc.)
    """

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        return features * self.weights


# AKA Aaron Pac-dgers
class OffensiveAgent(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.start = None
        self.riskyPositions = None
        self.width = None
        self.height = None
        self.risky_food = None
        self.boundary = None
        self.isRed = None
        weights = {}
        with open("./offense_weights.json", 'r') as openfile:
            weights = json.load(openfile)
        self.weights = weights
        # Chance to explore
        self.epsilon = 0.05
        # Learning Rate
        self.alpha = 0.2
        # For monte carlo
        self.depth = 5
        self.decay = 0.8

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        boundary_coordinate = None
        if gameState.isOnRedTeam(self.index):
            self.isRed = True
            boundary_coordinate = (gameState.data.layout.width - 2) // 2
        else:
            self.isRed = False
            boundary_coordinate = ((gameState.data.layout.width - 2) // 2) + 1
        self.boundary = []
        for y_coord in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary_coordinate, y_coord):
                self.boundary.append((boundary_coordinate, y_coord))
        self.risky_food = self.findRiskyAttackingFood(gameState)
        CaptureAgent.registerInitialState(self, gameState)

    # Q-Learning Functions ----------------------------------------------------------------------

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getActionQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        # for feature in features:
        #     print(feature + ": ", features[feature], " * ", self.weights[feature], "= " + str(features[feature] * self.weights[feature]))
        value = features * self.weights
        print("offense: ", value)
        return value

    def getQValue(self, gameState):
        features = self.getFeatures(gameState, Directions.STOP)
        value = features * self.weights
        return value

    def getMaxQValue(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return 0.0
        action_from_policy = self.getPolicy(gameState)
        return self.getActionQValue(gameState, action_from_policy)

    def getPolicy(self, gameState):
        values_and_actions = dict()
        legal_actions = gameState.getLegalActions(self.index)
        legal_actions.remove(Directions.STOP)
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if len(legal_actions) == 1:
            return reverse
        elif reverse in legal_actions:
            legal_actions.remove(reverse)
        for action in legal_actions:
            value = self.getActionQValue(gameState, action)
            values_and_actions[value] = action
        max_value = max(values_and_actions.keys())
        return values_and_actions[max_value]

    def chooseAction(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        if not legal_actions:
            return None

        if training:
            for action in legal_actions:
                self.updateWeights(gameState, action)

        action = None
        probability = util.flipCoin(self.epsilon)
        if probability:
            action = random.choice(legal_actions)
            # if self.closeToRewards(gameState):
            #     # Highest reward from monte carlo
            #     action = self.monteCarloSimulation(gameState, self.depth, self.decay)
            # else:
            #     action = random.choice(legal_actions)
        else:
            action = self.getPolicy(gameState)
        return action

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        current_position = gameState.getAgentState(self.index).getPosition()
        current_position = (int(current_position[0]), int(current_position[1]))
        next_position = successor.getAgentState(self.index).getPosition()
        next_position = (int(next_position[0]), int(next_position[1]))
        next_agent_state = successor.getAgentState(self.index)

        # Score of successor state
        score = successor.getScore()
        if not self.isRed:
            score *= -1
        features['score'] = score

        if next_agent_state.isPacman:
            features['onDefense'] = 0
        else:
            features['onDefense'] = 1
        food = self.getFood(gameState)
        if food[next_position[0]][next_position[1]]:
            features['eatsFood'] = 1
        else:
            features['eatsFood'] = 0

        # Distance to the nearest food
        closest_food = self.closestFood(gameState)
        if closest_food is not None:
            features['closestFood'] = closest_food

        # Distance to the nearest capsule
        closest_capsule = self.closestCapsule(gameState)
        features['closestCapsule'] = closest_capsule

        # Distance One/Five Ghosts
        features['distanceFiveGhostCount'] = 0
        features['distanceOneGhostCount'] = 0
        ghost_states = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
        ghost_positions = [ghost.getPosition() for ghost in ghost_states if ghost.isPacman and ghost.getPosition() is not None]
        for ghost_position in ghost_positions:
            distance_to_ghost = self.getMazeDistance(next_position, ghost_position)
            if distance_to_ghost <= 5:
                features['distanceFiveGhostCount'] += 1
            if distance_to_ghost <= 1:
                features['distanceOneGhostCount'] += 1


        # Distance to friendly agent
        teammate_index = None
        team_indices = self.getTeam(gameState)
        for index in team_indices:
            if index != self.index:
                teammate_index = index
        teammate_position = gameState.getAgentPosition(teammate_index)
        if teammate_position is not None:
            features['teammateDistance'] = self.getMazeDistance(current_position, teammate_position)

        # Distance to the closest risky food
        closest_risky_food = sys.maxint
        self.risky_food = self.updateRiskyFoodGrid(gameState)
        for position_risk_score in self.risky_food:
            distance = self.getMazeDistance(current_position, position_risk_score[0])
            if distance < closest_risky_food:
                closest_risky_food = distance
                features['closestRiskyFood'] = closest_risky_food

        # Normalize feature values to be between [0, 1]
        # features = self.normalize(features)
        # features = util.Counter(features)
        features.divideAll(30)
        return features

    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)
        reward = self.getStateRewardValue(gameState, nextState)

        old_q_value = self.getActionQValue(gameState, action)
        new_q_value = self.getMaxQValue(nextState)
        weight_correction = reward + self.decay * (new_q_value - old_q_value)
        for feature in features:
            new_weight = self.weights[feature] + self.alpha * weight_correction * features[feature]
            self.weights[feature] = new_weight

    def getStateRewardValue(self, gameState, nextState):
        reward = 0
        current_position = gameState.getAgentPosition(self.index)
        next_position = nextState.getAgentState(self.index).getPosition()

        opponents = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
        non_hidden_opponents = [opponent for opponent in opponents if
                                opponent.getPosition() is not None and not opponent.isPacman]
        if len(non_hidden_opponents) != 0:
            closest_opponent = min(
                [self.getMazeDistance(current_position, opponent.getPosition()) for opponent in non_hidden_opponents])

            # Decrease reward if Pacman gets eaten in nextState
            if closest_opponent <= 1 and next_position == self.start:
                reward -= 15

            # Increase reward if Pacman gets a risky food and can make it out of the tunnel
            for position_risk_score in self.risky_food:
                # Our next position has a risky food
                if next_position == position_risk_score[0]:
                    # We need to check if the risk score is lower than the closest pacman + 1
                    risk_of_grabbing_food = position_risk_score[1] + 1
                    if risk_of_grabbing_food < closest_opponent:
                        reward += 2.5

        return reward

    def final(self, gameState):
        action = game.Directions.STOP
        if training:
            self.updateWeights(gameState, action)
        json_object = json.dumps(self.weights, indent=4)
        print("Offense Final")
        with open("./offense_weights.json", "w") as outfile:
            outfile.write(json_object)

    # Helper Functions -------------------------------------------------------------------------

    def normalize(self, d):
        lower = -1
        higher = 1
        return {key: lower + (higher - lower) * value for key, value in d.iteritems() if value is not None}

    def isBlue(self):
        return not self.isRed

    def getPosition(self, gameState):
        return gameState.getAgentState(self.index).getPosition()

    def opponentPositions(self, gameState):
        return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

    def getLegalActionsFromPosition(self, position, gameState):
        walls = gameState.getWalls()
        x = int(position[0])
        y = int(position[1])
        legal_actions = [game.Directions.STOP]
        out_right = x + 1 >= self.width
        out_left = x - 1 < 1
        out_down = y - 1 < 1
        out_up = y + 1 >= self.height

        if not out_left and not walls[x - 1][y]:
            legal_actions.append(game.Directions.WEST)
        if not out_right and not walls[x + 1][y]:
            legal_actions.append(game.Directions.EAST)
        if not out_up and not walls[x][y + 1]:
            legal_actions.append(game.Directions.NORTH)
        if not out_down and not walls[x][y - 1]:
            legal_actions.append(game.Directions.SOUTH)
        return legal_actions

    def findCornerPositions(self, gameState):
        corner_positions = []
        if self.isRed:
            start_column = self.width // 2 + 1
            end_column = self.width

        else:
            start_column = 1
            end_column = self.width // 2
        for x in range(start_column, end_column):
            for y in range(1, gameState.data.layout.height):
                legal_actions = self.getLegalActionsFromPosition((x, y), gameState)
                legal_actions.remove(game.Directions.STOP)
                if len(legal_actions) == 1:
                    corner_positions.append((x, y))
        return corner_positions

    def calculateRiskScore(self, gameState, position, previousAction):
        position_variable = position
        previous_action_variable = previousAction
        risk_score = 0
        visited = set(position_variable)
        while True:
            legal_actions = self.getLegalActionsFromPosition(position_variable, gameState)
            if previous_action_variable in legal_actions:
                legal_actions.remove(previous_action_variable)
            legal_actions.remove(game.Directions.STOP)
            if len(legal_actions) > 1:
                return risk_score
            else:
                if legal_actions[0] == game.Directions.WEST:
                    position_variable = (position_variable[0] - 1, position_variable[1])
                elif legal_actions[0] == game.Directions.EAST:
                    position_variable = (position_variable[0] + 1, position_variable[1])
                elif legal_actions[0] == game.Directions.NORTH:
                    position_variable = (position_variable[0], position_variable[1] + 1)
                elif legal_actions[0] == game.Directions.SOUTH:
                    position_variable = (position_variable[0], position_variable[1] - 1)
                risk_score += 1
                previous_action_variable = legal_actions[0]
                if position_variable in visited:
                    return risk_score
                visited.add(position_variable)
        return risk_score

    def findRiskyPositions(self, gameState):
        """
        Returns a list of tuples (position, riskScore) where position is the
        position of the food, and riskScore is how many steps it takes from that
        position to be able to move in more than one direction
        """
        risky_positions = []
        corners = self.findCornerPositions(gameState)
        for position in corners:
            legal_actions = self.getLegalActionsFromPosition(position, gameState)
            risk_score = 1
            found_exit = False
            new_posn = position
            path_positions = dict()
            while not found_exit:
                legal_actions.remove(game.Directions.STOP)
                if len(legal_actions) > 1:
                    break
                if legal_actions[0] == game.Directions.WEST:
                    new_posn = (new_posn[0] - 1, new_posn[1])
                elif legal_actions[0] == game.Directions.EAST:
                    new_posn = (new_posn[0] + 1, new_posn[1])
                elif legal_actions[0] == game.Directions.NORTH:
                    new_posn = (new_posn[0], new_posn[1] + 1)
                elif legal_actions[0] == game.Directions.SOUTH:
                    new_posn = (new_posn[0], new_posn[1] - 1)
                if len(self.getLegalActionsFromPosition(new_posn, gameState)) > 2:
                    found_exit = True
                else:
                    risk_score += 1
                    legal_actions = self.getLegalActionsFromPosition(new_posn, gameState)
                    for p in path_positions.keys():
                        path_positions[p] += 1
                    path_positions[new_posn] = 1
            for p in path_positions.keys():
                risky_positions.append((p, path_positions[p]))
        return risky_positions

    def findRiskyAttackingFood(self, gameState):
        """
        Returns a list of positions with risky food
        """
        risky_food = []
        risky_positions = self.findRiskyPositions(gameState)
        for position_risk_score in risky_positions:
            position = position_risk_score[0]
            if not self.isRed:
                red_food = gameState.getRedFood()
                if red_food[position[0]][position[1]]:
                    risky_food.append(position_risk_score)
            else:
                blue_food = gameState.getBlueFood()
                if blue_food[position[0]][position[1]]:
                    risky_food.append(position_risk_score)
        return risky_food

    def updateRiskyFoodGrid(self, gameState):
        new_risky_food = []

        for position_risk_score in self.risky_food:
            position = position_risk_score[0]
            if self.isBlue():
                red_food = gameState.getRedFood()
                if red_food[position[0]][position[1]]:
                    new_risky_food.append(position_risk_score)
            else:
                blue_food = gameState.getBlueFood()
                if blue_food[position[0]][position[1]]:
                    new_risky_food.append(position_risk_score)
        return new_risky_food

    def closePositions(self, gameState, position, stepsToExplore, visitedPositions):
        """
        Returns a set of positions that can be reached within stepsToExplore steps
        """
        if stepsToExplore == 0:
            return
        legal_actions = self.getLegalActionsFromPosition(position, gameState)
        legal_actions.remove(game.Directions.STOP)
        for action in legal_actions:
            new_position = None
            if action == game.Directions.WEST:
                new_position = (position[0] - 1, position[1])
            if action == game.Directions.EAST:
                new_position = (position[0] + 1, position[1])
            if action == game.Directions.NORTH:
                new_position = (position[0], position[1] + 1)
            if action == game.Directions.SOUTH:
                new_position = (position[0], position[1] - 1)
            if new_position not in visitedPositions:
                visitedPositions.add(new_position)
                self.closePositions(gameState, new_position, stepsToExplore - 1, visitedPositions)
        return visitedPositions

    def closeToRewards(self, gameState):
        """
        Determines whether there are 2 or more rewards within 5 units of Pacman
        """
        current_position = self.getPosition(gameState)
        pellet_count = 0
        possible_positions = self.closePositions(gameState, current_position, 5, set())

        food = self.getFood(gameState)
        # if self.isRed:
        #     food = gameState.getBlueFood()
        # else:
        #     food = gameState.getBlueFood()
        for position in possible_positions:
            if food[int(position[0])][int(position[1])]:
                pellet_count += 1
        return pellet_count >= 1

    def montePossibleActions(self, state):
        """
        Returns the list of possible actions the Monte Carlo Simulation could take
        """
        legal_actions = state.getLegalActions(self.index)
        legal_actions.remove(Directions.STOP)
        if len(legal_actions) == 1:
            return [legal_actions[0]]
        else:
            reverse_direction = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
            if reverse_direction in legal_actions:
                legal_actions.remove(reverse_direction)
            return legal_actions

    def monteCarloScore(self, gameState, depth, decay):
        """
        Performs a Monte Carlo simulation on the given depth and returns the score maximizing aciton
        """
        copy = gameState.deepCopy()
        if depth == 0:
            new_actions = self.montePossibleActions(copy)
            new_action = random.choice(new_actions)
            next_state = copy.generateSuccessor(self.index, new_action)
            return int(self.getActionQValue(next_state, Directions.STOP))

        results = []
        new_actions = self.montePossibleActions(copy)
        for action in new_actions:
            new_state = copy.generateSuccessor(self.index, action)
            score = self.getActionQValue(new_state, Directions.STOP) + decay * self.monteCarloScore(new_state,
                                                                                                    depth - 1, decay)
            results.append(score)
        return int(max(results))

    def monteCarloSimulation(self, gameState, depth, decay):
        """
        Performs a Monte Carlo simulation on the given depth and returns the score maximizing aciton
        """
        copy = gameState.deepCopy()
        if depth == 0:
            new_actions = self.montePossibleActions(copy)
            new_action = random.choice(new_actions)
            next_state = copy.generateSuccessor(self.index, new_action)
            return self.getActionQValue(next_state, Directions.STOP)

        results = []
        actions = dict()
        new_actions = self.montePossibleActions(copy)
        for action in new_actions:
            new_state = copy.generateSuccessor(self.index, action)
            score = self.getActionQValue(new_state, Directions.STOP) + decay * self.monteCarloScore(new_state,
                                                                                                    depth - 1, decay)
            results.append(score)
            actions[score] = action
        return actions[max(results)]

    def closestFood(self, gameState):
        closest_food = sys.maxint
        not_sys_max_int = False
        food = self.getFood(gameState)
        current_position = self.getPosition(gameState)

        start = None
        end = None
        if self.isRed:
            start = self.width // 2 + 1
            end = self.width
        else:
            start = 1
            end = self.width // 2

        for x in range(start, end):
            for y in range(1, self.height):
                if food[x][y]:
                    distance = self.getMazeDistance((x, y), current_position)
                    if distance < closest_food:
                        closest_food = distance
                        not_sys_max_int = True
        if not not_sys_max_int:
            return None
        return closest_food

    def closestCapsule(self, gameState):
        current_position = gameState.getAgentPosition(self.index)
        capsules = self.getCapsules(gameState)
        if len(capsules) == 0 or capsules is None:
            return None
        return min([self.getMazeDistance(current_position, capsule) for capsule in capsules])


class DefensiveAgent(CaptureAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.start = None
        self.isRed = None
        self.risky_food = []
        self.riskyPositions = []
        self.width = None
        self.height = None
        self.boundary = []
        self.epsilon = 0.05
        self.alpha = 0.2
        self.decay = 0.8
        weights = {}
        with open("./defense_weights.json", 'r') as openfile:
            weights = json.load(openfile)
        self.weights = weights

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height

        boundary_coordinate = None
        if gameState.isOnRedTeam(self.index):
            self.isRed = True
            boundary_coordinate = (gameState.data.layout.width - 2) // 2
        else:
            self.isRed = False
            boundary_coordinate = ((gameState.data.layout.width - 2) // 2) + 1

        self.boundary = []
        for y_coord in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary_coordinate, y_coord):
                self.boundary.append((boundary_coordinate, y_coord))

        self.findRiskyPositions(gameState)
        self.risky_food = self.findRiskyDefendingFood(gameState)
        CaptureAgent.registerInitialState(self, gameState)

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        if not legal_actions:
            return None

        if training:
            for action in legal_actions:
                self.updateWeights(gameState, action)
        action = None

        probability = util.flipCoin(self.epsilon)
        actions = gameState.getLegalActions(self.index)
        if probability:
            action = random.choice(actions)
        else:
            action = self.getPolicy(gameState)
        return action

    def getPolicy(self, gameState):
        values_and_actions = dict()
        legal_actions = gameState.getLegalActions(self.index)
        legal_actions.remove(Directions.STOP)
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if len(legal_actions) == 1:
            return reverse
        elif reverse in legal_actions:
            legal_actions.remove(reverse)
        for action in legal_actions:
            value = self.getActionQValue(gameState, action)
            values_and_actions[value] = action
        max_value = max(values_and_actions.keys())
        return values_and_actions[max_value]

    def getMaxQValue(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return 0.0
        action_from_policy = self.getPolicy(gameState)
        return self.getActionQValue(gameState, action_from_policy)

    def getQValue(self, gameState):
        features = self.getFeatures(gameState, Directions.STOP)
        value = features * self.weights
        return value

    def getActionQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        value = features * self.weights
        print("defense: ", value)
        return value

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        current_position = gameState.getAgentState(self.index).getPosition()
        current_position = (int(current_position[0]), int(current_position[1]))

        their_food_list = self.getFood(successor).asList()
        our_food_list = self.getFoodYouAreDefending(successor).asList()
        score = len(our_food_list) - len(their_food_list)
        if not self.isRed:
            score *= -1
        features['score'] = score

        # Distance to the closest boundary point
        closest_boundary = sys.maxint
        for boundary_position in self.boundary:
            if boundary_position is not None and current_position is not None:
                distance = self.getMazeDistance(current_position, boundary_position)
                if distance < closest_boundary:
                    closest_boundary = distance
                    features['closestBoundary'] = closest_boundary

        # Number of enemies in Pacman mode
        pac_count = 0
        for opponent in self.getOpponents(successor):
            agent_state = successor.getAgentState(opponent)
            if agent_state.isPacman:
                pac_count += 1
        features['pacCount'] = pac_count

        # NEED TO UPDATE RISKY DEFENDING FOOD
        self.findRiskyDefendingFood(successor)
        features['riskyFoodCount'] = len(self.risky_food)

        # Sum of close risky positions' risk scores
        risk_score = 0
        nearby_tiles = self.BFSVisited(successor, current_position, set(current_position), 3)
        for pos in nearby_tiles:
            if pos in self.risky_food:
                risk_score += self.riskyPositions[pos]

        features['riskScore'] = risk_score / len(nearby_tiles)

        # Normalize feature values to be between [0, 1]
        # features = self.normalize(features)
        # features = util.Counter(features)
        features.divideAll(10)
        return features

    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)
        reward = self.getStateRewardValue(gameState, nextState)

        old_q_value = self.getActionQValue(gameState, action)
        new_q_value = self.getMaxQValue(nextState)
        weight_correction = reward + self.decay * (new_q_value - old_q_value)
        for feature in features:
            new_weight = self.weights[feature] + self.alpha * weight_correction * features[feature]
            self.weights[feature] = new_weight

    def getStateRewardValue(self, gameState, nextState):
        reward = 0
        current_position = gameState.getAgentPosition(self.index)
        next_position = nextState.getAgentState(self.index).getPosition()

        opponents = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
        non_hidden_opponents = [opponent for opponent in opponents if
                                opponent.getPosition() is not None and opponent.isPacman]
        if len(non_hidden_opponents) != 0:
            closest_opponent = min(
                [self.getMazeDistance(current_position, opponent.getPosition()) for opponent in non_hidden_opponents])

            # Increase reward if you eat someone
            opponents_next = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
            opponent_pacmans_next = [opponent for opponent in opponents_next if opponent.isPacman and opponent.getPosition() is not None]
            if len(opponent_pacmans_next) == len(non_hidden_opponents) - 1:
                reward += 3

            # Decrease reward if you get eaten by enemy
            scared_timer = gameState.getAgentState(self.index).scaredTimer
            if scared_timer > 0:
                if closest_opponent <= 1 and next_position == self.start:
                    reward -= 1.5

        return reward

    def final(self, gameState):
        action = game.Directions.STOP
        if training:
            self.updateWeights(gameState, action)
        json_object = json.dumps(self.weights, indent=4)
        print("Defense Final")
        with open("./defense_weights.json", "w") as outfile:
            outfile.write(json_object)

    ################################### HELPERS ###################################

    def normalize(self, d):
        lower = -1
        higher = 1
        return {key: lower + (higher - lower) * value for key, value in d.iteritems() if value is not None}

    def getPosition(self, gameState):
        return gameState.getAgentState(self.index).getPosition()

    def getLegalActionsFromPosition(self, position, gameState):
        walls = gameState.getWalls()
        x = int(position[0])
        y = int(position[1])
        legal_actions = [game.Directions.STOP]
        out_right = x + 1 >= self.width
        out_left = x - 1 < 1
        out_down = y - 1 < 1
        out_up = y + 1 >= self.height

        if not out_left and not walls[x - 1][y]:
            legal_actions.append(game.Directions.WEST)
        if not out_right and not walls[x + 1][y]:
            legal_actions.append(game.Directions.EAST)
        if not out_up and not walls[x][y + 1]:
            legal_actions.append(game.Directions.NORTH)
        if not out_down and not walls[x][y - 1]:
            legal_actions.append(game.Directions.SOUTH)
        return legal_actions

    def findCornerPositions(self, gameState):
        corner_positions = []
        if self.isRed:
            start_column = self.width // 2 + 1
            end_column = self.width

        else:
            start_column = 1
            end_column = self.width // 2
        for x in range(start_column, end_column):
            for y in range(1, gameState.data.layout.height):
                legal_actions = self.getLegalActionsFromPosition((x, y), gameState)
                legal_actions.remove(game.Directions.STOP)
                if len(legal_actions) == 1:
                    corner_positions.append((x, y))
        return corner_positions

    def calculateRiskScore(self, gameState, position, previousAction):
        position_variable = position
        previous_action_variable = previousAction
        risk_score = 0
        visited = set(position_variable)
        while True:
            legal_actions = self.getLegalActionsFromPosition(position_variable, gameState)
            if previous_action_variable in legal_actions:
                legal_actions.remove(previous_action_variable)
            if len(legal_actions) > 1:
                return risk_score
            else:
                if legal_actions[0] == game.Directions.WEST:
                    position_variable = (position_variable[0] - 1, position_variable[1])
                elif legal_actions[0] == game.Directions.EAST:
                    position_variable = (position_variable[0] + 1, position_variable[1])
                elif legal_actions[0] == game.Directions.NORTH:
                    position_variable = (position_variable[0], position_variable[1] + 1)
                elif legal_actions[0] == game.Directions.SOUTH:
                    position_variable = (position_variable[0], position_variable[1] - 1)
                risk_score += 1
                previous_action_variable = legal_actions[0]
                if position_variable in visited:
                    return risk_score
                visited.add(position_variable)
        return risk_score

    def findRiskyPositions(self, gameState):
        """
        Returns a list of tuples (position, riskScore) where position is the
        position of the food, and riskScore is how many steps it takes from that
        position to be able to move in more than one direction
        """
        risky_positions = []
        corners = self.findCornerPositions(gameState)
        for position in corners:
            legal_actions = self.getLegalActionsFromPosition(position, gameState)
            risk_score = 1
            found_exit = False;
            new_posn = position
            path_positions = dict()
            while not found_exit:
                if len(legal_actions) > 1:
                    break;
                if legal_actions[0] == game.Directions.WEST:
                    new_posn = (new_posn[0] - 1, new_posn[1])
                elif legal_actions[0] == game.Directions.EAST:
                    new_posn = (new_posn[0] + 1, new_posn[1])
                elif legal_actions[0] == game.Directions.NORTH:
                    new_posn = (new_posn[0], new_posn[1] + 1)
                elif legal_actions[0] == game.Directions.SOUTH:
                    new_posn = (new_posn[0], new_posn[1] - 1)
                if len(self.getLegalActionsFromPosition(new_posn, gameState)) > 2:
                    found_exit = True
                else:
                    risk_score += 1
                    legal_actions = self.getLegalActionsFromPosition(new_posn, gameState)
                    for p in path_positions.keys():
                        path_positions[p] += 1
                    path_positions[new_posn] = 1
            for p in path_positions.keys():
                risky_positions.append((p, path_positions[p]))
        return risky_positions

    def findRiskyDefendingFood(self, gameState):
        risky_food = []
        risky_positions = self.riskyPositions
        for position in risky_positions:
            if self.isRed():
                red_food = gameState.getRedFood()
                if red_food[position]:
                    risky_food.append(position)
            else:
                position[0] -= gameState.data.layout.width
                blue_food = gameState.getBlueFood()
                if blue_food[position]:
                    risky_food.append(position)
        self.risky_food = risky_food
        return risky_food

    def BFSVisited(self, gameState, cur, visited, depth):
        if depth == 0:
            return visited
        legal = self.getLegalActionsFromPosition(cur, gameState)
        if len(legal) == 0:
            return visited
        for action in legal:
            next = self.getSuccessor(gameState, action)
            nextPos = next.getAgentState(self.index).getPosition()
            nextPos = (int(nextPos[0]), int(nextPos[1]))
            nextBFS = {}
            if nextPos not in visited:
                visited.add(nextPos)
                nextBFS = self.BFSVisited(next, nextPos, visited, depth - 1)
            if len(nextBFS) != 0:
                visited.update(nextBFS)
                return visited
            return visited