# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import json

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions, AgentState
import game
from util import nearestPoint

#################
# Team creation #
#################
training = False


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    def __init__(self, index, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        self.height = None
        self.width = None

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
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height

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

    def getSuccessor(self, gameState, action):
        """
        - Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

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

    def getPolicy(self, gameState):
        """
        - Returns the legal action with the highest Qvalue
        - If there are no legal actions, the action returned will be None
        """
        values_and_actions = dict()
        legal_actions = gameState.getLegalActions(self.index)
        legal_actions.remove(Directions.STOP)
        if len(legal_actions) == 0:
            return None
        for action in legal_actions:
            value = self.getActionQValue(gameState, action)
            values_and_actions[value] = action
        max_value = max(values_and_actions.keys())
        policy = values_and_actions[max_value]
        return policy

    def getMaxQValue(self, gameState):
        """
        - Returns the Qvalue from the best possible legal action of the state
        - If there are no legal actions, the Qvalue returned will be 0
        """
        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return 0.0
        action_from_policy = self.getPolicy(gameState)
        q_value_from_action = self.getActionQValue(gameState, action_from_policy)
        return q_value_from_action

    def getQValue(self, gameState):
        """
        - Returns the Q value of the current gameState
        """
        features = self.getFeatures(gameState, Directions.STOP)
        value = features * self.weights
        return value

    def getActionQValue(self, gameState, action):
        """
        - Returns the Q value for taking a given action
        """
        features = self.getFeatures(gameState, action)
        return features * self.weights

    # Helper Functions -------------------------------------------------------------------
    def isBlue(self):
        """
        - Returns whether the agent is on the blue team or not
        """
        return not self.isRed

    def getPosition(self, gameState):
        """
        - Convenience method for getting the position of our agent
        """
        return gameState.getAgentState(self.index).getPosition()

    def opponentPositions(self, gameState):
        """
        - Returns the positions of our opponents or None if they aren't visible
        """
        return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

    def calculateRiskScore(self, gameState, position):
        """
        - Calculates the "risk score" for a given position
        - A risk score is how many walls are surrounding a position
        """
        walls = gameState.getWalls()
        left = (position[0] - 1, position[1])
        right = (position[0] + 1, position[1])
        up = (position[0], position[1] + 1)
        down = (position[0], position[1] - 1)

        can_left = left[0] > 1
        can_right = right[0] < self.width
        can_down = down[1] > 1
        can_up = up[1] < self.height

        risk_score = 0
        if can_left and walls[left[0]][left[1]]:
            risk_score += 1
        if can_right and walls[right[0]][right[1]]:
            risk_score += 1
        if can_up and walls[up[0]][up[1]]:
            risk_score += 1
        if can_down and walls[down[0]][down[1]]:
            risk_score += 1

        return risk_score

    def findRiskyPositions(self, gameState):
        """
        Returns a list of tuples (position, riskScore) where position is the
        position of the food, and riskScore is how many steps it takes from that
        position to be able to move in more than one direction
        """
        risky_positions = []
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                position = (x, y)
                risk_score = self.calculateRiskScore(gameState, position)
                if risk_score > 2:
                    risky_positions.append((position, risk_score))
        self.risky_positions = risky_positions

    def getLegalActionsFromPosition(self, position, gameState):
        """
        - Returns the legal actions you can take from a given position
        """
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

    def getTeammatePositions(self, gameState):
        teammate_states = [gameState.getAgentState(index) for index in self.getTeam(gameState)]
        my_agent_state = gameState.getAgentState(self.index)
        teammate_positions = [agent_state.getPosition() for agent_state in teammate_states if agent_state != my_agent_state]
        return teammate_positions


class OffensiveAgent(DummyAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.start = None
        self.width = None
        self.height = None
        self.risky_food = None
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
        if gameState.isOnRedTeam(self.index):
            self.isRed = True
        else:
            self.isRed = False
        self.findRiskyPositions(gameState)
        self.risky_food = self.findRiskyAttackingFood(gameState)
        CaptureAgent.registerInitialState(self, gameState)

    def closestCapsule(self, gameState):
        current_position = gameState.getAgentPosition(self.index)
        capsules = self.getCapsules(gameState)
        if len(capsules) == 0 or capsules is None:
            return None
        return min([self.getMazeDistance(current_position, capsule) for capsule in capsules])

    def closestFood(self, gameState, position):
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        frontier = [(position[0], position[1], 0)]
        visited = set()
        while frontier:
            x, y, distance = frontier.pop(0)
            x = int(x)
            y = int(y)
            if (x, y) in visited:
                continue
            if food[x][y]:
                return distance

            visited.add((x, y))
            legal_neighbors = Actions.getLegalNeighbors((x, y), walls)
            for new_x, new_y in legal_neighbors:
                frontier.append((new_x, new_y, distance + 1))
        return None

    def findRiskyAttackingFood(self, gameState):
        """
        Returns a list of positions with risky food
        """
        risky_food = []
        risky_positions = self.risky_positions
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

    def getFeatures(self, gameState, action):
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        current_position = gameState.getAgentState(self.index).getPosition()
        current_position = (int(current_position[0]), int(current_position[1]))
        direction_x, direction_y = Actions.directionToVector(action)
        next_position = (int(current_position[0] + direction_x), int(current_position[1] + direction_y))
        next_agent_state = successor.getAgentState(self.index)

        features['bias'] = 1.0

        features["score"] = self.getScore(successor)

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
        walls = gameState.getWalls()
        closest_food = self.closestFood(gameState, next_position)
        if closest_food is not None:
            features['closestFood'] = float(closest_food) / (walls.width * walls.height)

        # Distance to the nearest capsule
        closest_capsule = self.closestCapsule(gameState)
        if closest_capsule is not None:
            if closest_capsule < 6:
                features['closestCapsule'] = -closest_capsule

        # Distance One Ghosts (Could be eaten!)
        ghost_states = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
        ghost_positions = [ghost.getPosition() for ghost in ghost_states if
                           not ghost.isPacman and ghost.getPosition() is not None and (ghost.scaredTimer < 3)]
        for ghost_position in ghost_positions:
            distance_to_ghost = self.getMazeDistance(next_position, ghost_position)
            if distance_to_ghost <= 1:
                features['distanceOneGhostCount'] += 1

        # Distance to friendly agent
        teammate_positions = self.getTeammatePositions(gameState)
        if len(teammate_positions) > 0:
            min_teammate_distance = min([self.getMazeDistance(current_position, position) for position in teammate_positions])
            features['teammateDistance'] = min_teammate_distance

        if next_agent_state.isPacman:
            features['onDefense'] = 0
        else:
            features['onDefense'] = 1

        # Distance to the closest risky food
        closest_risky_food = 1000
        self.risky_food = self.updateRiskyFoodGrid(gameState)
        for position_risk_score in self.risky_food:
            position = position_risk_score[0]
            if food[position[0]][position[1]]:
                distance = self.getMazeDistance(current_position, position_risk_score[0])
                if distance < closest_risky_food:
                    closest_risky_food = distance
                    features['closestRiskyFood'] = closest_risky_food

        features.divideAll(10.0)

        return features

    def chooseAction(self, gameState):
            legal_actions = gameState.getLegalActions(self.index)
            if not legal_actions:
                return None

            if training:
                for action in legal_actions:
                    self.updateWeights(gameState, action)

            probability = util.flipCoin(self.epsilon)
            if probability:
                if self.closeToRewards(gameState):
                    # Highest reward from monte carlo
                    action = self.monteCarloSimulation(gameState, self.depth, self.decay)
                else:
                    action = random.choice(legal_actions)
            else:
                action = self.getPolicy(gameState)
            return action

    def getRewardStateValue(self, gameState, nextState):
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

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)
        print("Final weights for offensive agent:")
        print(self.weights)

class DefensiveAgent(DummyAgent):
    """
  An approximate Q-learning agent for the offensive agent
  """

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

    def chooseAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
      """
        # Pick Action
        legalActions = state.getLegalActions(self.index)

        foodLeft = len(self.getFood(state).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in legalActions:
                successor = self.getSuccessor(state, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        action = None
        if not legalActions:
            return action

        if training:
            for action in legalActions:
                self.updateWeights(state, action)

        p = self.epsilon
        if util.flipCoin(p):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        return action

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

    def median(self, data):
        n = len(data)
        index = n // 2
        if n % 2:
            return sorted(data)[index]
        return sum(sorted(data)[index - 1:index + 1]) / 2

    def getFeatures(self, gameState, action):
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        current_agent_state = gameState.getAgentState(self.index)
        next_agent_state = successor.getAgentState(self.index)
        next_position = next_agent_state.getPosition()
        next_position = (int(next_position[0]), int(next_position[1]))


        enemy_states = [successor.getAgentState(opponent) for opponent in self.getOpponents(successor)]
        pac_states = [enemy for enemy in enemy_states if enemy.isPacman and enemy.getPosition() is not None]
        features['pacCount'] = len(pac_states)

        if len(pac_states) > 0:
            distances = [self.getMazeDistance(next_position, pac.getPosition()) for pac in pac_states]
            features['closestPac'] = min(distances)

        our_food = self.getFoodYouAreDefending(successor)
        self.findRiskyDefendingFood(successor)
        for risky_food in self.risky_food:
            if our_food[risky_food[0]][risky_food[1]]:
                features['riskyFoodCount'] += 1

        # Sum of close risky positions' risk scores
        risk_score = 0
        nearby_tiles = self.BFSVisited(successor, next_position, set(next_position), 3)
        for pos in nearby_tiles:
            if pos in self.risky_food:
                risk_score += self.riskyPositions[pos]

        features['riskScore'] = risk_score / len(nearby_tiles)

        boundary_y_coords = []
        x_coord = None
        closest_boundary = sys.maxint
        for boundary_position in self.boundary:
            if boundary_position is not None and next_position is not None:
                boundary_y_coords.append(boundary_position[1])
                x_coord = boundary_position[0]
                distance = self.getMazeDistance(next_position, boundary_position)
                if distance < closest_boundary:
                    closest_boundary = distance
                    features['closestBoundary'] = closest_boundary
        boundary_position = (x_coord, self.median(boundary_y_coords))
        features["middleBoundaryDistance"] = self.getMazeDistance(next_position, boundary_position)

        if action == Directions.STOP:
            features['stop'] = 1

        reverse = Directions.REVERSE[current_agent_state.configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        features.divideAll(10.0)
        return features

        ################################### HELPERS ###################################

    def findRiskyDefendingFood(self, gameState):
        risky_food = []
        risky_positions = self.riskyPositions
        for position in risky_positions:
            if self.isRed():
                red_food = gameState.getRedFood()
                if red_food[position]:
                    risky_food.append(position)
            else:
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

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)
        print("Final weights for defensive agent:")
        print(self.weights)

