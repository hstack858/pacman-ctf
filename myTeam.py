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
import random, time, util
from game import Directions
import game
from sklearn import preprocessing
import numpy as np


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent'):
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

    # The following line is an example only; feel free to change it.
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

        '''
    You should change this in your own agent.
    '''

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
        weights = self.getWeights(gameState, action)
        return features * weights


# AKA Aaron Pac-dgers
class OffensiveAgent(PacAttack):
    # Dimmensions (32, 16)

    def __init__(self):
      self.presentFoodList = []
      self.isRetreating = False
      self.initialTarget = []
      self.riskyPositions = None
      self.width = None
      self.height = None
      self.initial_position = None
      self.risky_food = None
      self.boundary = None
      self.isRed = None


    def registerInitialState(self, gameState):
        self.findRiskyAttackingFood(gameState)
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        self.initial_position = self.getAgentState(self.index).getPosition()
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



    def evaluate(self, gameState, action):
        return self.getFeatures(gameState, action) * self.getWeights(gameState, action)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        current_position = successor.getAgentState(self.index).getPosition()

        # Score of successor state
        features['score'] = self.getScore(successor)

        # Distance to the nearest food
        food_list = self.getFood(successor).asList()
        closest_food = sys.maxint
        for food in food_list:
            if food:
                distance = self.getMazeDistance(current_position, food)
                if distance < closest_food:
                    closest_food = distance
        features['closestFood'] = closest_food

        # Distance to the nearest capsule
        capsule_list = self.getCapsules(successor)
        if len(capsule_list) > 0:
            closest_capsule = sys.maxint
            for cap in capsule_list:
                distance = self.getMazeDistance(current_position, cap)
                if distance < closest_capsule:
                    closest_capsule = distance
            features['closestCapsule'] = closest_capsule

        # Distance to the boundary
        closest_boundary = sys.maxint
        for boundary_position in self.boundary:
            distance = self.getMazeDistance(current_position, boundary_position)
            if distance < closest_boundary:
                closest_boundary = distance
        features['closestBoundary'] = closest_boundary

        # Distances to ghosts (closest estimation)
        ghosts = self.opponentPositions(gameState)
        min_ghost_position = sys.maxint
        visible_ghosts = 0
        for ghost_position in ghosts:
            if ghost_position is not None:
                visible_ghosts += 1
                distance = self.getMazeDistance(current_position, ghost_position)
                if distance < min_ghost_position:
                    min_ghost_position = distance
        if visible_ghosts == 0:
            ghosts = []
            distances = successor.getAgentDistances()
            for index in self.getOpponents(successor):
                ghosts.append(distances[index])
            min_ghost_position = min(ghosts)
        if min_ghost_position < 6:
            min_ghost_position = 6
        features['closestGhost'] = min_ghost_position

        # Distance to friendly agent
        teammate_index = None
        team_indices = self.getTeam(successor)
        for index in team_indices:
            if index != self.index:
                teammate_index = index
        # If not within 5, will be None
        teammate_position = successor.getAgentPosition(teammate_index)
        distance = None
        if teammate_position is not None:
            distance = self.getMazeDistance(current_position, teammate_position)
        else:
            distance = successor.getAgentDistances()[teammate_index]
        features['teammateDistance'] = distance

        # Distance to the closest risky food
        closest_risky_food = sys.maxint
        closest_risky_food_risk = 0
        for position_risk_score in self.risky_food:
            distance = self.getMazeDistance(current_position, position_risk_score[0])
            if distance < closest_risky_food:
                closest_risky_food = distance
                closest_risky_food_risk = position_risk_score[1]
        features['closestRiskyFood'] = (closest_risky_food, closest_risky_food_risk)

        # Normalize feature values to be between [0, 1]
        feature_values = []
        corresponding_names = []
        for feature_name in features.keys():
            feature_values.append(features[feature_name])
            corresponding_names.append(feature_name)
        normalized_feature_values = preprocessing.normalize([feature_values])
        for name, idx in enumerate(corresponding_names):
            features[name] = normalized_feature_values[idx]
        return features



    # We should use q-learning to determine the most effective values for these
    # Values we want to be higher should be positive and values we want to be lower should be negative
    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState, action)

        # When chased by an opponent ghost, risky food, distance to opponent,
        # distance to capsule, and distance to boundary are weighted more heavily
        score_weight = 100
        if self.isRed:
            score_weight = 1
        else:
            score_weight = -1

        closest_food_weight = -10

        closest_risky_food_weight = 5

        # Depends on how close the closest ghost is
        closest_capsule_weight = -2
        # If there are no more capsules then this doesn't matter
        capsule_list = self.getCapsules(successor)
        if len(capsule_list) == 0:
            closest_capsule_weight = 0

        closest_boundary_weight = -1
        # Depends on how many pellets we are carrying, how close ghosts are, and isRetreating
        num_carrying = successor.getAgentState(self.index).numCarrying
        closest_boundary_weight *= (0.1 * num_carrying)
        if self.isRetreating:
            closest_boundary_weight * 10

        closest_ghost_weight = 5
        # If close ghosts are scared, the agent shouldn't care about their distances
        opponents = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        visible_ghosts = []
        for opponent in opponents:
            if opponent.getPosition() is not None:
                visible_ghosts.append(opponent)
        if len(visible_ghosts) > 0:
            for ghost in visible_ghosts:
                if ghost.scaredTimer > 0:
                    # The ghost is scared
                    if ghost.scaredTimer > 12:
                        closest_ghost_weight = -2
                        closest_risky_food_weight = -5
                        score_weight += num_carrying * 5

                    elif ghost.scaredTimer > 6:
                        closest_ghost_weight = -1
                        score_weight += num_carrying * 3
                    else:
                        closest_ghost_weight = 10
                else:
                    # Visible and NOT scared
                    closest_ghost_weight = 20
                    closest_risky_food_weight = 10
                    closest_capsule_weight = -3
        else:
            # We don't see any ghosts close by
            # TODO: Incorporate risk factor somehow
            closest_ghost_weight = 0
            closest_capsule_weight = 0
            closest_boundary_weight = 0
            closest_risky_food_weight = -5

        closest_teammate_distance_weight = 3

        weights = {
            'score': score_weight,
            'closestFood': closest_food_weight,
            'closestCapsule': closest_capsule_weight,
            'closestBoundary': closest_boundary_weight,
            'closestGhost': closest_ghost_weight,
            'teammateDistance': closest_teammate_distance_weight,
            'closestRiskyFood': closest_risky_food_weight
        }
        return weights

    def opponentPositions(self, gameState):
      return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

    def atRiskOfDeath(self, gameState):
      # Returns a location tuple if the agent unless unobservable, then None
      enemy_positions = self.opponentPositions(gameState)
      current_position = self.getAgentState(self.index).getPosition()
      for posn in enemy_positions:
        if posn is not None:
          distance = self.getMazeDistance(posn, current_position)
          if distance < 3:
              return True
          for position_riskScore in self.riskyPositions:
              if current_position == position_riskScore[0]:
                  # Test to see if we can make it out
                  return abs(distance - position_riskScore[1] >= distance)

    def shouldHeadBack(self, gameState):
        num_carrying = gameState.getAgentState(self.index).numCarrying
        val = 0
        if self.isRed:
            if gameState.score > 20:
                val += .4
        else:
            if gameState.score < -20:
                val += .4
        if num_carrying > 5:
            val += .1
            if num_carrying > 10:
                val += .1
                if num_carrying > 15:
                    val += .1
                    if num_carrying > 15:
                        val += .1
        return val > .5


    def getLegalActionsFromPosition(self, position, gameState):
        walls = gameState.getWalls()
        legal_actions = []
        if not walls[position[0] - 1][position[1]]:
            legal_actions.append('Left')
        if not walls[position[0] + 1][position[1]]:
            legal_actions.append('Right')
        if not walls[position[0]][position[1] + 1]:
            legal_actions.append('Up')
        if not walls[position[0] + 1][position[1] - 1]:
            legal_actions.append('Down')
        return legal_actions

    def findRiskyPositions(self, gameState):
        """
        Returns a list of tuples (position, riskScore) where position is the
        position of the food, and riskScore is how many steps it takes from that
        position to be able to move in more than one direction
        """
        risky_positions = []
        start_column = None
        if self.isRed:
            start_column = gameState.data.layout.width
        else:
            start_column = 0
        for x in range(start_column, start_column + gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                legal_actions = self.getLegalActionsFromPosition((x, y), gameState)
                if len(legal_actions) == 1:
                    risk_score = 1
                    found_exit = False;
                    while not found_exit:
                        if legal_actions[0] == 'Left':
                            legal_actions = self.getLegalActionsFromPosition((x - 1, y), gameState)
                        elif legal_actions[0] == 'Right':
                            legal_actions = self.getLegalActionsFromPosition((x + 1, y), gameState)
                        elif legal_actions[0] == 'Up':
                            legal_actions = self.getLegalActionsFromPosition((x, y + 1), gameState)
                        elif legal_actions[0] == 'Down':
                            legal_actions = self.getLegalActionsFromPosition((x, y - 1), gameState)
                        if len(legal_actions) != 1:
                            found_exit = True
                        else:
                            risk_score += 1
                risky_positions.append(((x, y), risk_score))
        self.riskyPositions = risky_positions

    def findRiskyAttackingFood(self, gameState):
        risky_food = []
        risky_positions = self.findRiskyPositions(gameState)
        for position in risky_positions:
            if self.isBlue():
                red_food = gameState.getRedFood()
                if red_food[position]:
                    risky_food.append(position)
            else:
                position[0] -= gameState.data.layout.width
                blue_food = gameState.getBlueFood()
                if blue_food[position]:
                    risky_food.append(position)
        return risky_food

    def closePositions(self, gameState, position, stepsToExplore, visitedPositions):
        if stepsToExplore == 0:
            return
        legal_actions = self.getLegalActionsFromPosition(position, gameState)
        for action in legal_actions:
            new_position = None
            if action == 'Left':
                new_position = (position[0] - 1, position[1])
            if action == 'Right':
                new_position = (position[0] + 1, position[1])
            if action == 'Up':
                new_position = (position[0], position[1] + 1)
            if action == 'Down':
                new_position = (position[0], position[1] - 1)
            if new_position not in visitedPositions:
                visitedPositions.add(new_position)
                self.closePositions(gameState, new_position, stepsToExplore-1, visitedPositions)
        return visitedPositions

    def closeToRewards(self, gameState):
        current_position = self.getPosition()
        # Count how many pellets are within 4 units from pacman
        pellet_count = 0
        possible_positions = self.closePositions(gameState, current_position, 4, set())
        food = gameState.getFood()
        for position in possible_positions:
            if food[position[0]][position[1]]:
                pellet_count += 1
        return pellet_count >= 2


    def montePossibleActions(self, state):
        legal_actions = state.getLegalActions(self.index)
        legal_actions.remove(Directions.STOP)
        if len(legal_actions == 1):
            return legal_actions[0]
        else:
            reverse_direction = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
            if reverse_direction in legal_actions:
                legal_actions.remove(reverse_direction)
            return legal_actions

    def monteCarloSimulation(self, gameState, depth, decay):
        copy = gameState.deepCopy()
        if depth == 0:
            new_actions = self.montePossibleActions(copy)
            new_action = random.choice(new_actions)
            next_state = copy.generateSuccessor(self.index, new_action)
            return self.evaluate(next_state, Directions.STOP)

        results = []
        actions = dict()
        new_actions = self.montePossibleActions(copy)
        for action in new_actions:
            new_state = copy.generateSuccessor(self.index, action)
            score = self.evaluate(new_state, Directions.STOP) + decay * self.monteCarloSimulation(new_state, depth - 1, decay)
            results.append(score)
            actions[score] = action
        return actions[max(results)]

    def chooseAction(self, gameState):
        start = time.time()

        if self.atRiskOfDeath(gameState):
        # avoid dying bro
        else:
            if self.shouldHeadBack(gameState):
            # return home
                self.isRetreating = True
            else:
                if self.closeToRewards(gameState):
                    # monte-carlo
                    return self.monteCarloSimulation(gameState, 4, 0.7)
                else:
                    # A*
            if (time.time() - start) > 1.5:
                print 'eval time for offensive agent %d: %.4f' % (
                    self.agent.index, time.time() - start)

            


class DefensiveAgent(PacAttack):

    def __init__(self):

    # ...

    def chooseAction(self, gameState):
        if theyHavePowerPellet(gameState):
        # don't die
        else:
            if crossingRisk(gameState, val):
            # move to better defensive position
            else:
        # play defense fucker


class attackTheirPac(PacAttack):

    def __init__(self):

    # ...

    def chooseAction(self, gameState):
# third agent type specificly designed for when we want to ignore food and only focus on capturing them
# useful if they're already down one pac, or depending on game state


