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
from sklearn import preprocessing


#################
# Team creation #
#################

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
class OffensiveAgent(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.riskyPositions = None
        self.width = None
        self.height = None
        self.risky_food = None
        self.boundary = None
        self.isRed = None
        self.weights = None
        # Chance to explore
        self.epsilon = 0.1
        # Learning Rate
        self.alpha = 0.2
        # For monte carlo
        self.depth = 5
        self.decay = 0.9


    def registerInitialState(self, gameState):
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
        self.weights = {
            'score': 0,
            'closestFood': 0,
            'closestCapsule': 0,
            'closestBoundary': 0,
            'closestGhost': 0,
            'teammateDistance': 0,
            'closestRiskyFood': 0
        }

    # Q-Learning Functions ----------------------------------------------------------------------

    def evaluate(self, gameState, action):
        return self.getFeatures(gameState, action) * self.getWeights(gameState, action)

    def getValue(self, gameState):
        values = []
        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return 0.0
        for action in legal_actions:
            values.append(self.evaluate(gameState, action))
        return max(values)

    def getPolicy(self, gameState):
        values_and_actions = dict()
        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return None
        for action in legal_actions:
            values_and_actions[self.evaluate(gameState, action)] = action
        max_value = max(values_and_actions.keys())
        return values_and_actions[max_value]

    def chooseAction(self, gameState):
        start = time.time()
        probability = util.flipCoin(self.epsilon)
        if probability:
            if self.closeToRewards(gameState):
                # Highest reward from monte carlo
                action = self.monteCarloSimulation(gameState, self.depth, self.decay)
            else:
                # A* to find closest pellet
                action = self.aStarClosestPellet(gameState)
        else:
            action = self.getPolicy(gameState)
        if (time.time() - start) > 1.5:
            print 'eval time for offensive agent %d: %.4f' % (
                self.agent.index, time.time() - start)
        return action

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
        self.risky_food = self.updateRiskyFoodGrid(gameState)
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

    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        # Reward function needs work
        # Reward for surviving
        reward = nextState.getScore() - gameState.getScore()
        for feature in features:
            weight_correction = reward + self.decay * self.getValue(nextState) - self.evaluate(gameState, action)
            self.weights[feature] = self.weights[feature] + self.alpha * weight_correction * features[feature]

    # Helper Functions -------------------------------------------------------------------------

    def isBlue(self):
        return not self.isRed()

    def getPosition(self, gameState):
        return gameState.getAgentState(self.index).getPosition()

    def opponentPositions(self, gameState):
        return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

    def getLegalActionsFromPosition(self, position, gameState):
        walls = gameState.getWalls()
        legal_actions = []
        out_right = position[0] + 1 >= self.width
        out_left = position[0] - 1 < 1
        out_down = position[1] - 1 < 1
        out_up = position[1] + 1 >= self.height

        if not out_left and not walls[position[0] - 1][position[1]]:
            legal_actions.append('Left')
        if not out_right and not walls[position[0] + 1][position[1]]:
            legal_actions.append('Right')
        if not out_up and not walls[position[0]][position[1] + 1]:
            legal_actions.append('Up')
        if not out_down and not walls[position[0]][position[1] - 1]:
            legal_actions.append('Down')
        return legal_actions

    def findCornerPositions(self, gameState):
        corner_positions = []
        if self.isRed:
            start_column = self.width // 2 + 1
            end_column = self.width

        else:
            start_column = 1
            end_column = self.width // 2 + 1
        for x in range(start_column, end_column):
            for y in range(1, gameState.data.layout.height):
                legal_actions = self.getLegalActionsFromPosition((x, y), gameState)
                if len(legal_actions) == 1:
                    corner_positions.append((x, y))
        return corner_positions

    def calculateRiskScore(self, gameState, position, previousAction):
        print("calculateRiskScore")
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
                print(len(legal_actions) > 1)
                print(legal_actions)
                print(position_variable)
                if legal_actions[0] == 'Left':
                    position_variable = (position_variable[0] - 1, position_variable[1])
                elif legal_actions[0] == 'Right':
                    position_variable = (position_variable[0] + 1, position_variable[1])
                elif legal_actions[0] == 'Up':
                    position_variable = (position_variable[0], position_variable[1] + 1)
                elif legal_actions[0] == 'Down':
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
        #
        print("findRiskyPositions")
        risky_positions = []
        corners = self.findCornerPositions(gameState)
        for position in corners:
            legal_actions = self.getLegalActionsFromPosition(position, gameState)
            risk_score = 1
            found_exit = False;
            new_posn = position
            path_positions = []
            while not found_exit:
                if len(legal_actions) > 1:
                    break;
                if legal_actions[0] == 'Left':
                    new_posn = (new_posn[0] - 1, new_posn[1])
                elif legal_actions[0] == 'Right':
                    new_posn = (new_posn[0] + 1, new_posn[1])
                elif legal_actions[0] == 'Up':
                    new_posn = (new_posn[0], new_posn[1] + 1)
                elif legal_actions[0] == 'Down':
                    new_posn = (new_posn[0], new_posn[1] - 1)
                if len(self.getLegalActionsFromPosition(new_posn, gameState)) > 2:
                    found_exit = True
                else:
                    risk_score += 1
                    legal_actions = self.getLegalActionsFromPosition(new_posn, gameState)
                    for p in path_positions:
                        p[1] += 1
                        path_positions.append((new_posn, 1))
            for p in path_positions:
                risky_positions.append(p)
        return risky_positions

    def findRiskyAttackingFood(self, gameState):
        """
        Returns a list of positions with risky food
        """
        risky_food = []
        risky_positions = self.findRiskyPositions(gameState)
        for position_risk_score in risky_positions:
            if self.isBlue():
                red_food = gameState.getRedFood()
                if red_food[position_risk_score[0]]:
                    risky_food.append(position_risk_score)
            else:
                position_risk_score[0] -= gameState.data.layout.width
                blue_food = gameState.getBlueFood()
                if blue_food[position_risk_score[0]]:
                    risky_food.append(position_risk_score)
        return risky_food

    def updateRiskyFoodGrid(self, gameState):
        new_risky_food = []
        for position_risk_score in self.risky_food:
            if self.isBlue():
                red_food = gameState.getRedFood()
                if red_food[position_risk_score[0]]:
                    new_risky_food.append(position_risk_score)
            else:
                position_risk_score[0] -= gameState.data.layout.width
                blue_food = gameState.getBlueFood()
                if blue_food[position_risk_score[0]]:
                    new_risky_food.append(position_risk_score)
        return new_risky_food

    def closePositions(self, gameState, position, stepsToExplore, visitedPositions):
        """
        Returns a set of positions that can be reached within stepsToExplore steps
        """
        print("closePositions")
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
                self.closePositions(gameState, new_position, stepsToExplore - 1, visitedPositions)
        return visitedPositions

    def closeToRewards(self, gameState):
        """
        Determines whether there are 2 or more rewards within 5 units of Pacman
        """
        current_position = self.getPosition(gameState)
        pellet_count = 0
        possible_positions = self.closePositions(gameState, current_position, 5, set())
        food = gameState.getFood()
        for position in possible_positions:
            if food[position[0]][position[1]]:
                pellet_count += 1
        return pellet_count >= 2

    def montePossibleActions(self, state):
        """
        Returns the list of possible actions the Monte Carlo Simulation could take
        """
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
        """
        Performs a Monte Carlo simulation on the given depth and returns the score maximizing aciton
        """
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
            score = self.evaluate(new_state, Directions.STOP) + decay * self.monteCarloSimulation(new_state, depth - 1,
                                                                                                  decay)
            results.append(score)
            actions[score] = action
        return actions[max(results)]

    def aStarClosestPellet(self, gameState):
        print("aStar")
        from util import PriorityQueue
        priority_queue = PriorityQueue()
        priority_queue.push((self.getPosition(gameState), []))
        visited = set()

        while not priority_queue.isEmpty():
            coords, path = priority_queue.pop()
            if coords in visited:
                continue
            visited.add(coords)
            if self.isRed():
                if gameState.getBlueFood()[coords[0]][coords[1]]:
                    return path[0]
            elif gameState.redFood()[coords[0]][coords[1]]:
                return path[0]
            for action in self.getLegalActionsFromPosition(coords):
                if action == "Left":
                    new_posn = (coords[0] - 1, coords[1])
                    longer_path = path + "Left"
                    priority_queue.push((new_posn, longer_path))
                elif action == "Right":
                    new_posn = (coords[0] - 1, coords[1])
                    longer_path = path + "Right"
                    priority_queue.push((new_posn, longer_path))

                elif action == "Up":
                    new_posn = (coords[0], coords[1] + 1)
                    longer_path = path + "Up"
                    priority_queue.push((new_posn, longer_path))

                elif action == "Down":
                    new_posn = (coords[0], coords[1] - 1)
                    longer_path = path + "Down"
                    priority_queue.push((new_posn, longer_path))


class DefensiveAgent(CaptureAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.foodDefensePos = None
        self.closestPacPos = None
        self.isRed = None
        self.initial_position = None
        self.risky_food = []
        self.riskyPositions = []
        self.width = None
        self.height = None
        self.boundary = []
        self.lastEatenFood = None
        self.oldFood = None
        self.epsilon = 0.1
        self.alpha = 0.2
        self.decay = 0.9
        # TODO: adjust  these two possibly
        self.guardDepth = 3
        self.runAway = 2

    def registerInitialState(self, gameState):
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        self.initial_position = self.getPosition(gameState)
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
        self.risky_food = self.findRiskyDefendingFood(gameState)
        self.closestPacPos = self.findClosestPac(gameState)[0]
        if self.isRed:
            self.oldFood = gameState.getRedFood()
        else:
            self.oldFood = gameState.getBlueFood()

    def getValue(self, gameState):
        values = []
        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return 0.0
        for action in legal_actions:
            values.append(self.evaluate(gameState, action))
        return max(values)

    def getPolicy(self, gameState):
        values_and_actions = dict()
        legal_actions = gameState.getLegalActions(self.index)
        if len(legal_actions) == 0:
            return None
        for action in legal_actions:
            values_and_actions[self.evaluate(gameState, action)] = action
        max_value = max(values_and_actions.keys())
        return values_and_actions[max_value]

    def evaluate(self, gameState, action):
        return self.getFeatures(gameState, action) * self.getWeights(gameState, action)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        current_position = successor.getAgentState(self.index).getPosition()
        pacTuple = self.pacTuple(gameState)

        features['score'] = self.getScore(successor)

        closest_boundary = sys.maxint
        for boundary_position in self.boundary:
            distance = self.getMazeDistance(current_position, boundary_position)
            if distance < closest_boundary:
                closest_boundary = distance

        features['closestBoundary'] = closest_boundary

        features['closestPacman'] = pacTuple[2]

        features['closestPacmanRisk'] = self.riskyPositions[pacTuple[1]]

        self.findRiskyDefendingFood(gameState)
        features['riskyFoodCount'] = len(self.risky_food)

        return features

    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = nextState.getScore() - gameState.getScore()
        for feature in features:
            weight_correction = reward + self.decay * self.getValue(nextState) - self.evaluate(gameState, action)
            self.weights[feature] = self.weights[feature] + self.alpha * weight_correction * features[feature]

    """
    Q Learning Instead.
    def getWeights(self, gameState, action):
        scoreWeight = 0
        closestBoundaryWeight = 0
        closestPacmanWeight = 0
        closestPacmanRiskWeight = 0
        riskyFoodCountWeight = 0
        weights = {
            'score': scoreWeight,
            'closestBoundary': closestBoundaryWeight,
            'closestPacman': closestPacmanWeight,
            'closestPacmanRisk': closestPacmanRiskWeight,
            'riskyFoodCount': riskyFoodCountWeight,
        }
        return weights
    """

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

    def getPosition(self, gameState):
        return gameState.getAgentState(self.index).getPosition()

    def findCornerPositions(self, gameState):
        corner_positions = []
        if self.isRed:
            start_column = self.width // 2 + 1
            end_column = self.width

        else:
            start_column = 1
            end_column = self.width // 2 + 1
        for x in range(start_column, end_column):
            for y in range(1, gameState.data.layout.height):
                legal_actions = self.getLegalActionsFromPosition((x, y), gameState)
                if len(legal_actions) == 1:
                    corner_positions.append((x, y))
        return corner_positions

    def getPosition(self, gameState):
        return gameState.getAgentState(self.index).getPosition()

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

    def pacTuple(self, gameState):
        agentPos = self.getPosition(gameState)
        pacPos = self.findClosestPac(gameState)[0]
        dist = self.getMazeDistance(agentPos, pacPos)
        return (agentPos, pacPos, dist)

    def bestFoodDefense(self, gameState):
        foodRisk = self.findRiskyDefendingFood(gameState)
        riskyPos = self.riskyPositions
        aggregateFoodRisk = util.Counter()
        if self.isRed:
            start_column = 0
        else:
            start_column = gameState.data.layout.width
        for x in range(start_column + 2, start_column + gameState.data.layout.width - 2):
            for y in range(2, gameState.data.layout.height - 2):
                avgRisk = 0
                for i in range(x - 2, x + 2):
                    for j in range(y - 2, y + 2):
                        if foodRisk[(i, j)] != None:
                            risk = riskyPos[(x, y)] / 25
                            if (i, j) in self.getCapsules(gameState):
                                risk = risk * 2

                            avgRisk += risk

                aggregateFoodRisk[(x, y)] = avgRisk

        bestPos = None
        bestRisk = -99999999
        for pos in aggregateFoodRisk.keys():
            if aggregateFoodRisk[pos] > bestRisk:
                bestRisk = aggregateFoodRisk[pos]
                bestPos = pos

        self.foodDefensePos = bestPos
        return bestPos

    def updateFood(self, gameState):
        food = None
        if self.isRed:
            food = gameState.getRedFood()
        else:
            food = gameState.getBlueFood()

        if len(food) < len(self.oldFood):
            for oldF in self.oldFood:
                if oldF not in food:
                    self.lastEatenFood = oldF

            self.oldFood = food

    def estimateClosestPac(self, gameState):
        pos = self.lastEatenFood
        if self.getMazeDistance(pos, gameState.getAgentState(self.index)) < 3:
            return self.bestFoodDefense(gameState)

        return pos

    def findClosestPac(self, gameState):
        pos = None
        closest = 99999999
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        opponentPos = [gameState.getAgentPosition(j) for j in self.getOpponents(gameState)]
        for k in range(len(opponents)):
            if opponents[k] != None:
                dist = self.getMazeDistance(self.getPosition(gameState), opponentPos[k])
                if dist < closest and opponents[k].isPacman:
                    pos = opponentPos[k]
                    closest = dist

        if pos == None:
            pos = self.estimateClosestPac(gameState)

        self.closestPacPos = pos
        return pos

    def pacRisk(self, gameState):
        nearestPac = self.pacTuple(gameState)
        risk = self.riskyPositions
        ourRisk = risk[nearestPac[0]]

        if nearestPac[2] < ourRisk + self.runAway:  # TODO: this val may need to be altered depending on performance...
            return True

        return False

    def ghostRisk(self, gameState):
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        opponentPos = [gameState.getAgentPosition(j) for j in self.getOpponents(gameState)]
        closest = 99999999
        risk = self.riskyPositions
        ourRisk = risk[self.getAgentPosition(self.index)]
        for k in range(len(opponents)):
            if opponents[k] != None:
                dist = self.getMazeDistance(self.getPosition(gameState), opponentPos[k])
                if dist < closest and not opponents[k].isPacman:
                    closest = dist

        if closest != 99999999 and closest < ourRisk + self.runAway:
            return True

        return False

    def atRiskOfDeath(self, gameState):
        agentState = gameState.getAgentState(self.index)
        if agentState.isPacman and self.ghostRisk(gameState):
            return True

        elif agentState.scaredTimer > 0 and self.pacRisk(gameState):
            return True

        else:
            return False

    def defendFood(self, gameState):
        return self.evaluate(gameState, Directions.STOP) > 0  # TODO: see how this works with our Q-learning

    def runAway(self, gameState):
        nearestPac = self.pacTuple(gameState)
        actions = self.getLegalActions(self.index)
        actions.remove(Directions.STOP)

        bestDist = nearestPac[2]
        bestAction = Directions.STOP
        for action in actions:
            nextState = gameState.generateSuccessor(self.index, action)
            nextPos = nextState.getAgentPosition(self.index)
            newDist = self.getMazeDistance(nextPos, nearestPac[1])
            if newDist > bestDist:
                bestDist = newDist
                bestAction = action

        return bestAction

    def guard(self, gameState):
        start = gameState.getAgentPosition(self.index)
        nearestPac = self.pacTuple(gameState)
        visitable = self.BFSVisited(gameState, start, {start}, self.guardDepth)
        bestDist = 99999999
        bestPos = None
        for pos in visitable:
            dist = self.getMazeDistance(pos, nearestPac[1])
            if range < self.guardDepth and bestDist > dist:
                bestDist = dist
                bestPos = pos

        return self.goTo(bestPos)

    def BFSVisited(self, gameState, cur, visited, depth):
        if depth == 0:
            return visited
        legal = self.getLegalActionsFromPosition(cur, gameState)
        for action in legal:
            next = self.getSuccessor(gameState, action)
            nextPos = next.getAgentState(self.index).getPosition()
            if nextPos not in visited:
                visited.add(nextPos)
                return visited.update(self.BFSVisited(next, nextPos, visited, depth - 1))

    def getCorrectAction(tree, start):
        nextPos = tree[start]
        if nextPos[0] - 1 == start[0]:
            return Directions.EAST
        elif nextPos[0] + 1 == start[0]:
            return Directions.WEST
        elif nextPos[1] - 1 == start[1]:
            return Directions.NORTH
        elif nextPos[1] + 1 == start[1]:
            return Directions.SOUTH

        return Directions.STOP

    def goTo(self, gameState, target):
        start = gameState.getAgentPosition(self.index)
        queue = util.Queue()
        queue.push(start)
        visited = [start]
        path = {start: start}
        while not queue.isEmpty():
            cur = queue.pop()
            if cur == target:
                return self.getCorrectAction(path, start)
            else:
                for action in self.getLegalActionsFromPosition(cur):
                    next = cur
                    if action == Directions.NORTH:
                        next = (cur[0], cur[1] + 1)
                    elif action == Directions.SOUTH:
                        next = (cur[0], cur[1] - 1)
                    elif action == Directions.EAST:
                        next = (cur[0] + 1, cur[1])
                    elif action == Directions.WEST:
                        next = (cur[0] - 1, cur[1])

                    if next not in visited:
                        queue.push(next)
                        visited.append(next)
                        path[cur] = next

        return Directions.STOP

    def targetPac(self, gameState, target):
        return self.goTo(gameState, target)

    def zoneDefense(self, gameState, center):
        agentPos = self.getAgentPosition(self.index)

        if self.getMazeDistance(agentPos, center) > self.guardDepth:
            return self.goTo(gameState, center)

        return self.guard(gameState)

    def chooseAction(self, gameState):
        self.updateFood(gameState)
        if self.atRiskOfDeath(gameState):
            return self.runAway(gameState)

        elif self.defendFood(gameState):
            center = self.bestFoodDefense()
            return self.zoneDefense(gameState, center)

        else:
            target = self.closestPacPos
            return self.targetPac(gameState, target)