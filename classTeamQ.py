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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import math
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ApproximateQLearningAgent', second = 'MCTSAgent'):
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

class GeneralAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        self.start_pos = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.chased = False  # if pacman is being chased while on offense

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start_pos, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction
        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """Finds the next successor which is a grid position (location tuple)."""
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """Computes a linear combination of features and feature weights"""
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """Returns a counter of features for the state"""
        features = util.Counter()
        for f, v in self.getWeights(gameState, action).items():
            features[f] = 0

        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # distance to start after taking action
        successor = self.getSuccessor(gameState, action)
        oldPos = gameState.getAgentState(self.index).getPosition()
        myPos = successor.getAgentState(self.index).getPosition()
        if self.chased == True:
            dist = self.getMazeDistance(myPos, self.start_pos)
            features['distanceToStart'] = dist

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        if gameState.getAgentState(self.index).isPacman:  # on offense / pacman:
            if self.chased == False:
                foodList = self.getFood(successor).asList()
                features['numFood'] = len(foodList)

                if len(foodList) < len(
                        self.getFood(gameState).asList()):  # if eating food, don't penalize dist to nearest food
                    features['distanceToFood'] = 0
                else:
                    features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in foodList])

            defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
            features['numDefenders'] = len(defenders)
            dist = 0
            if len(defenders) > 0:
                self.chased = True
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
                dist = min(dists)
            features['defenderDistance'] = 6 - dist

        else:
            self.chased = False
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
            features['numInvaders'] = len(invaders)
            if len(invaders) > 0:
                features['invaderDistance'] = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])

        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1, 'distanceToStart': -200, 'stop': -5000, 'reverse': -5, 'distanceToFood': -5,
                'numFood': -27, 'numInvaders': -10, 'invaderDistance': -20, 'numDefenders': -20, 'defenderDistance': 10}

    def getEnemyPos(self, gameState):
        enemies = []
        for enemy in self.getOpponents(gameState):
            # print(enemy)
            pos = gameState.getAgentPosition(enemy)
            if pos != None:
                enemies.append((enemy, pos))
        return enemies

    def enemyDist(self, gameState):
        pos = self.getEnemyPos(gameState)
        minDist = None
        if len(pos) > 0:
            minDist = float('inf')
            myPos = gameState.getAgentPosition(self.index)
            for i, p in pos:
                dist = self.getMazeDistance(p, myPos)
                if dist < minDist:
                    minDist = dist
        return minDist

class GeneralAgent2(GeneralAgent):

    def registerInitialState(self, gameState):
        self.start_pos = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.chased = False  # if pacman is being chased while on offense
        self.capsules = len(self.getCapsules(gameState))
        self.food = len(self.getFood(gameState).asList())
        self.scared = 0

    def isChased(self, gameState):
        """Return true if agent is on offense and has an opponent close to them, false otherwise"""
        if not gameState.getAgentState(self.index).isPacman:
            return False

        if self.scared > 0:
            return False

        myPos = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None and self.getMazeDistance(myPos, a.getPosition()) <= 5]
        return len(defenders) > 0  # chased if there are defenders, not chased if no defenders

    def chooseAction(self, gameState):
        self.chased = self.isChased(gameState)
        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, self.index, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start_pos, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction
        c = random.choice(bestActions)

        if len(self.getCapsules(gameState)) < self.capsules:  # if ate a capsule
            self.capsules -= 1
            self.scared = 30

        self.scared = max(self.scared - 1, 0)
        return c

    def getSuccessor(self, gameState, idx, action):
        """Finds the next successor which is a grid position (location tuple)."""
        successor = gameState.generateSuccessor(idx, action)
        pos = successor.getAgentState(idx).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(idx, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """Computes a linear combination of features and feature weights"""
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """Returns a counter of features for the state"""
        features = util.Counter()
        for f in self.getWeights(gameState, action).keys():
            features[f] = 0

        successor = self.getSuccessor(gameState, self.index, action)
        features['successorScore'] = self.getScore(successor)

        # distance to start after taking action
        myPos = successor.getAgentState(self.index).getPosition()

        if action == Directions.STOP:
            features['stop'] = 1

        if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
            features['reverse'] = 1

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        if gameState.getAgentState(self.index).isPacman:  # agent was on offense
            if self.isChased(successor) == False:  # on offense, not chased:
                foodList = self.getFood(successor).asList()
                features['numFood'] = len(foodList)

                if len(foodList) < len(self.getFood(gameState).asList()): # if eating food, don't penalize dist to nearest food
                    features['distanceToFood'] = 0
                elif len(foodList) > 0:
                    features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in foodList])

            if self.scared == 0:  # on offense, opponents are not scared:
                defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
                features['numDefenders'] = len(defenders)

                if self.isChased(successor) == True:  # if there are defenders chasing (if len(defenders) > 0)
                    features['defenderDistance'] = min([self.getMazeDistance(myPos, a.getPosition()) for a in defenders])
                else:  # return a far distance (6)
                    features['defenderDistance'] = 6

            if self.isChased(successor) == True:
                dist = self.getMazeDistance(myPos, self.start_pos)
                features['distanceToStart'] = dist

            elif self.isChased(gameState) == True:
                features['distanceToStart'] = 500

        else:  # agent was on defense
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
            features['numInvaders'] = len(invaders)
            if len(invaders) > 0:
                features['invaderDistance'] = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])

            foodList = self.getFood(successor).asList()

            if util.flipCoin(0.15):  # with some small probability, move closer to the other side to go on offense
                #print('directed move')
                features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in foodList])

        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 500, 'distanceToStart': -50, 'stop': -5000, 'reverse': -5, 'distanceToFood': -5,
          'numFood': -27, 'numInvaders': -10, 'invaderDistance': -20, 'numDefenders': -20, 'defenderDistance': 100}


class DefenseToOffenseAgent(GeneralAgent2):

    def getFeatures(self, gameState, action):
        """Returns a counter of features for the state"""
        features = util.Counter()
        for f in self.getWeights(gameState, action).keys():
            features[f] = 0

        successor = self.getSuccessor(gameState, self.index, action)
        features['successorScore'] = self.getScore(successor)

        # distance to start after taking action
        myPos = successor.getAgentState(self.index).getPosition()

        if self.isChased(successor) == True:
            dist = self.getMazeDistance(myPos, self.start_pos)
            features['distanceToStart'] = dist
        elif self.isChased(gameState) == True:
            features['distanceToStart'] = 500

        if action == Directions.STOP:
            features['stop'] = 1

        if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
            features['reverse'] = 1

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        if self.isChased(successor) == False:  # on offense, not chased:
            foodList = self.getFood(successor).asList()
            features['numFood'] = len(foodList)

            if len(foodList) < len(self.getFood(gameState).asList()): # if eating food, don't penalize dist to nearest food
                features['distanceToFood'] = 0
            elif len(foodList) > 0:
                features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in foodList])

        if successor.getAgentState(self.index).isPacman and self.scared == 0:  # on offense, opponents are not scared:
            defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
            features['numDefenders'] = len(defenders)

            if self.isChased(successor) == True:  # if there are defenders chasing (if len(defenders) > 0)
                features['defenderDistance'] = min([self.getMazeDistance(myPos, a.getPosition()) for a in defenders])
            else:  # return a far distance (6)
                features['defenderDistance'] = 6

        if not successor.getAgentState(self.index).isPacman:
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
            features['numInvaders'] = len(invaders)
            if len(invaders) > 0:
                features['invaderDistance'] = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])

        return features


class Node:
  
    def __init__(self, state, action, parent, index, first_act, opponents):
        # total = wins;  score = total / visits
        self.idx = index
        self.state = state
        self.act = action  # move from parent to node
        self.parent = parent
        self.children = []
        self.total = 0
        self.visits = 0
        self.depth = 0
        self.samples = []
        self.first_act = first_act
        self.opponents = opponents

        if parent != None:
            self.depth = parent.depth + 1

    def is_leaf_node(self):
        return len(self.children) == 0

    def select_node(self):
        """Choose a node using the selection policy formula"""
        c = 4000
        lis = []
        for n in self.children:
            if n.visits == 0:
                s = float('inf')
            else:
                s = (n.total / n.visits) + (c * ((math.log(self.visits) / n.visits) ** 0.5))
            lis.append((n, s))

        v = max([e[1] for e in lis])
        best = [nd for nd, val in lis if val == v]
        child = random.choice(best)
        return child

    def all_moves_in_state(self, state, indexes):
        if indexes == []:
            return [state]
        else:
            state_lis = []
            idx = indexes[0]
            for act in state.getLegalActions(idx):
                successor = state.generateSuccessor(idx, act)
                state_lis += self.all_moves_in_state(successor, indexes[1:])
            return state_lis

    def expand_node(self):
        if self.depth > 7:
            return self

        state = self.state
        if self.parent == None:  # is the root node
            for act in state.getLegalActions(self.idx):
                if self.act != Directions.REVERSE[self.state.getAgentState(self.idx).configuration.direction]:
                    next_state = state.generateSuccessor(self.idx, act)

                    opp_indexes = [i for i in self.opponents if state.getAgentState(i).getPosition() != None]
                    for successor_state in self.all_moves_in_state(next_state, opp_indexes):
                        self.children.append(Node(successor_state, act, self, self.idx, act, self.opponents))

        else:
            for act in state.getLegalActions(self.idx):
                next_state = state.generateSuccessor(self.idx, act)

                opp_indexes = [i for i in self.opponents if state.getAgentState(i).getPosition() != None]
                for successor_state in self.all_moves_in_state(next_state, opp_indexes):
                    self.children.append(Node(successor_state, act, self, self.idx, self.first_act, self.opponents))

        return random.choice(self.children)  # random node

    def backpropogate(self, result):
        self.visits += 1
        self.total += result
        self.samples.append(result)

        if self.parent != None:
            self.parent.backpropogate(result)


class MCTSAgent(DefenseToOffenseAgent):

    def chooseAction(self, gameState):
        self.chased = self.isChased(gameState)
        if self.chased:  # being chased on offense / pacman
            act = self.MCTS(gameState)
        else:  # on defense / ghost
            # act = DefenseToOffenseAgent.chooseAction(self, gameState)
            act = GeneralAgent2.chooseAction(self, gameState)
        return act

    def median(self, ls):
        ls = sorted(ls)
        m = len(ls) // 2
        return (ls[m] + ls[-m]) / 2

    def bestState(self, node):
        """Best action"""
        counter = {}
        for n in node.children:
            sample = sum(n.samples) / len(n.samples)  # [sample] = [n.total / n.visits]
            if n.act in counter.keys():
                counter[n.act] += [sample]
            else:
                counter[n.act] = [sample]

        count = {}
        for act in counter.keys():  # or self.median(count[act]); or min(count[act])
            count[act] = sum(counter[act]) / len(counter[act]) # average for each list

        lis = count.items()
        # print([(a, count[a], counter[a]) for a in counter.keys()])
        v = max(count.values())
        best = [act for act, val in lis if val == v]
        return random.choice(best)

    def simulate(self, state, depth, first_act):
        if depth > 4:
            return self.evaluateMCTS(state, first_act)

        if state.getAgentState(self.index).getPosition() == self.start_pos:  # was caught
            return -200000

        if not self.isChased(state):  # if not being chased anymore
            if state.getAgentState(self.index).isPacman:  # if still on offense and not chased
                actions = state.getLegalActions(self.index)
                act = random.choice(actions)
                return self.evaluate(state, act)
            else:  # if back on defense, but wasn't caught
                return 100000

        # random action for the pacman
        actions = state.getLegalActions(self.index)
        act = random.choice(actions)
        successor = self.getSuccessor(state, self.index, act)

        # random action for the close opponents
        indexes = [i for i in self.getOpponents(state) if state.getAgentState(i).getPosition() != None]
        for idx in indexes:
            actions = successor.getLegalActions(idx)
            act = random.choice(actions)
            successor = self.getSuccessor(successor, idx, act)

        depth += 1
        return self.simulate(successor, depth, first_act)

    def simulation(self, node):
        """Run a simulation, return a resulting score"""
        return self.simulate(node.state, 0, node.first_act)

    def evaluateMCTS(self, gameState, first_act):
        """Computes a linear combination of features and feature weights for MCTS"""
        features = self.getFeaturesOffenseMCTS(gameState, first_act)
        weights = self.getWeightsOffenseMCTS()
        return features * weights

    def getFeaturesOffenseMCTS(self, gameState, first_act):
        """Returns a counter of features for the state"""
        features = util.Counter()
        for f in self.getWeightsOffenseMCTS().keys():
            features[f] = 0

        features['stateScore'] = self.getScore(gameState)

        myPos = gameState.getAgentState(self.index).getPosition()

        if self.isChased(gameState) == True:  # if chased on offense
            dist = self.getMazeDistance(myPos, self.start_pos)
            features['distanceToStart'] = dist

        if first_act == Directions.STOP:
            features['stop'] = 1

        if first_act == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
            features['reverse'] = 1

        if gameState.getAgentState(self.index).isPacman and self.isChased(gameState) == False:  # on offense, not chased
            foodList = self.getFood(gameState).asList()
            features['numFood'] = len(foodList)
            if len(foodList) > 0:
                features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in foodList])

        if gameState.getAgentState(self.index).isPacman and self.scared == 0:  # on offense, opponents are not scared
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
            features['numDefenders'] = len(defenders)

            dist = 6
            if self.isChased(gameState) == True:  # if there are defenders chasing / if len(defenders > 0)
                dist = min([self.getMazeDistance(myPos, a.getPosition()) for a in defenders])
                #print('f', dist)
                features['defenderDistance'] = dist
        return features

    def getWeightsOffenseMCTS(self):
        return {'stateScore': 5000, 'distanceToStart': -400, 'stop': -100000, 'reverse': -2, 'distanceToFood': -5,
          'numFood': -25, 'numInvaders': -10, 'invaderDistance': -20, 'numDefenders': -20, 'defenderDistance': 150}

    def MCTS(self, state):
        start = time.time()
        root = Node(state, None, None, self.index, None, self.getOpponents(state))
        root.expand_node()

        while time.time() - start < 0.5:  # 0.9 seconds
            n = root
            while not n.is_leaf_node():
                n = n.select_node()
            if n.visits != 0:  # if leaf node not visited yet, then expand it
                n = n.expand_node()
            result = self.simulation(n)
            n.backpropogate(result)
        return self.bestState(root)


class QLearningAgent(GeneralAgent):

    def __init__(self, gameState):
        # super(QLearningAgent, self).__init__(index)
        GeneralAgent.__init__(self, gameState)
        self.agent = GeneralAgent(self.index)
        self.q_value = util.Counter()
        self.score = 0
        self.lastState = []
        self.lastAction = []
        self.gamma = 0.9 # reward discount factor
        self.alpha = 0.9 # learning rate
        self.numTraining = 5
        # Count the number of games we have played
        self.episodes = 0

    def getWeights(self, gameState, action):
        return {'successorScore': 1, 'distanceToStart': -200, 'stop': -5000, 'reverse': -50, 'distanceToFood': -50,
                'numFood': -27, 'numInvaders': -10, 'invaderDistance': -20, 'numDefenders': -20, 'defenderDistance': 10}

    def updateQ(self, state, action, qmax):
        reward = GeneralAgent.evaluate(self, state, action)
        # print(reward)
        q = self.getQValue(state, action)
        self.q_value[(state, action)] = q + self.alpha * (reward + self.gamma * qmax - q)

    def getQValue(self, state, action):
        return self.q_value[(state, action)]

    def getMaxQ(self, state):
        q_dict = {}
        q_list = []
        for actions in state.getLegalActions(self.index):
            q = self.getQValue(state, actions)
            q_dict[actions] = q
            q_list.append(q)
        if len(q_list) == 0:
            return 0

        max_key = max(q_dict, key = q_dict.get)
        return max_key, max(q_list)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # update Q-value
        if len(self.lastState) > 0:
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]
            max_q = self.getMaxQ(gameState)[1]
            self.updateQ(last_state, last_action, max_q)
            action = self.getMaxQ(gameState)[0]
            # print(action)

        if util.flipCoin(0.5):
            action = random.choice(actions)
        else:
            action = self.learn(gameState)

        # update attributes
        self.score = gameState.getScore()
        self.lastState.append(gameState)
        self.lastAction.append(action)
        return action

    def learn(self, state):
        legal = state.getLegalActions(self.index)
        if self.episodes * 1.0/ self.numTraining <0.5:
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)
            if len(self.lastAction) > 0:
                last_action = self.lastAction[-1]
                if GeneralAgent.enemyDist(self, state) > 2:
                    if (Directions.REVERSE[last_action] in legal) and len(legal)>1:
                        legal.remove(Directions.REVERSE[last_action])
        qq = util.Counter()
        for action in legal:
          qq[action] = self.getQValue(state, action)
        return qq.argMax()


class ApproximateQLearningAgent(QLearningAgent):

    def __init__(self, gameState):
        QLearningAgent.__init__(self, gameState)
        self.weights = util.Counter()

    def getQValue(self, state, action):
        return self.evaluate(state, action)

    def updateQ(self, state, action, qmax):
        """Update weights based on transition"""
        reward = GeneralAgent.evaluate(self, state, action)
        q = self.getQValue(state, action)
        self.q_value[(state, action)] = q + self.alpha * (reward + self.gamma * qmax - q)

        diff = (reward + self.alpha * self.evaluate(state, action)) - self.getQValue(state, action)
        features = self.getFeatures(state, action)
        for i in features:
            self.weights[i] = self.weights[i] + self.alpha * diff * features[i]
