# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        oldPackman = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        scaredTime = min(newScaredTimes)
        score = 0
        ghostDistances = []
        foodDistances = []
        oldFoodDistances = []
        if(scaredTime > 4):
            score = 10
            for food in newFood.asList():
                foodDistances.append(util.manhattanDistance(food, newPos))
            minFoodDistance = min(foodDistances)
            if(len(foodDistances) > 0):
                minFoodDistance = min(foodDistances)
                score = score + 20/minFoodDistance
        for ghost in newGhostStates:
            ghostDistances.append(util.manhattanDistance(ghost.getPosition(),successorGameState.getPacmanPosition()))
        if(len(ghostDistances) > 0 and min(ghostDistances) <=1):
            return -9999
        else:
            for food in newFood.asList():
                foodDistances.append(util.manhattanDistance(food, newPos))
            if(len(foodDistances) > 0):
                minFoodDistance = min(foodDistances)
                score = score + 20/minFoodDistance
            if(currentGameState.getNumFood() > successorGameState.getNumFood()):
                score += 100
            if(len(ghostDistances) > 0):
                score = score - 10 / min(ghostDistances)
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def maxAgent(state,depth):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(0)
            move = Directions.STOP
            maxValue = -99999
            for action in legalActions:
                value = minAgent(state.generateSuccessor(0,action),depth, 1)
                if maxValue < value:
                    maxValue = value
                    move = action
            if(depth == self.depth):
                return move
            return maxValue


        def minAgent(state, depth, agent):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(agent)
            values = []
            for action in legalActions:
                nextState = state.generateSuccessor(agent,action)
                if agent < state.getNumAgents()-1:
                    values.append(minAgent(nextState, depth, agent+1))
                else:
                    values.append(maxAgent(nextState, depth-1))
            return min(values)

        return maxAgent(gameState, self.depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxAgent(state,depth, agent, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            move = Directions.STOP
            maxValue = -99999
            for action in state.getLegalActions(agent):
                value = minAgent(state.generateSuccessor(agent,action),depth, 1, alpha, beta)
                if maxValue < value:
                    maxValue = value
                    move = action
                if maxValue > beta:
                    break
                else:
                    alpha = max(alpha,maxValue)
            if(depth == self.depth):
                return move
            return maxValue


        def minAgent(state, depth, agent, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            minValue = 99999
            for action in state.getLegalActions(agent):
                nextState = state.generateSuccessor(agent,action)
                if agent < state.getNumAgents()-1:
                    value = minAgent(nextState, depth, agent+1, alpha, beta)
                    minValue = min(minValue, value)
                else:
                    minValue = min(minValue, maxAgent(nextState, depth-1, 0, alpha, beta))
                if minValue < alpha:
                    break
                else:
                    beta = min(beta, minValue)
            return minValue

        return maxAgent(gameState, self.depth, 0, float("-inf"), float("inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxAgent(state,depth):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(0)
            move = Directions.STOP
            maxValue = -99999
            for action in legalActions:
                value = minAgent(state.generateSuccessor(0,action),depth, 1)
                if maxValue < value:
                    maxValue = value
                    move = action
            if(depth == self.depth):
                return move
            return maxValue


        def minAgent(state, depth, agent):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(agent)
            value = 0
            chance = 1.0/len(legalActions)
            for action in legalActions:
                nextState = state.generateSuccessor(agent,action)
                if agent < state.getNumAgents()-1:
                    value += minAgent(nextState, depth, agent+1) * chance
                else:
                    value += maxAgent(nextState, depth-1) * chance
            return value

        return maxAgent(gameState, self.depth)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = []
    score = currentGameState.getScore()
    ghostDistances = []
    foodDistances = []
    nonScaryGhosts = 0
    for ghost in newGhostStates:
        if ghost.scaredTimer == 0:
             nonScaryGhosts += 1
        newScaredTimes.append(ghost.scaredTimer)
        ghostDistances.append(util.manhattanDistance(ghost.getPosition(),currentGameState.getPacmanPosition()))
    scaredTime = min(newScaredTimes)
    if(scaredTime > 5):
        score += 10
        for food in newFood.asList():
            foodDistances.append(util.manhattanDistance(food, newPos))
        if(len(foodDistances) > 0):
            minFoodDistance = min(foodDistances)
            score += 10/minFoodDistance
        return score
    if(len(ghostDistances) > 0 and min(ghostDistances) <=1):
        return -9999
    else:
        for food in newFood.asList():
            foodDistances.append(util.manhattanDistance(food, newPos))
        if(len(foodDistances) > 0):
            minFoodDistance = min(foodDistances)
            score = score + 10/minFoodDistance
            if(minFoodDistance > min(ghostDistances)+5):
                score = score - 2
        if(len(ghostDistances) > 0):
            score = score - 10 / min(ghostDistances) - 20 * nonScaryGhosts
    return score

# Abbreviation
better = betterEvaluationFunction

