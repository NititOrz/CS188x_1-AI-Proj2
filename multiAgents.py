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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        has_food_next = successorGameState.hasFood(newPos[0],newPos[1])
        ghostPositions = successorGameState.getGhostPositions()
        capsulePositions = currentGameState.getCapsules()

        if successorGameState.isWin():
          return float("inf") - 20

        score = successorGameState.getScore()
        foodList = newFood.asList()

        closestFoodDist = float('Inf')
        for foodPos in foodList:
          toFoddDist = manhattanDistance(foodPos,newPos)
          if  toFoddDist < closestFoodDist:
            closestFoodDist = toFoddDist

        closestGhostDist = float('Inf')
        for ghostPos in ghostPositions:
          distFromGhost = manhattanDistance(newPos,ghostPos)
          if distFromGhost < closestGhostDist:
            closestGhostDist = distFromGhost

        closestCapDist = float('Inf')
        for capPos in capsulePositions:
          distFromCap = manhattanDistance(newPos,capPos)
          if distFromCap < closestCapDist:
            closestCapDist = distFromCap

        if action == Directions.STOP:
          score -= 3

        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
          score += 100
        if closestGhostDist <= 3:
          score += closestGhostDist/(closestCapDist+1)*100
        else:
          score += 103
        score -= closestFoodDist

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
        def MinValue(gameState, depth, agentindex, numGhost):
          if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
          v = float('Inf')
          legalMoves = gameState.getLegalActions(agentindex)
          for action in legalMoves:
            nextState = gameState.generateSuccessor(agentindex, action)
            if agentindex == numGhost:
              v = min(v, MaxValue(nextState, depth - 1, numGhost))
            else:
              v = min(v, MinValue(nextState, depth, agentindex + 1, numGhost))
            
          return v

        def MaxValue(gameState, depth, numGhost):
          if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
          v = -float('Inf')
          legalMoves = gameState.getLegalActions(0)
          for action in legalMoves:
            nextState = gameState.generateSuccessor(0,action)
            v = max(v, MinValue(nextState, depth , 1, numGhost))
          return v

        legalMoves = gameState.getLegalActions()
        bestAction = Directions.STOP
        numGhosts = gameState.getNumAgents() - 1
        for action in legalMoves:
          nextState = gameState.generateSuccessor(0, action)
          prevscore = score
          score = max(score, MinValue(nextState, self.depth, 1, numGhosts))
          if score > prevscore:
            bestaction = action
        return bestaction


        "Add more of your code here if you want to"

        util.raiseNotDefined()

# class AlphaBetaAgent(MultiAgentSearchAgent):
#     """
#       Your minimax agent with alpha-beta pruning (question 3)
#     """

#     def getAction(self, gameState):
#         """
#           Returns the minimax action using self.depth and self.evaluationFunction
#         """
#         "*** YOUR CODE HERE ***"

#         def MinValue(gameState, depth, agentindex, numGhost, alpha, beta):
#           if gameState.isWin() or gameState.isLose() or depth == 0:
#                 return self.evaluationFunction(gameState)
#           v = float('Inf')
#           legalMoves = gameState.getLegalActions(agentindex)
#           for action in legalMoves:
#             nextState = gameState.generateSuccessor(agentindex, action)
#             if agentindex == numGhost:
#               v = min(v, MaxValue(nextState, depth - 1, numGhost, alpha, beta))
#               if v < alpha:
#                 return v
#             else:
#               v = min(v, MinValue(nextState, depth, agentindex + 1, numGhost, alpha, beta))
#               if v <= alpha:
#                 return v
#             beta = min(beta, v)
#           return v

#         def MaxValue(gameState, depth, numGhost, alpha, beta):
#           if gameState.isWin() or gameState.isLose() or depth == 0:
#                 return self.evaluationFunction(gameState)
#           v = -float('Inf')
#           legalMoves = gameState.getLegalActions(0)
#           for action in legalMoves:
#             nextState = gameState.generateSuccessor(0,action)
#             v = max(v, MinValue(nextState, depth , 1, numGhost, alpha, beta))
#             if v > beta:
#               return v
#             alpha = max(alpha,v)
#           return v

#         alpha = -(float("inf"))
#         beta = float("inf")
#         legalMoves = gameState.getLegalActions()
#         bestAction = Directions.STOP
#         numGhosts = gameState.getNumAgents() - 1
#         score = -(float("inf"))
#         for action in legalMoves:
#           nextState = gameState.generateSuccessor(0, action)
#           prevscore = score
#           score = max(score, MinValue(nextState, self.depth, 1, numGhosts, alpha, beta))
#           if score > prevscore:
#             bestaction = action
#           if score > beta:
#             return bestaction
#           alpha = max(alpha, score)
#         return bestaction

#         util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_prune(self, gameState, depth, agentIndex, alpha, beta):
      # init the variables
      maxEval= float("-inf")

      # if this is a leaf node with no more actions, return the evaluation function at this state
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      # otherwise, for evert action, find the successor, and run the minimize function on it. when a value
      # is returned, check to see if it's a new max value (or if it's bigger than the minimizer's best, then prune)
      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)
        
        # run minimize (the minimize function will stack ghost responses)
        tempEval = self.min_prune(successor, depth, 1, alpha, beta)

        #prune
        if tempEval > beta:
          return tempEval

        if tempEval > maxEval:
          maxEval = tempEval
          maxAction = action

        #reassign alpha
        alpha = max(alpha, maxEval)

      # if this is the first depth, then we're trying to return an ACTION to take. otherwise, we're returning a number. This
      # could theoretically be a tuple with both, but i'm lazy.
      if depth == 1:
        return maxAction
      else:
        return maxEval



    def min_prune(self, gameState, depth, agentIndex, alpha, beta):
      minEval= float("inf")

      # we don't know how many ghosts there are, so we have to run minimize
      # on a general case based off the number of agents
      numAgents = gameState.getNumAgents()

      # if a leaf node, return the eval function!
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      # for every move possible by this ghost
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
      
        # if this is the last ghost to minimize
        if agentIndex == numAgents - 1:
          # if we are at our depth limit, return the eval function
          if depth == self.depth:
            tempEval = self.evaluationFunction(successor)
          else:
            #maximize!
            tempEval = self.max_prune(successor, depth+1, 0, alpha, beta)

        # pass this state on to the next ghost
        else:
          tempEval = self.min_prune(successor, depth, agentIndex+1, alpha, beta)

        #prune
        if tempEval < alpha:
          return tempEval
        if tempEval < minEval:
          minEval = tempEval
          minAction = action

        # new beta
        beta = min(beta, minEval)
      return minEval

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        maxAction = self.max_prune(gameState, 1, 0, float("-inf"), float("inf"))
        return maxAction

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

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

