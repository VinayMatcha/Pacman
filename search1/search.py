# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    fringeCalculated = util.Stack()     # Using stack as we are implementing Depth first Search
    finalPath = []                      # List of actions to be returned by this function
    visited = []                        # List to keep track of the visited states
    fringeCalculated.push([(problem.getStartState() , None, 0)])   # Pushing the start state along with action and cost as a tuple into the stack
    while not fringeCalculated.isEmpty():             #We will run the algorithm until the stack is empty or we will return the path we find Goal State
        consideredPath = fringeCalculated.pop()       # Pop the list of last list states from the stack
        state = consideredPath[len(consideredPath)-1] # Get the last tuple(i.e) last node from the path to get it's successors
        if problem.isGoalState(state[0]):             # Checking whether the state is a goal state and sending the final path if it is a goal state
            for path in consideredPath[1:]:
                finalPath.append(path[1])              # adding only directions
            return finalPath
        """Check whether the state is visited or not. Added in order to prevent infinite search. If not make we will now mark it as visited and we will get 
        succeessors of the present node and we will append new paths upto these successors into the fringe."""
        if state[0] not in visited:
            visited.append(state[0])
            for child in problem.getSuccessors(state[0]):
                temp_list = consideredPath[:]
                temp_list.append(child)          #Append the child node tuple into the state which we are extracting
                fringeCalculated.push(temp_list) #Appening the path list which we will check in next loops
    return fringeCalculated.pop()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringeCalculated = util.Queue()     # Using Queue as we are implementing breadth first Search
    finalPath = []                      # List of actions to be returned by this function
    visited = []                        # List to keep track of the visited states
    fringeCalculated.push([(problem.getStartState() , None, 0)])   # Pushing the start state along with action and cost as a tuple into the Queue
    while not fringeCalculated.isEmpty():            # We will run the algorithm until the queue is empty or we will return the path we find Goal State
        consideredPath = fringeCalculated.pop()      # Pop the list of states from the queue
        state = consideredPath[len(consideredPath)-1]   # Get the last tuple in the state list
        if problem.isGoalState(state[0]):  # Checking whether the state is a goal state and sending the final path if it is a goal state
            for path in consideredPath[1:]:
                finalPath.append(path[1]) # adding only directions
            return finalPath
        """Check whether the state is visited or not. Added in order to prevent infinite search. If not make we will now mark it as visited and we will get 
        succeessors of the present node and we will append new paths upto these successors into the fringe."""
        if state[0] not in visited:
            visited.append(state[0])  # If not make the state as visited
            for child in problem.getSuccessors(state[0]):  # Get the successors of the current state
                path = consideredPath[:]
                path.append(child)                  # Append the child tuple into the state list popped
                fringeCalculated.push(path)         # Appening the path list which we will check in next loops
    return finalPath




def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringeCalculated = util.PriorityQueue() # Using priority queue as we need get the node of least cost
    finalPath = []                          # List of actions to be returned by this function
    visited = []                            # List to keep track of the visited states
    fringeCalculated.push([(problem.getStartState() , None, 0)],0)   # Pushing the start state along with action and cost as a tuple into the priority Queue
    while not fringeCalculated.isEmpty():  # We will run the algorithm till the stack is empty
        consideredPath = fringeCalculated.pop()  # Pop the list of states from the stack
        state = consideredPath[len(consideredPath)-1]  # Get the last tuple in the state list
        if problem.isGoalState(state[0]):  # Check whether the state is a goal state
            for path in consideredPath[1:]:
                finalPath.append(path[1])  # adding only directions
            return finalPath
        """Check whether the state is visited or not. Added in order to prevent infinite search. If not make we will now mark it as visited and we will get 
        succeessors of the present node and we will append new paths upto these successors, with modified cost into the fringe."""
        if state[0] not in visited:
            visited.append(state[0])  # If not make the state as visited
            for child in problem.getSuccessors(state[0]):  # Get the successors of the current state
                temp_list = consideredPath[:]
                temp_list.append([child[0], child[1], child[2]+state[2]]) # Adding successor to the popped list with modified cost
                fringeCalculated.push(temp_list, child[2]+state[2])      #Appening the path list which we will check in next loops
    return finalPath

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringeCalculated = util.PriorityQueue() # Using priority queue as we need get the node of least cost + heauristic value
    finalPath = []                          # List of actions to be returned by this function
    visited = []                            # List to keep track of the visited states
    fringeCalculated.push([(problem.getStartState(), None, 0)],heuristic(problem.getStartState(),problem))
    while not fringeCalculated.isEmpty():
        consideredPath = fringeCalculated.pop()
        state = consideredPath[len(consideredPath)-1]
        if problem.isGoalState(state[0]):
            for path in consideredPath[1:]:
                finalPath.append(path[1])
            return finalPath
        """Check whether the state is visited or not. Added in order to prevent infinite search. If not make we will now mark it as visited and we will get 
        succeessors of the present node and we will append new paths upto these successors, with modified cost and heauristic values into the fringe."""
        if state[0] not in visited:
            visited.append(state[0])
            for child in problem.getSuccessors(state[0]):  # Get the successors of the current state and adding them to the expanded paths list(fringe)
                totalCost = child[2] + state[2] + heuristic(child[0], problem)
                tempList = consideredPath[:]
                tempList.append([child[0],child[1], child[2] + state[2]]) # Adding successor to the popped list with modified cost
                fringeCalculated.push(tempList, totalCost)
    return finalPath


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
