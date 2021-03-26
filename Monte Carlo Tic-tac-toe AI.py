# -*- coding: utf-8 -*-

# -- Sheet --

#Imports
import random as rand
import numpy as np
from copy import deepcopy

#Constant c used in calculation for UCT, weight between exploration and exploitation
uct_constant = 0.8

#Create a class for nodes, with parents, board, player and children. As well as values for calculating UCT
class Node:
    def __init__(self,parent,board,player):
        self.parent = parent
        self.board = board
        self.player = player
        self.children = []
        self.q = 0
        self.n = 0
        self.uct = 0

#Method for creating an empty board for 3x3 tic tac toe
def create_board():
    return np.array([' ',' ',' ',' ',' ',' ',' ',' ',' '])

#Method for creating an empty tree with board as the root
def create_tree(board):
    root = Node(parent = None, board = board, player = 'O')
    tree = [root]
    return tree

#Method for checking if the current state is a terminal state
def check_terminal(node):
    board = node.board
    if board[0] == board[3] == board[6] and board[0] != ' ':
        #print('Player '+ board[0] + ' won the game')
        if board[0] == 'X':
            return 1
        if board[0] == 'O':
            return -1
    elif board[0] == board[1] == board[2] and board[0] != ' ':
        #print('Player ' + board[0] + ' won the game')
        if board[0] == 'X':
            return 1
        if board[0] == 'O':
            return -1
    elif board[0] == board[4] == board[8]  and board[0] != ' ':
        #print('Player ' + board[0] + ' won the game')
        if board[0] == 'X':
            return 1
        if board[0] == 'O':
            return -1
    elif board[1] == board[4] == board[7] and board[1] != ' ':
        #print('Player ' + board[1] + ' won the game')
        if board[1] == 'X':
            return 1
        if board[1] == 'O':
            return -1
    elif board[2] == board[5] == board[8] and board[2] != ' ':
        #print('Player '+ board[2] + ' won the game')
        if board[2] == 'X':
            return 1
        if board[2] == 'O':
            return -1
    elif board[3] == board[4] == board[5] and board[3] != ' ':
        #print('Player '+ board[3] + ' won the game')
        if board[3] == 'X':
            return 1
        if board[3] == 'O':
            return -1
    elif board[6] == board[7] == board[8] and board[6] != ' ':
        #print('Player ' +board[6] + ' won the game')
        if board[6] == 'X':
            return 1
        if board[6] == 'O':
            return -1
    elif board[6] == board[4] == board[2] and board[6] != ' ':
        #print('Player ' + board[6] + ' won the game')
        if board[6] == 'X':
            return 1
        if board[6] == 'O':
            return -1
    elif board[0] != ' ' and board[1] != ' ' and board[2] != ' ' and board[3] != ' ' and board[4] != ' ' and board[5] != ' ' and board[6] != ' ' and board[7] != ' ' and board[8] != ' ':  
        #print('Draw')
        return 0
    else:
        return 10

#Selection method. Policy is to select the node with the highest UCT. 
def selection(tree):
    selected_node = tree[0]
    max_uct = -1000
    for node in tree:
        if not node.children:
            if node.uct > max_uct:
                max_uct = node.uct
                selected_node = node
    return selected_node

#Tic tac toe action for inserting mark in the selected square
def action(node,tree,i,player):
    if node.board[i] == ' ' and player == 'X':
        board = deepcopy(node.board)
        board[i] = player
        #Create a new node with parent node's board + X in the selected square
        node1 = Node(parent = node, board = board, player = 'O')
        node.children.append(node1)
        tree.append(node1)
        node1.parent = node

    if node.board[i] == ' ' and player == 'O':
        board = deepcopy(node.board)
        board[i] = player
        #Create a new node with parent node's board + O in the selected square
        node1 = Node(parent = node, board = board, player = 'X')
        node.children.append(node1)
        tree.append(node1)
        node1.parent = node

#Method expansion, expands all children to the selected node
def expansion(node,tree):
    if check_terminal(node) != 10:
        return node
    
    else:
        #Expand all possible children
        for i in range(0,len(node.board)):
            action(node,tree,i,node.player)

        #Select the first child that has not been selected before
        for node_temp in node.children:
            if node_temp.n == 0:
                simulation_node = node_temp
                return simulation_node

#Method for simulation
def simulation(node):

    tempNode = deepcopy(node)
    
    #While a terminal state is not reached
    while check_terminal(tempNode) == 10:
        
        #Randomized square for the next action
        i = rand.randint(0,8)

        #If square i empty and current player X, insert X in square i and switch player to O. 
        if tempNode.board[i] == ' ' and tempNode.player == 'X':
            tempNode.board[i] = 'X'
            tempNode.player = 'O'
        
        #If square i empty and current player O, insert O in square i and switch player to X. 
        if tempNode.board[i] == ' ' and tempNode.player == 'O':
            tempNode.board[i] = 'O'
            tempNode.player = 'X'
    
    #Check if the new state is a terminal node
    return check_terminal(tempNode)

#Update the value for uct
def update_uct(node):
    node.uct = (node.q/node.n) + uct_constant * np.sqrt(np.log(node.parent.n)/node.n)

#Method for backpropagataion
def backpropagation(node, value):

    done = False

    #Update all parent nodes up to root node
    while done == False:
        node.n += 1
        node.q += value

        if node.parent == None:
            node.uct = (node.q/node.n)
            done = True

        else:
            if node.parent.n == 0: 
                node.parent.n = 1
            node.uct = (node.q/node.n) + uct_constant * np.sqrt(np.log(node.parent.n)/node.n)
            node = node.parent

#Calculate the best action from a certain state. Based on the highest UCT value of all children nodes
def best_action(node):
    best_action = node
    max_uct = -1000
    for child in node.children:
        if child.uct > max_uct:
            best_action = child
            max_uct = child.uct
            
    return best_action.board

#Run 300 iterations of the 4 steps in the MCT algorithm on a board. Returns the best action from the current state.
def play(board):
    tree = create_tree(board)
    for i in range(100):
        selected_node = selection(tree)
        expansion_node = expansion(selected_node, tree)
        value = simulation(expansion_node)
        backpropagation(expansion_node,value)
    return best_action(tree[0])

#Method for printing the board.
def printBoard(board):
    print(board[0] + '|' + board[1] + '|' + board[2])
    print('-+-+-')
    print(board[3] + '|' + board[4] + '|' + board[5])
    print('-+-+-')
    print(board[6] + '|' + board[7] + '|' + board[8])
    print()
    print('----------------------------------------')
    print()

#Method for initializing a board
def start_game():
    board = create_board()
    printBoard(board)
    return board

#Method for making moves, input a square and board and the square in that board gets updated with a X.    
def make_move(i,board):
    if board[i] == ' ':
        board[i] = 'X'
    else: 
        print('Invalid move')
        printBoard(board)
        return board
    if check_terminal(create_tree(board)[0]) == 1:
        print("X wins")
        printBoard(board)
        return board
    if check_terminal(create_tree(board)[0]) == -1:
        print("O wins")
        printBoard(board)
        return board
    if check_terminal(create_tree(board)[0]) == 0:
        print("Draw")
        printBoard(board)
        return board
    new_board = play(board)
    printBoard(new_board)
    return new_board

board = start_game()
board = make_move(1,board)

board = make_move(4,board)

board = make_move(7,board)

# -- Sheet 2 --

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class policy(object):
    def __init__(self):
        self.tree = {}
        pass

class VanilaMCTS(object):
    def __init__(self, n_iterations=50, depth=15, exploration_constant=5.0, n_rows = 3, tree = None, win_mark=3, game_board=None, player=None):
        self.n_iterations = n_iterations
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.total_n = 0

        self.leaf_node_id = None

        self.n_rows = n_rows
        self.win_mark = win_mark

        if tree == None:
            self.tree = self._set_tictactoe(game_board, player)
        
        else:
            self.tree = tree

    def _set_tictactoe(self, game_board, player):
        root_id = (0,)
        tree = {root_id: {'state': game_board,
                          'player': player,
                          'child': [],
                          'parent': None,
                          'n': 0,
                          'w': 0,
                          'q': None}}
        return tree

    def selection(self):
        # select leaf node which have maximum uct value
        # in:
        # - tree
        # out:
        # - leaf node id (node to expand)
        # - depth (depth of node root=0)
        
        leaf_node_found = False
        leaf_node_id = (0,) # root node id
        # print('-------- selection ----------')

        while not leaf_node_found:
            node_id = leaf_node_id
            n_child = len(self.tree[node_id]['child'])
            # print('n_child: ', n_child)

            if n_child == 0:
                leaf_node_id = node_id
                leaf_node_found = True
            else:
                maximum_uct_value = -100.0
                for i in range(n_child):
                    action = self.tree[node_id]['child'][i]

                    # print('leaf_node_id', leaf_node_id)
                    child_id = node_id + (action,)
                    w = self.tree[child_id]['w']
                    n = self.tree[child_id]['n']
                    total_n = self.total_n
                    # parent_id = self.tree[node_id]['parent']
                    # if parent_id == None:
                    #     total_n = 1
                    # else:
                    #     total_n = self.tree[parent_id]['n']

                    if n == 0:
                        n = 1e-4
                    exploitation_value = w / n
                    exploration_value  = np.sqrt(np.log(total_n)/n)
                    uct_value = exploitation_value + self.exploration_constant * exploration_value

                    if uct_value > maximum_uct_value:
                        maximum_uct_value = uct_value
                        leaf_node_id = child_id

        depth = len(leaf_node_id) # as node_id records selected action set
        # print('leaf node found: ', leaf_node_found)
        # print('n_child: ', n_child)
        # print('selected leaf node: ')
        # print(self.tree[leaf_node_id])
        return leaf_node_id, depth

    def expansion(self, leaf_node_id):
        
        # create all possible outcomes from leaf node
        # in: tree, leaf_node
        # out: expanded tree (self.tree),
        #     randomly selected child node id (child_node_id)
        
        leaf_state = self.tree[leaf_node_id]['state']
        winner = self._is_terminal(leaf_state)
        possible_actions = self._get_valid_actions(leaf_state)

        child_node_id = leaf_node_id # default value
        if winner is None:
            #when leaf state is not terminal state
            childs = []
            for action_set in possible_actions:
                action, action_idx = action_set
                state = deepcopy(self.tree[leaf_node_id]['state'])
                current_player = self.tree[leaf_node_id]['player']

                if current_player == 'o':
                    next_turn = 'x'
                    state[action] = 1
                else:
                    next_turn = 'o'
                    state[action] = -1

                child_id = leaf_node_id + (action_idx, )
                childs.append(child_id)
                self.tree[child_id] = {'state': state,
                                       'player': next_turn,
                                       'child': [],
                                       'parent': leaf_node_id,
                                       'n': 0, 'w': 0, 'q':0}
                self.tree[leaf_node_id]['child'].append(action_idx)
            rand_idx = np.random.randint(low=0, high=len(childs), size=1)
            # print(rand_idx)
            # print('childs: ', childs)
            child_node_id = childs[rand_idx[0]]
        return child_node_id

    def _is_terminal(self, leaf_state):
        
        #check terminal
        #in: game state
        #out: who wins? ('o', 'x', 'draw', None)
        #     (None = game not ended)
        
        def __who_wins(sums, win_mark):
            if np.any(sums == win_mark):
                return 'o'
            if np.any(sums == -win_mark):
                return 'x'
            return None

        def __is_terminal_in_conv(leaf_state, win_mark):
            # check row/col
            for axis in range(2):
                sums = np.sum(leaf_state, axis=axis)
                result = __who_wins(sums, win_mark)
                if result is not None:
                    return result
            # check diagonal
            for order in [-1,1]:
                diags_sum = np.sum(np.diag(leaf_state[::order]))
                result = __who_wins(diags_sum, win_mark)
                if result is not None:
                    return result
            return None

        win_mark = self.win_mark
        n_rows_board = len(self.tree[(0,)]['state'])
        window_size = win_mark
        window_positions = range(n_rows_board - win_mark + 1)

        for row in window_positions:
            for col in window_positions:
                window = leaf_state[row:row+window_size, col:col+window_size]
                winner = __is_terminal_in_conv(window, win_mark)
                if winner is not None:
                    return winner

        if not np.any(leaf_state == 0):
            #no more action i can do
            return 'draw'
        return None

    def _get_valid_actions(self, leaf_state):
        #return all possible action in current leaf state
        #in:
        #- leaf_state
        #out:
        #- set of possible actions ((row,col), action_idx)
        
        actions = []
        count = 0
        state_size = len(leaf_state)

        for i in range(state_size):
            for j in range(state_size):
                if leaf_state[i][j] == 0:
                    actions.append([(i, j), count])
                count += 1
        return actions

    def simulation(self, child_node_id):
        #simulate game from child node's state until it reaches the resulting state of the game.
        #in:
        #- child node id (randomly selected child node id from `expansion`)
        #out:
        #- winner ('o', 'x', 'draw')
        self.total_n += 1
        state = deepcopy(self.tree[child_node_id]['state'])
        previous_player = deepcopy(self.tree[child_node_id]['player'])
        anybody_win = False

        while not anybody_win:
            winner = self._is_terminal(state)
            if winner is not None:
                # print('state')
                # print(state)
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(4.5,4.56))
                # plt.pcolormesh(state, alpha=0.6, cmap='RdBu_r')
                # plt.grid()
                # plt.axis('equal')
                # plt.gca().invert_yaxis()
                # plt.colorbar()
                # plt.title('winner = ' + winner + ' (o:1, x:-1)')
                # plt.show()
                anybody_win = True
            else:
                possible_actions = self._get_valid_actions(state)
                # randomly choose action for simulation (= random rollout policy)
                rand_idx = np.random.randint(low=0, high=len(possible_actions), size=1)[0]
                action, _ = possible_actions[rand_idx]

                if previous_player == 'o':
                    current_player = 'x'
                    state[action] = -1
                else:
                    current_player = 'o'
                    state[action] = 1

                previous_player = current_player
        return winner

    def backprop(self, child_node_id, winner):
        player = deepcopy(self.tree[(0,)]['player'])

        if winner == 'draw':
            reward = 0
        elif winner == player:
            reward = 1
        else:
            reward = -1

        finish_backprob = False
        node_id = child_node_id
        while not finish_backprob:
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward
            self.tree[node_id]['q'] = self.tree[node_id]['w'] / self.tree[node_id]['n']
            parent_id = self.tree[node_id]['parent']
            if parent_id == (0,):
                self.tree[parent_id]['n'] += 1
                self.tree[parent_id]['w'] += reward
                self.tree[parent_id]['q'] = self.tree[parent_id]['w'] / self.tree[parent_id]['n']
                finish_backprob = True
            else:
                node_id = parent_id

    def solve(self):
        for i in range(self.n_iterations):
            leaf_node_id, depth_searched = self.selection()
            child_node_id = self.expansion(leaf_node_id)
            winner = self.simulation(child_node_id)
            self.backprop(child_node_id, winner)

            # print('----------------------------')
            # print('iter: %d, depth: %d' % (i, depth_searched))
            # print('leaf_node_id: ', leaf_node_id)
            # print('child_node_id: ', child_node_id)
            # print('child node: ')
            # print(self.tree[child_node_id])
            if depth_searched > self.depth:
                break

        # SELECT BEST ACTION
        current_state_node_id = (0,)
        action_candidates = self.tree[current_state_node_id]['child']
        # qs = [self.tree[(0,)+(a,)]['q'] for a in action_candidates]
        best_q = -100
        for a in action_candidates:
            q = self.tree[(0,)+(a,)]['q']
            if q > best_q:
                best_q = q
                best_action = a

#for test
#if __name__ == '__main__':
    #mcts = VanilaMCTS(n_iterations=100, depth=10, exploration_constant=1.4, tree = None, n_rows=3, win_mark=3)
    #leaf_node_id, depth = mcts.selection()
    #child_node_id = mcts.expansion(leaf_node_id)

    #print('child node id = ', child_node_id)
    #print(' [*] simulation ...')
    #winner = mcts.simulation(child_node_id)
    #print(' winner', winner)
    #mcts.backprop(child_node_id, winner)
    #best_action, max_q = mcts.solve()
    #print('best action= ', best_action, ' max_q= ', max_q)

