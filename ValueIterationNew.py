import numpy as np
import math as  math
import matplotlib.pyplot as plt

class Dynamics:
    row = 5
    col = 5
    initial_state = np.zeros((row, col))
    policy = np.full((row, col),' ', dtype='U{}'.format(len('  ')))
    # print(initial_state)
    theta = 0.0001
    gamma = 0.25
    actions = ["au","ad","al","ar"]
    stochastic_prob = {"right_direction": 0.8, "veer_right":0.05, "veer_left":0.05, "break":0.1}
    obstacle_states = ["22","32"]
    water_states = ["42"]
    goal_state = ["44"]
    transition_probabilities = {'00': {'au': {'01': 0.05, '00': 0.95},
        'ad': {'10': 0.8, '01': 0.05, '00': 0.15},
        'ar': {'01': 0.8, '10': 0.05, '00': 0.15},
        'al': {'10': 0.05, '00': 0.95}},

'01': {'au': {'02': 0.05, '00': 0.05, '01': 0.9},
        'ad': {'11': 0.8, '02': 0.05, '00': 0.05, '01': 0.1},
        'ar': {'02': 0.8, '11': 0.05, '01': 0.15},
        'al': {'00': 0.8, '11': 0.05, '01': 0.15}},

'02': {'au': {'03': 0.05, '01': 0.05, '02': 0.9},
        'ad': {'12': 0.8, '03': 0.05, '01': 0.05, '02': 0.1},
        'ar': {'03': 0.8, '12': 0.05, '02': 0.15},
        'al': {'01': 0.8, '12': 0.05, '02': 0.15}},

'03': {'au': {'04': 0.05, '02': 0.05, '03': 0.9},
        'ad': {'13': 0.8, '04': 0.05, '02': 0.05, '03': 0.1},
        'ar': {'04': 0.8, '13': 0.05, '03': 0.15},
        'al': {'02': 0.8, '13': 0.05, '03': 0.15}},

'04': {'au': {'03': 0.05, '04': 0.95},
        'ad': {'14': 0.8, '03': 0.05, '04': 0.15,},
        'ar': {'14': 0.05, '04': 0.95},
        'al': {'03': 0.8, '14': 0.05, '04': 0.15}},

'10': {'au': {'00': 0.8, '11': 0.05, '10': 0.15},
        'ad': {'20': 0.8, '11': 0.05, '10': 0.15},
        'ar': {'11': 0.8, '00': 0.05, '20': 0.05, '10': 0.1},
        'al': {'10': 0.9, '00': 0.05, '20': 0.05}},

'11': {'au': {'01': 0.8, '12': 0.05, '10': 0.05, '11': 0.1},
        'ad': {'21': 0.8, '12': 0.05, '10': 0.05, '11': 0.1},
        'ar': {'12': 0.8, '01': 0.05, '21': 0.05, '11': 0.1},
        'al': {'10': 0.8, '01': 0.05, '21': 0.05, '11': 0.1}},

'12': {'au': {'02': 0.8, '13': 0.05, '11': 0.05, '12': 0.1},
        'ad': {'13': 0.05, '11': 0.05,'12': 0.9},
        'ar': {'13': 0.8, '02': 0.05, '12': 0.15},
        'al': {'11': 0.8, '02': 0.05, '12': 0.15}},


'13': {'au': {'03': 0.8, '14': 0.05, '12': 0.05, '13': 0.1},
        'ad': {'23': 0.8, '14': 0.05, '12': 0.05, '13': 0.1},
        'ar': {'14': 0.8, '03': 0.05, '23': 0.05, '13': 0.1},
        'al': {'12': 0.8, '03': 0.05, '23': 0.05, '13': 0.1}},

'14': {'au': {'04': 0.8, '13': 0.05, '14': 0.15},
        'ad': {'24': 0.8, '13': 0.05, '14': 0.15},
        'ar': {'04': 0.05, '24': 0.05, '14': 0.9},
        'al': {'13': 0.8, '04': 0.05, '24': 0.05, '14': 0.1}},

'20': {'au': {'10': 0.8, '21': 0.05, '20': 0.15},
        'ad': {'30': 0.8, '21': 0.05, '20': 0.15},
        'ar': {'21': 0.8, '10': 0.05, '30': 0.05, '20': 0.1},
        'al': {'10': 0.05, '30': 0.05, '20': 0.9}},

'21': {'au': {'11': 0.8, '20': 0.05, '21': 0.15},
        'ad': {'31': 0.8, '20': 0.05, '21': 0.15},
        'ar': {'11': 0.05, '31': 0.05, '21': 0.9},
        'al': {'20': 0.8, '11': 0.05, '31': 0.05, '21': 0.1}},

'22': {'au': {'12': 0.8, '23': 0.05, '21': 0.05, '22': 0.1},
        'ad': {'23': 0.05, '21': 0.05, '22': 0.9},
        'ar': {'23': 0.8, '12': 0.05, '22': 0.15},
        'al': {'21': 0.8, '12': 0.05, '22': 0.15}},

'23': {'au': {'13': 0.8, '24': 0.05, '23': 0.15},
        'ad': {'33': 0.8, '24': 0.05, '23': 0.15},
        'ar': {'24': 0.8, '13': 0.05, '33': 0.05, '23': 0.1},
        'al': {'13': 0.05, '33': 0.05, '23': 0.9}},

'24': {'au': {'14': 0.8, '23': 0.05, '24': 0.15},
        'ad': {'34': 0.8, '23': 0.05, '24': 0.15},
        'ar': {'14': 0.05, '34': 0.05, '24': 0.9},
        'al': {'23': 0.8, '14': 0.05, '34': 0.05, '24': 0.1}},

'30': {'au': {'20': 0.8, '31': 0.05, '30': 0.15},
        'ad': {'40': 0.8, '31': 0.05, '30': 0.15},
        'ar': {'31': 0.8, '20': 0.05, '40': 0.05, '30': 0.1},
        'al': {'20': 0.05, '40': 0.05, '30': 0.9}},

'31': {'au': {'21': 0.8, '30': 0.05, '31': 0.15},
        'ad': {'41': 0.8, '30': 0.05, '31': 0.15},
        'ar': {'21': 0.05, '41': 0.05, '31': 0.9},
        'al': {'30': 0.8, '21': 0.05, '41': 0.05, '31': 0.1}},


'32': {'au': {'33': 0.05, '31': 0.05, '32': 0.9},
        'ad': {'42': 0.8, '33': 0.05, '31': 0.05, '32': 0.1},
        'ar': {'33': 0.8, '42': 0.05, '32': 0.15},
        'al': {'31': 0.8, '42': 0.05, '32': 0.15}},

'33': {'au': {'23': 0.8, '34': 0.05, '33': 0.15},
        'ad': {'43': 0.8, '34': 0.05, '33': 0.15},
        'ar': {'34': 0.8, '23': 0.05, '43': 0.05, '33': 0.1},
        'al': {'23': 0.05, '43': 0.05, '33': 0.9}},


'34': {'au': {'24': 0.8, '33': 0.05, '34': 0.15},
        'ad': {'44': 0.8, '33': 0.05, '34': 0.15},
        'ar': {'24': 0.05, '44': 0.05, '34': 0.9},
        'al': {'33': 0.8, '24': 0.05, '44': 0.05, '34': 0.1}},

'40': {'au': {'30': 0.8, '41': 0.05, '40': 0.15},
        'ad': {'41': 0.05, '40': 0.95},
        'ar': {'41': 0.8, '30': 0.05, '40': 0.15},
        'al': {'30': 0.05, '40': 0.95}},

'41': {'au': {'31': 0.8, '42': 0.05, '40': 0.05, '41': 0.1},
        'ad': {'42': 0.05, '40': 0.05, '41': 0.9},
        'ar': {'42': 0.8, '31': 0.05, '41': 0.15},
        'al': {'40': 0.8, '31': 0.05, '41': 0.15}},

'42': {'au': {'43': 0.05, '41': 0.05, '42': 0.9},
        'ad': {'43': 0.05, '41': 0.05, '42': 0.9},
        'ar': {'43': 0.8, '42': 0.2},
        'al': {'41': 0.8, '42': 0.2}},
        
'43': {'au': {'33': 0.8, '44': 0.05, '42': 0.05, '43': 0.1},
        'ad': {'44': 0.05, '42': 0.05, '43': 0.9},
        'ar': {'44': 0.8, '33': 0.05, '43': 0.15},
        'al': {'42': 0.8, '33': 0.05, '43': 0.15}},

'44': {'au': {'34': 0.8, '44': 0.15, '43': 0.05},
        'ad': {'43': 0.05, '44': 0.95},
        'ar': {'34': 0.05, '44': 0.95},
        'al': {'43': 0.8, '34': 0.05, '44': 0.15}}}


class ValueIteration:
    def run_Grid_World():
        transition_probabilities = Dynamics.transition_probabilities
        value_state = Dynamics.initial_state
        policy = Dynamics.policy
        count = 0
        while True:
            delta = 0            
            # print("Value Iteration",count)     
            for row in range(Dynamics.row):
                for col in range(Dynamics.col):
                    curr_state = "{}{}".format(row, col)
                    if curr_state in Dynamics.obstacle_states or curr_state in Dynamics.goal_state:
                        continue
                    else:
                        initial_val = value_state[row][col]    
                        value, action = ValueIteration.compute_value_function(transition_probabilities[curr_state],value_state)
                        if curr_state == '42':
                            print(value)
                        value_state[row][col] = value
                        policy[row][col] = action
                        delta = max(delta, abs(initial_val - value_state[row][col]))
            count += 1
            if delta < Dynamics.theta:
                break
        return value_state, policy, count
                
                
    def compute_value_function(probability_values,value_states):
        max_action_val = -float('inf')
        max_policy = ""
        # print(f"Prb:",probabilities)
        for action, probs in probability_values.items():
            new_val = 0
            for next_state,prb in probs.items():
                reward = 0
                r, c = [int(char) for char in next_state]
                next_state_value = value_states[r][c]
                if(next_state in Dynamics.water_states):
                    reward = -10
                if(next_state in Dynamics.goal_state):
                    reward = 10
                new_val += prb * (reward + (Dynamics.gamma * next_state_value))
                # if curr == '42':
                #     print(new_val)
            if new_val > max_action_val:
                max_action_val = new_val
                max_policy = action
        # print(max_policy)
        return max_action_val, max_policy
    
def main():
        optimal_value, policy, count = ValueIteration.run_Grid_World()        
        np.set_printoptions(precision=4, suppress=True)
        # optimal_value = np.round(optimal_value, 4).astype(float)
        print(f"\n\nTotal value iterations for the algorithm to converge:", count)
        print(f"\n\nFinal Value function:\n") 
        for row in optimal_value:
            for value in row:
                print(round(value,4).astype(float), end="\t\t")
            print()
        directions = {'au': '↑', 'ad': '↓', 'al': '←', 'ar': '→'}   
        policy[4][4] = "G"
        policy_mapping = np.vectorize(lambda x: directions.get(x, x))
        final_policy = policy_mapping(policy)   
        print(f"\n\nFinal Policy:\n")
        for row in final_policy:
            for value in row:
                print(value, end="\t")
            print()
        print("\n\n")
        
if __name__ == "__main__":
        main()