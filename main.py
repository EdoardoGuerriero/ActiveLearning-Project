#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from simulation import train_with_active_learning

if __name__ == '__main__':
    
    steps = [x+1 for x in range(9)]
    max_step = max(steps)
    
    '''
	All strategies applied:
    random / entropy / cost-effective / monte-carlo / max-margin / least-conf
    '''

    strategys = ['max-margin', 'least-conf']
    # I suggest to create a folder beforehand to store the results of each simulation separatly 
    folder_names = ['Max-Margin', 'Least-Conf']
    
    for i in range(30):
        sim = i+1
        for s, strategy in enumerate(strategys):
            
            print(strategy)
            
            folder_name = folder_names[s]+' '+str(sim)
            strategy = strategy
            
            for step in steps:
                train_with_active_learning(folder_name, strategy, str(sim), step, max_step)
