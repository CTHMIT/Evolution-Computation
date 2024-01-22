#
# ev3.py: An elitist (mu+mu) generational-with-overlap EA
#
#
# To run: python ev3.py --input ev3.cfg
#
# Basic features of ev3:
#   - Supports self-adaptive mutation
#   - Uses binary tournament selection for mating pool
#   - Uses elitist truncation selection for survivors
#

import optparse
import sys
import yaml
import math
from random import Random
from Population import *
import matplotlib.pyplot as plt


# EV3 Config class 
class EV3_Config:
    """
    EV3 configuration class
    """
    # class variables
    sectionName='EV3'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'crossoverFraction': (float,True),
             'minLimit': (float,True),
             'maxLimit': (float,True),
             'MultiminLimit': (float,True),
             'MultimaxLimit': (float,True)}
     
    # constructor
    def __init__(self, inFileName):
        #read YAML config and get EV3 section
        infile=open(inFileName,'r')
        ymlcfg=yaml.safe_load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing {} section in cfg file'.format(self.sectionName))
         
        # iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]
 
                # verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))
                 
                # create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)
     
    # string representation for class data    
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))
         

# Simple fitness function example: 1-D Rastrigin function        
def fitnessFunc(x, A=10):
     # return -10.0-(0.04*x)**2+10.0*math.cos(0.04*math.pi*x)
     return x**2 - A * math.cos(2 * math.pi * x) # Problem #2 x^2 - Acos(2pix)


# Print some useful stats to screen
def printStats(pop,gen):
    print('Generation:',gen)
    avgval=0
    maxval=pop[0].fit 
    sigma=pop[0].sigma

    ind_maxval = []
    ind_sigma = []
    
    for ind in pop:
        avgval+=ind.fit
        if ind.fit > maxval:
            maxval=ind.fit
            sigma=ind.sigma
        ind_maxval.append(ind.fit)
        ind_sigma.append(ind.sigma)
        print(ind)
        
        
    
    print('Max fitness',maxval)
    print('Sigma',sigma)
    print('Avg fitness',avgval/len(pop))
    print('')
    
    return ind_maxval, ind_sigma


# EV3:        
def ev3(cfg, program_name):
    """
    • N = number of lattice sites
    • i is the lattice site index
    • Si is the “state” on site i (i.e., the particle type/color on site i)
    • u is the self-energy vector, t is the interaction energy matrix
    • 3-particle universe with self energies: Red=10, Blue=20, Green=30
    • We label the colors Red:0, Blue:1, Green:2
    • The self-energy vector would then be: u = [10,20,30]
    • selfEnergyVector: [...] (a list of length M, where M is number of particle types)
    • interactionEnergyMatrix: [[…],[…],…] (MxM matrix as a list of lists)
    • latticeLength: N (an int, where N is length of the 1-D lattice)
    • numParticleTypes: M (an int, where M is the number of particle types)
    """
    # New parameter for Problem #1
    # Lattice length
    N = 11
    Ptype = 3
    # self energy vector
    u = [1,2,3]
    # interaction energy matrix
    t = [[10,4,1],[4,10,5],[1,5,10]] 

    # New parameter for Problem #2
    A = 10
    n = 2    

    # Problem #1: Total energy function
    class oneD:
        u=None
        t=None
        def fitnessFunc(x):
            
            totalenergy = 0
            len_state = len(x)
            for i in range(len_state):
                usi = -u[x[i]]
                if i == 0:
                    ti = 0 + t[x[i]][x[i+1]] # t[-1,0]=0
                elif i == len_state-1:
                    ti = t[x[i-1]][x[i]] + 0 # t[N-1,N]=0
                else :
                    ti = t[x[i-1]][x[i]] + t[x[i]][x[i+1]]

                totalenergy -= usi + ti # E = sum(ui + t0 + t1)

            return totalenergy
    class Multi:
        A = None
        n = None
        def fitnessFunc(x):
            output = -A * n
            for i in range(n):
                output -= x[i] * x[i] - A * math.cos(2 * math.pi * x[i])
            
            return output



    #start random number generators
    uniprng=Random()
    uniprng.seed(cfg.randomSeed)
    normprng=Random()
    normprng.seed(cfg.randomSeed+101)

    Individual.uniprng=uniprng
    Individual.normprng=normprng
    Population.uniprng=uniprng
    Population.crossoverFraction=cfg.crossoverFraction

    #set static params on classes
    # (probably not the most elegant approach, but let's keep things simple...)
    if program_name == 'oneD':
        oneD.u = u
        oneD.t = t
        oneDuniverse.length = N   
        oneDuniverse.Ptype = Ptype
        oneDuniverse.fitFunc=oneD.fitnessFunc
        oneDuniverse.learningRate=1.0/math.sqrt(N)
        Population.name=oneDuniverse
        
    if program_name == 'multi':
        Multivariate.minLimit=cfg.MultiminLimit 
        Multivariate.maxLimit=cfg.MultimaxLimit 
        Multi.A = A
        Multi.n = n
        Multivariate.length = N 
        Multivariate.fitFunc = Multi.fitnessFunc
        Multivariate.learningRate=1.0/math.sqrt(n)
        Population.name=Multivariate

  
    #create initial Population (random initialization)
    population=Population(cfg.populationSize)
        
    #print initial pop stats    
    printStats(population,0)

    ind_maxval = []
    ind_sigma = []
    plt.figure(f'EV3-{program_name}')
    plt.ion()
    plt.draw()

    #evolution main loop
    for i in range(cfg.generationCount):
        #create initial offspring population by copying parent pop
        offspring=population.copy()
        
        #select mating pool
        offspring.conductTournament()

        #perform crossover
        offspring.crossover()
        
        #random mutation
        offspring.mutate()
        
        #update fitness values
        offspring.evaluateFitness()        
            
        #survivor selection: elitist truncation using parents+offspring
        population.combinePops(offspring)
        population.truncateSelect(cfg.populationSize)
        
        #print population stats    
        maxval, sigma = printStats(population,i+1)
        ind_maxval.extend(maxval)
        ind_sigma.extend(sigma)

        
        plt.subplot(211)
        plt.plot(ind_maxval, "o")
        plt.subplot(212)
        plt.plot(ind_sigma)
        plt.pause(0.08)
    plt.savefig(f'EV3-{program_name}')
    plt.show()
      
# Main entry point
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    try:
        # get command-line options
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
        (options, args) = parser.parse_args(argv)
        
        #validate options
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")
        
        #Get EV3 config params
        cfg=EV3_Config(options.inputFileName)
        
        #print config params
        print(cfg)
                    
        #run EV3
        ev3(cfg, program_name = 'oneD')
        ev3(cfg, program_name= 'multi')
        
        if not options.quietMode:                    
            print('EV3 Completed!')    
    
    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)
    

if __name__ == '__main__':
    main()
    
