
import optparse
import sys
import yaml
import math
import random 
import matplotlib.pyplot as plt
import numpy as np

class EV1_Config:
    """
    7111093024 EV2 configuration class
    """
    # class variables
    sectionName='7111093024EV2'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'minLimit': (float,True),
             'maxLimit': (float,True),
             'maxfitvalue':(float,True)}
     
    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EC_Engine section
        infile=open(inFileName,'r')
        ymlcfg=yaml.safe_load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing EV2 section in cfg file')
         
        #iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]
 
                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))
                 
                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)
     
    #string representation for class data    
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))


#Simple 1-D fitness function example
#        
def fitnessFunc(x, UseOriginial = False):
    if UseOriginial:
        return 50.0 - x*x
    else:
        return -10-(0.04*x) ** 2 + 10*np.cos(0.04*np.pi*x)


#Find index of worst individual in population
def findWorstIndex(l):
    minval=l[0].fit
    imin=0
    for i in range(len(l)):
        if l[i].fit < minval:
            minval=l[i].fit
            imin=i
    return imin


#Print some useful stats to screen
def printStats(cfg, pop, gen):
    print('Generation:',gen)
    avgval=0
    maxval=pop[0].fit
    
    x = []
    y = []
    plt.ion()
    plt.figure('Base')
    for p in pop:
        avgval+=p.fit

        if p.fit > maxval: maxval=p.fit
        # print(str(p.x)+'\t'+str(p.fit))
        y.append(p.fit)
        x.append(p.x)
        
        

    xp = np.arange(cfg.minLimit,cfg.maxLimit)
    yp = fitnessFunc(xp)
    plt.clf()
    plt.grid()
    plt.plot(x, y, "o")
    plt.plot(xp, yp)
    plt.title('Generation : {}  Max : {:.5f}  Avg : {:.5f} Std : {:.5f}'.format(gen,maxval,avgval/len(pop),np.std(y)))
    plt.pause(0.08)
    plt.show()
    plt.savefig('EV2-1')
    # print('Max fitness : {:.7f}'.format(maxval))
    # print('Avg fitness : {:.7f}'.format(avgval/len(pop)))
    # print('Std fitness : {:.7f}'.format(np.std(y)))
    # print('')

    return maxval ,avgval/len(pop), np.std(y)

#  ====== EV2 - 1 Uncorrelated mutation with single sigma ======
def N(mu=0, sigma=1):
        return np.random.normal(mu, sigma)

class EV2_func():
    def __init__(self):
        pass

    def crossover(parents):
        '''
        Instead of x = ( x1 + x2 ) / 2, 
        use x = a * x1 + (1 - a) * x2, 
        a : 0 <= a <= 1 (a is random value)
        '''
        # stochastic arithmetic crossover
        x1 = parents[0].x
        x2 = parents[1].x
        a = random.random()       
        chi = a * x1 + ( 1 - a ) * x2
        return chi
            
    def mutate(x, sigma, learning_rate):   
        sigma_2 = sigma * math.exp(learning_rate * N(0,1))
        return x + (sigma_2 * N(0,1)), sigma_2



#A trivial Individual class
class Individual:
    def __init__(self,x=0,fit=0):
        self.x=x
        self.fit=fit


def ev1(cfg, Create_children = 5):
    # start random number generator
    prng=random.Random()
    prng.seed(cfg.randomSeed)
    
    
    population = []
    sigma_list = []
    for i in range(cfg.populationSize):
        x = prng.uniform(cfg.minLimit,cfg.maxLimit)
        
        ind = Individual(x, fitnessFunc(x))
        
        population.append(ind)
        
        
    printStats(cfg, population,0)
    #evolution main loop
    plt_data_mAx, plt_data_aVg, plt_data_sTd = [], [], []

    underflow, overflow = 0.5 , 0.5
    sigma = np.sqrt(np.var([i.x for i in population]))
    sigma_list.append(sigma)
    
    
    for i in range(cfg.generationCount):
        sigma = np.sqrt(np.var([i.x for i in population])) # sigma
        learning_rate = 10 / (cfg.populationSize ** 0.5)
        sigma_list.append(sigma)
        
 
        # ====== EV2 - 3 multiple children per generation ======
        childx_list = []
        for _ in range(Create_children):
            
            parents=prng.sample(population,2)
            #randomly select two parents
            # ====== EV2 - 2  Modify crossover operator ======
            child=EV2_func.crossover(parents) # stochastic arithmetic crossover
            
            child_2, sigma_2 = EV2_func.mutate(child, sigma, learning_rate)
            
            childx_list.append(child_2)
            
            # boundary rules (no overflow / underflow)
            if sigma_2 < underflow:
                underflow = sigma_2
            if sigma_2 > overflow:
                overflow = sigma_2
            
            #random mutation using normal distribution
            #survivor selection: replace worst
        
        for childx in childx_list:
            
            childx=Individual(childx, fitnessFunc(childx))
            iworst=findWorstIndex(population)
            if childx.fit > population[iworst].fit:
                population[iworst]=childx
                
            
            #print stats    
            mAx ,aVg ,sTd = printStats(cfg, population, i+1)
            plt_data_mAx.append(mAx)
            plt_data_aVg.append(aVg)
            plt_data_sTd.append(sTd)

            if mAx >= cfg.maxfitvalue:   
                break
    
        
        
        plt.figure('sigma')
        plt.clf()
        
        plt.plot(sigma_list, label = "sigma")
        plt.title('sigma : {:.2f}'.format(sigma))
        plt.grid()
        plt.legend()
        plt.pause(0.08)
        plt.show()
        plt.savefig('EV2-2')

    plt.ioff()
    plt.figure('MAS')
    plt.plot(plt_data_mAx, label = "max")
    plt.plot(plt_data_aVg, label = "avg")
    plt.plot(plt_data_sTd, label = "std")   
    plt.grid()
    plt.legend()
    plt.savefig('EV2-3')
    plt.show()
    plt.pause(0)
    plt.close()

#
# Main entry point
#
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    try:
        #
        # get command-line options
        #
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
        (options, args) = parser.parse_args(argv)
        
        #validate options
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")
        
        #Get EV1 config params
        cfg=EV1_Config(options.inputFileName)
        
        #print config params
        print(cfg)
                    
        #run EV1
        ev1(cfg)
        
        if not options.quietMode:                    
            print('7111093024 EV2 Completed!')  
    
    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)


if __name__ == '__main__':
    main()


