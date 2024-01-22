import math

#A simple 1-D Individual class
class Individual:
    """
    Individual
    """
    minSigma=1e-100
    maxSigma=1
    learningRate=1
    uniprng=None
    normprng=None
    fitFunc=None

    def __init__(self):
        self.fit=self.__class__.fitFunc(self.x)
        self.sigma=self.uniprng.uniform(0.9,0.1) #use "normalized" sigma
    
    def murate(self):
        self.sigma=self.sigma*math.exp(self.learningRate*self.normprng.normalvariate(0,1))
        if self.sigma < self.minSigma: self.sigma=self.minSigma
        if self.sigma > self.maxSigma: self.sigma=self.maxSigma
    
    def evaluateFitness(self):
        if self.fit == None: self.fit=self.__class__.fitFunc(self.x)
        

class oneDuniverse(Individual):

        length = None
        Ptype = None
        def __init__(self):
            self.x = []
            for _ in range(self.length):
                self.x.append(self.uniprng.randint(0, self.Ptype-1))
            super().__init__()

        def crossover(self, other):
            for i in range(self.length):
                tmp=self.x[i]
                self.x[i]=other.x[i]
                other.x[i]=tmp
            
            self.fit=None
            other.fit=None

        def mutate(self):
            self.murate() 
            
            for i in range(self.length):
                if self.uniprng.random() < self.sigma:
                    self.x[i]=self.uniprng.randint(0,self.Ptype-1)
            
            self.fit=None
                
        def __str__(self):
            return str(self.x)+'\t'+'%0.8e'%self.fit+'\t'+'%0.8e'%self.sigma

class Multivariate(Individual):
    minLimit=None
    maxLimit=None
    length=None
    

    def __init__(self):
        self.x=[self.uniprng.uniform(self.minLimit,self.maxLimit)for _ in range(self.length)]   
        super().__init__() 
        
    def crossover(self, other):
        #perform crossover "in-place"
        alpha=self.uniprng.random()
        
        for i in range(self.length):
            tmp=self.x[i]*alpha+other.x[i]*(1-alpha)
            other.x[i]=self.x[i]*(1-alpha)+other.x[i]*alpha
            self.x[i]=tmp
            
            if self.x[i] > self.maxLimit: self.x[i]=self.maxLimit
            if self.x[i] < self.minLimit: self.x[i]=self.minLimit
            if other.x[i] > self.maxLimit: other.x[i]=self.maxLimit
            if other.x[i] < self.minLimit: other.x[i]=self.minLimit
        
        self.fit=None
        other.fit=None
    
    def mutate(self):
        self.murate()
        for i in range(self.length):
            self.x[i]=self.x[i]+(self.maxLimit-self.minLimit)*self.sigma*self.normprng.normalvariate(0,1)
            if self.x[i] > self.maxLimit: self.x[i]=self.maxLimit
            if self.x[i] < self.minLimit: self.x[i]=self.minLimit
            
        self.fit=None
    
    def evaluateFitness(self):
        if self.fit == None: self.fit=self.__class__.fitFunc(self.x)
        
    def __str__(self):
        return str(self.x)+'\t'+'%0.8e'%self.fit+'\t'+'%0.8e'%self.sigma          

