import numpy as np
import pandas as pd
import cv2 as cv
from scipy.special import softmax

class Specimen():
    
    __slots__ = ['genes']

    def __init__(self, genes) -> None:
        self.genes = genes

    def __str__(self) -> str:
        s = ""
        for k, v in self.genes.items():
            s += f"{k}: {v}\n"
        return s
    
    def evaluate(self):
        return 0

    def mutate(self):
        return None    

    def copy(self):
        return Specimen(self.genes)
    
    @classmethod
    def crossover(cls, *parents):
        new_genes = {}
        n_parents = len(parents)
        for k in parents[0].genes.keys():
            parent_idx = np.random.randint(0, n_parents)
            new_genes[k] = parents[parent_idx].genes[k]
        return new_genes

class R3Quadratic(Specimen):
    
    def __init__(self, genes = None) -> None:
        if genes is None:
            self.genes = {
                'x': np.random.uniform(-1, 1),
                'y': np.random.uniform(-1, 1),
                'z': np.random.uniform(-1, 1)
            }
        else:
            self.genes = genes


    def evaluate(self):
        a = self.genes['x']**2+\
            self.genes['y']**2+\
            2*self.genes['z']**2
        return -a # we are minimising


    def mutate(self, p=0.1, strength=0.1):
        for k in self.genes.keys():
            if np.random.random() < p:
                self.genes[k] += np.random.uniform(-strength, strength)

    def copy(self):
        return R3Quadratic(self.genes)

    
    @classmethod
    def crossover(cls, *parents):
        new_genes = super().crossover(*parents)
        return R3Quadratic(new_genes)
    
class Rastring(Specimen):

    __slots__ = ['n']

    def __init__(self, n = 5, genes = None) -> None:
        self.n = n
        if genes is None:
            self.genes = {}
            for i in range(n):
                self.genes[f'x{i+1}'] = np.random.uniform(-5.12, 5.12)
        else:
            self.genes = genes


    def evaluate(self):
        A = 10
        x = np.array(list(self.genes.values()))
        s = np.sum(np.square(x) - A*np.cos(2*np.pi*x)) + A * self.n
        return -s # minimisng


    def mutate(self, p=0.3, strength=0.05):
        for k in self.genes.keys():
            if np.random.random() < p:
                self.genes[k] += np.random.uniform(-strength, strength)

    def copy(self):
        return Rastring(self.genes)

    
    @classmethod
    def crossover(cls, *parents):
        new_genes = super().crossover(*parents)
        return Rastring(n=len(new_genes), genes=new_genes)
    
class CircleCutting(Specimen):

    __slots__ = ['rectangles', 'radius']

    value_map = {}
    width_map = {}
    height_map = {}

    def __init__(self, radius, init_rectangles=100, genes = None) -> None:
        self.radius = radius
        if genes is None:
            self.genes=[self.__new_rectangle() for _ in range(init_rectangles)]
        else:
            self.genes = genes


    @classmethod
    def use_file(cls, file_path):
        df = pd.read_csv(file_path, header=None)
        cls.value_map = df[2]
        cls.width_map = df[0]
        cls.height_map = df[1]
        cls.density_map = df[2] / (df[0] * df[1])
        cls.rectangle_weights = softmax(cls.density_map.values * 500)


    def __new_rectangle(self):
        while True:
            r = np.inf
            while r >= self.radius**2:
                x, y = np.random.uniform(-self.radius, self.radius, size=2)
                r = x**2 + y**2
            x, y = np.round([2*x, 2*y])/2
            i = np.random.choice(np.arange(len(self.value_map)), size=1, p=CircleCutting.rectangle_weights)[0]
            if self.__is_rectangle_in_circle([i, x, y]):
                return [i, x, y]
    
    def __is_rectangle_in_circle(self, rectangle):
        i, x, y = rectangle
        width = self.width_map[i]
        height = self.height_map[i]
        if (x - width/2)**2 + (y - height/2)**2 > self.radius**2:
            return False
        if (x - width/2)**2 + (y + height/2)**2 > self.radius**2:
            return False
        if (x + width/2)**2 + (y - height/2)**2 > self.radius**2:
            return False
        if (x + width/2)**2 + (y + height/2)**2 > self.radius**2:
            return False
        return True
    
    @classmethod
    def rectangles_overlapping(cls, r1, r2):
        t1, x1, y1 = r1
        t2, x2, y2 = r2
        r1_tl = x1 - cls.width_map[t1] / 2, y1 + cls.height_map[t1] / 2
        r1_br = x1 + cls.width_map[t1] / 2, y1 - cls.height_map[t1] / 2
        r2_tl = x2 - cls.width_map[t2] / 2, y2 + cls.height_map[t2] / 2
        r2_br = x2 + cls.width_map[t2] / 2, y2 - cls.height_map[t2] / 2
        if r1_tl[0] < r2_br[0] and r2_tl[0] < r1_br[0] and\
            r1_br[1] < r2_tl[1] and r2_br[1] < r1_tl[1]:
            return True
        return False


    def no_rectangles_overlap(self):
        for i, r1 in enumerate(self.genes):
            for j, r2 in enumerate(self.genes[i+1:]):
                j += i + 1
                if CircleCutting.rectangles_overlapping(r1, r2):
                    return False, i, j
        return True, -1, -1


    def evaluate(self):
        no_rectangles_overlap, i, j = self.no_rectangles_overlap()
        if not no_rectangles_overlap:
            # print(f'Invalid specimen beacuse rectangles {i} and {j} overlap')
            return -np.inf
        value = 0
        for i, rectangle in enumerate(self.genes):
            if not self.__is_rectangle_in_circle(rectangle):
                # print(f'Invalid specimen beacuse rectangle {i} is partially out side of the circle')
                return -np.inf
            else:
                value += self.value_map[rectangle[0]]
        return value


    def mutate(self, p=0.01, strength=200):
        for i, rectangle in enumerate(self.genes):
            if np.random.random() < p:
                new_type = np.random.choice(np.arange(len(self.value_map)), size=1, p=CircleCutting.rectangle_weights)[0]
                self.genes[i][0] = new_type
            if np.random.random() < p:
                self.genes[i][1] += np.random.randint(-strength, strength+1) / 2
            if np.random.random() < p:
                self.genes[i][2] += np.random.randint(-strength, strength+1) / 2
            if np.random.random() < p:
                if np.random.random() < 0.5:
                    width = self.width_map[rectangle[0]]
                    offspring = self.genes[i].copy()
                    self.genes[i][1] -= width / 2
                    offspring[1] += width / 2
                    self.genes.append(offspring)
                else:
                    height = self.height_map[rectangle[0]]
                    offspring = self.genes[i].copy()
                    self.genes[i][2] -= height / 2
                    offspring[2] += height / 2
                    self.genes.append(offspring)
        while np.random.random() < 0.3:
            self.genes.append(self.__new_rectangle())
        new_genes = []
        for i, rectangle in enumerate(self.genes):
            if np.random.random() < (1 - p):
                new_genes.append(rectangle) # copy ?
        self.genes = new_genes


    def copy(self):
        return CircleCutting(self.radius, self.genes) 

    @classmethod
    def adds_overlap(cls, new_rectangle, other_rectangles):
        for other in other_rectangles:
            if cls.rectangles_overlapping(new_rectangle, other):
                return True
        return False

    @classmethod
    def crossover(cls, *parents):
        p = 1/len(parents)
        new_genes = []
        for parent in parents:
            for rectangle in parent.genes:
                if np.random.random() >= p:
                    continue
                if rectangle not in new_genes and not cls.adds_overlap(rectangle, new_genes):
                    new_genes.append(rectangle.copy())
        return CircleCutting(radius=parents[0].radius, genes=new_genes)
    
    def __str__(self):
        s = ""
        for rectangle in self.genes:
            s += f"[{rectangle[0]}, {rectangle[1]}, {rectangle[2]}],\n"
        return s
    
    def get_image(self):
        img = np.zeros((2*self.radius,2*self.radius,3), np.uint8)
        offset_x = self.radius
        offset_y = self.radius
        for i, x, y in self.genes:
            tl_x = int(x - self.width_map[i] / 2 + offset_x)
            tl_y = int(y + self.height_map[i] / 2 + offset_y)
            br_x = int(x + self.width_map[i] / 2 + offset_x)
            br_y = int(y - self.height_map[i] / 2 + offset_y)
            r = np.random.randint(64, 193)
            g = np.random.randint(64, 193)
            b = np.random.randint(64, 193)
            cv.rectangle(img, (tl_x, tl_y), (br_x, br_y), (r, g, b), -1)
        cv.circle(img, (offset_x, offset_y), self.radius, (255, 255, 255), 3)
        return img

class MLP(Specimen):

    X = None
    Y = None
    activation_function = None
    loss_function = None
    output_activation = None

    def __init__(self, genes=None, architecture=None) -> None:
        if genes is None:
            if architecture is None: 
                raise ValueError
            self.genes = []
            for i in range(len(architecture) - 1):
                self.genes.append((
                    np.random.uniform(-1, 1, size=(architecture[i+1], architecture[i])),
                    np.random.uniform(-1, 1, size=(architecture[i+1], 1))
                ))
        else:
            self.genes = genes

    @classmethod
    def set_parameters(cls, X, Y, activation_function, output_activation, loss_function):
        cls.X = X
        cls.Y = Y
        cls.activation_function = activation_function
        cls.loss_function = loss_function
        cls.output_activation = output_activation

    def evaluate(self):
        output = self.X
        for i, (W, b) in enumerate(self.genes):
            if i < len(self.genes) - 1:
                output = MLP.activation_function(W @ output + b)
            else:
                output = MLP.output_activation(W @ output + b)
        return -MLP.loss_function(self.Y.T, output.T)

    def mutate(self, p=0.01, strength=0.2):
        for W, b in self.genes:
            dW = (np.random.uniform(size=W.shape) < p) * np.random.uniform(-strength/2, strength/2, size=W.shape)
            db = (np.random.uniform(size=b.shape) < p) * np.random.uniform(-strength/2, strength/2, size=b.shape)
            W += dW
            b += db
    
    def copy(self):
        raise NotImplementedError
    
    @classmethod
    def crossover(cls, *parents):
        new_genes = []
        if len(parents) == 1:
            for i in range(len(parents[0].genes)):
                new_genes.append((
                    np.copy(parents[0].genes[i][0]),
                    np.copy(parents[0].genes[i][1])
                ))    
            return MLP(new_genes)
        # assuming 2 parents
        
        for i in range(len(parents[0].genes)):
            W1, b1 = parents[0].genes[i]
            W2, b2 = parents[1].genes[i]

            pW = np.random.uniform(size=W1.shape) < 0.5
            pb = np.random.uniform(size=b1.shape) < 0.5
            new_genes.append((W1 * pW + W2 * (1-pW), b1 * pb + b2 * (1-pb)))
        return MLP(new_genes)
        

        


