import numpy as np
import heapq
import matplotlib.pyplot as plt

def distance(a, b):
    return np.abs(a-b)

def penalty(delta, x):
    if x < 1:
        return np.inf
    return (delta-x)**2

def get_path(item):
    path = []
    while item.backpointer is not None:
        path.append((item.i, item.j, item.k))
        item = item.backpointer
    path.append((item.i, item.j, item.k))
    return path

def plot_1d_alignment(x, y, w, path):
    plt.figure(figsize=(6, 4))
    min_x = min(x)
    max_y = max(y)
    diff = distance(min_x, max_y)
    diff += 1.5
    center = (len(y) / 2)-(len(x)/2)
    colors = ['C1', 'C2', 'C4', 'C6', 'C9']
    plt.scatter(np.arange(x.shape[0])+center, x + diff, c="C3", s=50*w)
    plt.scatter(np.arange(y.shape[0]), y - diff, c="C0", s=50)
    plt.plot(np.arange(x.shape[0])+center, x + diff, "-o", c="C3")
    plt.plot(np.arange(y.shape[0]), y - diff, "-o", c="C0")
    
    for x_i, y_j, d_k in path:
        plt.plot([x_i+center, y_j], [x[x_i] + diff, y[y_j] - diff], c=colors[x_i])
        
    plt.axis("off")
    plt.show()
    plt.savefig('dtw_dyna_style.png')

class Item:
    def __init__(self, i, j, k, cost=None):
        self.i = i
        self.j = j
        self.k = k
        self.cost = cost
        self.backpointer = None
        #self.updated = True

    def __str__(self):
        return f'({self.i}, {self.j}, {self.k}):{self.cost}'# -> {self.backpointer}'
    
    def __repr__(self):
        return f'({self.i}, {self.j}, {self.k}):{self.cost}'# -> {self.backpointer}'
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __eq__(self, other):
        return self.cost == other.cost
    
    def __gt__(self, other):
        return self.cost > other.cost
    
    def __le__(self, other):
        return self.cost <= other.cost
    
    def __ge__(self, other):
        return self.cost >= other.cost
    
    def __ne__(self, other):
        return self.cost != other.cost
    
    def update(self, cost):
        self.cost = cost
    #     self.updated = True

    # def is_same_loc(self, other):
    #     return self.i == other.i and self.j == other.j and self.k == other.k
    
    def set_backpointer(self, backpointer):
        self.backpointer = backpointer

class Heap:
    def __init__(self):
        self.heap = []
        self.seen = set()
        
    def push(self, item):
        #for item_ in self.heap:
        #    if item_.is_same_loc(item):
        #        if item_.cost > item.cost:
        #            item_.update(item.cost)
        #            item_.set_backpointer(item.backpointer)
        #        return
        if (item.i, item.j) in self.seen:
            return
        else:
            heapq.heappush(self.heap, item)
        #self.seen.add((item.i, item.j, item.k))

    def pop(self):
        #popped  = []
        #temp_item = heapq.heappop(self.heap)
        #while not temp_item.updated:
        #    popped.append(temp_item)
        #    temp_item = heapq.heappop(self.heap)
        #    if temp_item.updated:
        #        break
        #if temp_item.updated:
        #    return_item = temp_item
        #for item in popped:
        #    self.push(item)
        #return return_item
        return heapq.heappop(self.heap)
    
    def heapify(self):
        heapq.heapify(self.heap)
    
    def __len__(self):
        return len(self.heap)
    
    def __str__(self):
        return str(self.heap)
    
    def __repr__(self):
        return str(self.heap)

def main():
    x = np.array([0, 8, 5, 1, 3])
    x_weights = np.array([5, 2, 1, 1, 3])
    y = np.array([0, 8, 8, 0, 10, 10, 10, 5, 1, 3, 3, 3])

    alpha = 1

    N = x.shape[0]
    M = y.shape[0]
    d = np.zeros((N, M)) # filled with pairwise distances
    for i in range(N):
        for j in range(M):
            d[i, j] = distance(x[i], y[j])

    print(d.shape)

    Q = Heap()
    Q.push(Item(0, 0, 1, 0)) # cost of aligning nothing to nothing is 0

    # chart = Heap()

    while len(Q) > 0:
        #print(f'Q        : {Q}')
        item = Q.pop()
        print(f'Popped: {item}')

        new = Item(item.i+1, item.j+1, 1)
        another = Item(item.i, item.j+1, item.k+1)

        
        if item.i == N-1 and item.j == M-1:
            final_cost = item.cost + alpha*penalty(x_weights[item.i], item.k)
            print(f'Found : {item} with cost {final_cost}')
            break
        
        new_out = False
        if new.i >= N or new.j >= M or new.k > new.j+1:
            new_out = True

        another_out = False
        if another.i >= N or another.j >= M or another.k > another.j+1:
            another_out = True

        if new_out and another_out:
            continue


        if not new_out:
            new_cost = d[new.i, new.j] + item.cost + alpha*penalty(x_weights[item.i], item.k)
        if not another_out:
            another_cost = d[another.i, another.j] + item.cost

        new.update(new_cost)
        another.update(another_cost)

        new.set_backpointer(item)
        another.set_backpointer(item)

        Q.push(new)
        Q.push(another)
        #item.updated = False
        #Q.push(item)
        # chart.push(item)

        #print(f'Updated Q: {Q}')

    path = get_path(item)
    print(path)
    plot_1d_alignment(x, y, x_weights, path)


if __name__ == "__main__":
    main()  