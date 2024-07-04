class Heap:
    #comparator(a, b) returns True if b should swap with a
    def __init__(self, list, comparator):
        self.heap = []
        self.comparator = comparator
        for item in list:
            self.Push(item)

    def Heapify(self):
        for idx in range(len(self.heap) - 1,0,-1):
            if (idx % 2 == 0 and self.comparator(self.heap[(idx//2)-1], self.heap[idx])):
                tmp = self.heap[idx]
                self.heap[idx] = self.heap[(idx//2)-1]
                self.heap[(idx//2)-1] = tmp
            elif (idx % 2 == 1 and self.comparator(self.heap[((idx+1)//2)-1], self.heap[idx])):
                tmp = self.heap[idx]
                self.heap[idx] = self.heap[((idx+1)//2)-1]
                self.heap[((idx+1)//2)-1] = tmp

    def Pop(self):
        if len(self.heap) == 0:
            return 0
        tmp = self.heap[0]
        self.heap[0] = self.heap[len(self.heap)-1]
        del self.heap[len(self.heap)-1]
        idx = 0
        while idx < (len(self.heap)/2) - 1:
            if (idx+1)*2 > len(self.heap)-1:
                largerChild = ((idx+1)*2)-1
            elif (self.comparator(self.heap[(idx+1)*2], self.heap[((idx+1)*2)-1])):
                largerChild = ((idx+1)*2)-1
            else:
                largerChild = (idx+1)*2
            if (self.comparator(self.heap[idx],self.heap[largerChild])):
                tmp = self.heap[idx]
                self.heap[idx] = self.heap[largerChild]
                self.heap[largerChild] = tmp
                idx = largerChild
            else:
                break
        return tmp
    
    def Push(self,item):
        self.heap.append(item)
        idx = len(self.heap)-1
        while idx > 0:
            if (idx % 2 == 0 and self.comparator(self.heap[(idx//2)-1], self.heap[idx])):
                tmp = self.heap[(idx//2)-1]
                self.heap[(idx//2)-1] = self.heap[idx]
                self.heap[idx] = tmp
                idx = (idx//2)-1
            elif(idx % 2 == 1 and self.comparator(self.heap[((idx+1)//2)-1], self.heap[idx])):
                tmp = self.heap[((idx+1)//2)-1]
                self.heap[((idx+1)//2)-1] = self.heap[idx]
                self.heap[idx] = tmp
                idx = ((idx+1)//2)-1
            else:
                break

    def GetCount(self):
        return len(self.heap)

    def Peek(self):
        if len(self.heap) > 0:
            return self.heap[0]
        else:
            return 0
    def PrintHeap(self):
        print(self.heap)
