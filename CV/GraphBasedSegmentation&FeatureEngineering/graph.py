import numpy as np
from PIL import Image, ImageFilter
from random import random

class Node: #one node refers to one pixel in graph
    def __init__(self,parent, rank = 0, size = 1):
        self.parent = parent
        self.rank = rank
        self.size = size
        self.number = parent
    def __repr__(self):
        return '(parent=%s, rank=%s, size=%s, number = %s)'%(self.parent,self.rank,self.size,self.number)

class Forest: #forest defines sets of all nodes and operation among nodes.
    def __init__(self,num_nodes):
            self.nodes = [Node(i) for i in range(num_nodes)]
            self.num_sets = num_nodes

    def size_of(self,i):
            return self.nodes[i].size

    def find(self,n):
            temp = n
            while(temp != self.nodes[temp].parent):
                temp = self.nodes[temp].parent
            self.nodes[n].parent = temp
            return temp

    def merge(self, a , b):
            if self.nodes[a].rank >self.nodes[b].rank:
                self.nodes[b].parent = a
                self.nodes[a].size = self.nodes[a].size + self.nodes[b].size
            else:
                self.nodes[a].parent = b
                self.nodes[b].size = self.nodes[a].size + self.nodes[b].size
                if(self.nodes[a].rank == self.nodes[b].rank):
                    self.nodes[b].rank = self.nodes[b].rank + 1
            self.num_sets = self.num_sets - 1
    def printnodes(self):
            for node in self.nodes:
                print (node)
    def dict_parent2node(self):
        dict = {}
        for node in self.nodes:
            dict[self.find(node.number)]= []
        for node in self.nodes:
            dict[self.find(node.number)].append(node.number)
        return dict

#diff function defines the difference between 2 nodes
def diff(img, x1, y1, x2, y2):
    _out = np.sum((img[x1, y1] - img[x2, y2]) ** 2)
    return np.sqrt(_out)

#threshold function
def threshold(size, const):
    return (const * 1.0 / size)

# an edge defines the id of two nodes and the weight of edge.
def create_edge(img,width, x,y, x1,y1,diff):
    vertex_id = lambda x,y: x*width + y
    w = diff(img,x,y,x1,y1) #weight of edge
    return (vertex_id(x,y),vertex_id(x1,y1),w)

#build graph with edges and nodes
def build_graph(img,width,height, diff, neighbour8 = False):
    graph = []
    for x in range(height):
        for y in range(width):
            if (x>=1):
                graph.append(create_edge(img,width,x,y,x-1,y,diff))
            if (y>=1):
                graph.append(create_edge(img,width,x,y,x,y-1,diff))
            if neighbour8:
                if (x>0 and y>0):
                    graph.append(create_edge(img,width,x,y,x-1,y-1,diff))
                if (x>0 and y< width -1):
                    graph.append(create_edge(img,width,x,y,x-1,y+1,diff))
    return graph

#merge 2 components if they are too small
def merge_small_components(forest, graph,min_size):
    for edge in graph:
        a = forest.find(edge[0])
        b = forest.find(edge[1])
        # merge 2 nodes if area_size < min_size
        if a!=b and (forest.size_of(a)<min_size or forest.size_of(b)<min_size):
            forest.merge(a,b)
    return forest

#segment graph and return forest
def segment_graph(graph,num_nodes,const, min_size,threshold_func):
    weight = lambda edge: edge[2]
    forest = Forest(num_nodes)
    sorted_graph = sorted(graph,key = weight)
    threshold = [threshold_func(1,const)] *num_nodes

    for edge in sorted_graph:
        parent_a = forest.find(edge[0])
        parent_b = forest.find(edge[1])
        a_condition = weight(edge) <= threshold[parent_a]
        b_condition = weight(edge) <= threshold[parent_b]

        if parent_a != parent_b and a_condition and b_condition:
            forest.merge(parent_a,parent_b)
            a = forest.find(parent_a)
            #print(a)
            threshold[a] = weight(edge) + threshold_func(forest.nodes[a].size,const)
    return merge_small_components(forest,sorted_graph,min_size)

def generate_image(forest, width, height,colors):
    img = Image.new('RGB', (width, height))
    im = img.load()
    dict = forest.dict_parent2node()
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            im[x, y] = colors[comp]

    #return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
    return img

import os


def get_forest(img_file,sigma,K,min_size,neighbour = 8):
    size = img_file.size
    print('Image info:', img_file.format,size, img_file.mode)
    #gauss grid
    smoothgraph = img_file.filter(ImageFilter.GaussianBlur(sigma))
    smoothgraph = np.array(smoothgraph)
    size = img_file.size
    width = size[0]
    height = size[1]

    diff_func = diff
    graph = build_graph(smoothgraph,width,height,diff_func,neighbour ==8)
    forest = segment_graph(graph,width*height,K,min_size,threshold)
    print("number of components: %d" % forest.num_sets)
    return forest

def calc_IOU(image,gt,width,height):
    #calculate IOU
    gt_fore_pixel_num = 0
    fore_pixel_num = 0
    correct_pixel_num = 0
    for i in range(width):
        for j in range(height):
            gr,gg,gb = gt.getpixel((i,j))
            r,g,b = image.getpixel((i,j))
            if(gr<= 128 and gg<=128 and gb<=128):
                gt_fore_pixel_num += 1
                if(r==0 and g ==0 and b == 0):
                    correct_pixel_num += 1
            if(r == 0 and g ==0 and b ==0 ):
                fore_pixel_num += 1
    IOU = correct_pixel_num / (gt_fore_pixel_num +fore_pixel_num - correct_pixel_num) * 1.00
    print("IOU:",IOU)
    return IOU

def graphbased_segmentation(imagename, sigma = 0.5 ,K = 200, min_size = 20, neighbour = 8):
    neighbour = 8
    gt_folder = "./data/gt/"
    img_folder = './data/imgs/'
    img_path = os.path.join(img_folder, imagename)
    img_file = Image.open(img_path)
    gt_path = os.path.join(gt_folder, imagename)
    gt_file = Image.open(gt_path)

    rgb_im = img_file.convert('RGB')
    rgb_gt = gt_file.convert('RGB')
    #print(forest.dict_parent2node())

    forest = get_forest(img_file,sigma,K,min_size,neighbour)

    size = img_file.size
    width = size[0]
    height = size[1]

    #generate graph with black or white color
    area_dict = forest.dict_parent2node()
    counter = {}
    colors = {}
    for key in area_dict.keys():
        counter[key] = 0
        colors[key] = (255,255,255)
    for parent in area_dict.keys():
        area = area_dict[parent]
        for number in area:
            i = number % width
            j = (number - i) / width
            gr, gg, gb = rgb_gt.getpixel((i, j))
            if(gr<=128 and gg<=128 and gb<=128):
                counter[parent] += 1
    for parent in area_dict.keys():
        if(counter[parent] >= len(area_dict[parent])/2):
            colors[parent] = (0,0,0)
        else:
            colors[parent] = (255,255,255)

    image = generate_image(forest, size[0],size[1],colors)
    image.save("./foreground/foreground mark"+imagename+".png")

    IOU = calc_IOU(image,rgb_gt,width,height)
    return IOU

if __name__ == '__main__':

    namelist = [str(11+(i*100)) +".png" for i in range(0,10)]
    IOUlist =[]
    for graph in namelist:
        IOUlist.append(round(graphbased_segmentation(graph,K=120,min_size=20),2))
    print(IOUlist)