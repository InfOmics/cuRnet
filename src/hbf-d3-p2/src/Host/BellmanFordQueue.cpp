/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

H-BF is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include "Host/GraphSSSP.hpp"
#include <queue>

void GraphSSSP::BellmanFord_Queue_init() {
	Distance = new weight_t[V];
    parent = new int[V];
    BellmanFord_Queue_reset();
}

void GraphSSSP::BellmanFord_Queue_end() {
	delete[] Distance;
	delete[] parent;
}

void GraphSSSP::BellmanFord_Queue_reset() {
    std::fill(Distance, Distance + V, std::numeric_limits<weight_t>::infinity());
    std::fill(parent, parent + V, -1);
}

void GraphSSSP::BellmanFord_Result(float*& dist, int*& paths) {
    dist = Distance;
    paths = parent;

    /*auto visited = new bool[V]();
    for (int source = 0; source < 500; source++) {
        //std::cout << source <<std::endl;
        int pred = source;
        while (true) {
            visited[pred] = true;
            pred = parent[pred];
            if (pred == -1)
                break;
            if (visited[pred]) {
                int tmp = pred;
                std::cout << "loop" << std::endl;
                do {
                    std::cout << pred << ", ";
                    pred = parent[pred];
                } while (pred != tmp);
                std::cout << std::endl;
                break;
            }
        }
        std::fill(visited, visited + V, false);
    }
    std::cout << "----------------" << std::endl;*/
}

weight_t* GraphSSSP::BellmanFord_Result() {
    //for (int i = 0; i < V; i++)
    //    std::cout << i << "\t" << parent[i] << "\t" << Distance[i] << std::endl;



    /*auto visited = new bool[V]();
    for (int source = 0; source < 500; source++) {
        //std::cout << source <<std::endl;
        int pred = source;
        while (true) {
            visited[pred] = true;
            pred = parent[pred];
            if (pred == -1)
                break;
            if (visited[pred]) {
                int tmp = pred;
                std::cout << "loop" << std::endl;
                do {
                    std::cout << pred << ", ";
                    pred = parent[pred];
                } while (pred != tmp);
                std::cout << std::endl;
                break;
            }
        }
        std::fill(visited, visited + V, false);
    }
    std::cout << "----------------" << std::endl;*/
    return Distance;
}

void GraphSSSP::BellmanFord_Queue(const node_t source) {
	std::queue<node_t> Queue;
	Queue.push(source);

	Distance[source] = 0;
    parent[source] = -1;

    int count = 0;
	while (Queue.size() > 0) {
        node_t next = Queue.front();
        /*if (count == 1000) {
            count = 0;
            std::cout << Queue.size() << "\t" << Distance[next] << std::endl;
        }*/
		Queue.pop();
        //std::cout << "BF " << "\t" << next << std::endl;

		for (int i = OutNodes[next]; i < OutNodes[next + 1]; i++) {
			node_t dest = OutEdges[i];
        //    std::cout << "  e: " << "\t" << dest << std::endl;
            if (Relax(next, dest, Weights[i], Distance)) {
				Queue.push(dest);
                parent[dest] = next;
            }
		}
        count++;
	}
}

void GraphSSSP::BellmanFord_Frontier(const node_t source, std::vector<int>& Frontiers) {
	std::queue<node_t> Queue1;
	std::queue<node_t> Queue2;
	Queue1.push(source);

	Distance[source] = 0;

	int level = 0, visited = 0;
	while (Queue1.size() > 0) {

		Frontiers.push_back(Queue1.size());
		while (Queue1.size() > 0) {
			const node_t next = Queue1.front();
			Queue1.pop();
			visited++;
			for (int i = OutNodes[next]; i < OutNodes[next + 1]; i++) {
				const node_t dest = OutEdges[i];
                if (level == 1)
                    std::cout << "(" << next << "," << dest << ") -> " << Weights[i] << std::endl;
				if (Relax(next, dest, Weights[i], Distance))
					Queue2.push(dest);
			}
		}
		if (Queue2.size() > 0) {
			level++;
			Frontiers.push_back(Queue2.size());
		}

		while (Queue2.size() > 0) {
			const node_t next = Queue2.front();
			Queue2.pop();
			visited++;
			for (int i = OutNodes[next]; i < OutNodes[next + 1]; i++) {
				const node_t dest = OutEdges[i];
                if (level == 1)
                    std::cout << "(" << next << "," << dest << ") -> " << Weights[i] << std::endl;
				if (Relax(next, dest, Weights[i], Distance))
					Queue1.push(dest);
			}
		}
		if (Queue1.size() > 0)
			level++;
	}
}


bool GraphSSSP::Relax(const node_t u, const node_t v,
                      const weight_t weight, weight_t* Distance) {

    //std::cout << "(" << u << "," << v << ") : " << (Distance[u] + weight)
    //          <<  " " << Distance[v] << std::endl;
	if (Distance[u] + weight < Distance[v]) {
		Distance[v] = Distance[u] + weight;
		return true;
	}
	return false;
}
