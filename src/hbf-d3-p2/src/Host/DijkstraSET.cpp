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
#include <set>
#include "Host/GraphSSSP.hpp"
#include "XLib.hpp"
using namespace timer;

void GraphSSSP::DijkstraSET(const node_t source) {
	typedef std::pair<weight_t, node_t> Node;

	std::set<Node> PriorityQueue;
	PriorityQueue.insert(Node(0, source));

	dist_t* DijkstraDistance = new dist_t[V];
	std::fill(DijkstraDistance, DijkstraDistance + V, std::numeric_limits<dist_t>::max());
	DijkstraDistance[source] = 0;
	Timer<HOST> TM;
    TM.start();

	while (!PriorityQueue.empty()) {
		const node_t next = PriorityQueue.begin()->second;
		PriorityQueue.erase(PriorityQueue.begin());

		for (int j = OutNodes[next]; j < OutNodes[next + 1]; j++) {
			const node_t dest = OutEdges[j];

			if (DijkstraDistance[next] + Weights[j] < DijkstraDistance[dest]) {
				PriorityQueue.erase(Node(DijkstraDistance[dest], dest));
				DijkstraDistance[dest] = DijkstraDistance[next] + Weights[j];
				PriorityQueue.insert(Node(DijkstraDistance[dest], dest));
			}
		}
	}
	TM.getTime("Dijkstra SET");

    BellmanFord_Queue_init();
    BellmanFord_Queue(source);

    if (!std::equal(DijkstraDistance, DijkstraDistance + V, Distance))
        __ERROR("wrong distance")

    BellmanFord_Queue_end();
}
