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
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include "XLib.hpp"
#include "Host/GraphSSSP.hpp"

using namespace timer;

void GraphSSSP::BoostDijkstra(const int source) {
	using namespace boost;

	typedef adjacency_list < listS, vecS, directedS, no_property, property<edge_weight_t, node_t> > graph_t;
	typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
	typedef std::pair<node_t, node_t> Edge;

	Edge* edge_array = new Edge[E];
	int k = 0;
	for (int i = 0; i < V; i++) {
		for (int j = OutNodes[i]; j < OutNodes[i + 1]; j++)
			edge_array[k++] = Edge(i, OutEdges[j]);
	}

	graph_t g(edge_array, &edge_array[E], Weights, V);

	std::vector<vertex_descriptor> p(num_vertices(g));
	std::vector<int> d(num_vertices(g));
	vertex_descriptor s = vertex(source, g);

	Timer<HOST> TM;
    TM.start();

	dijkstra_shortest_paths(g, s,
            predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
            distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));

	TM.getTime("Boost Dijkstra");

    BellmanFord_Queue_init();
    BellmanFord_Queue(source);

	graph_traits < graph_t >::vertex_iterator vi, vend;
	int i = 0;
	for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
		if (d[*vi] != Distance[i++])
			__ERROR("distance error");
	}
    BellmanFord_Queue_end();
}
