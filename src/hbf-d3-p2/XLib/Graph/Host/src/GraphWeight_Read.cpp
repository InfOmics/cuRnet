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
#include "../include/GraphWeight.hpp"
#include "XLib.hpp"
#include <random>

namespace graph {

/*void GraphWeight::readBoost(GraphBasePCSF& graph_boost) {


    using namespace boost;
    using    EdgeIterator = graph_traits<GraphBasePCSF>::edge_iterator;

    using PropMap = boost::property_map<GraphBasePCSF, boost::edge_weight_t>::type;
    PropMap edge_weight_map = get(boost::edge_weight, graph_boost);

    EdgeIterator it, end_it;
    int count = 0;
    for (boost::tie(it, end_it) = boost::edges(graph_boost); it != end_it; ++it) {
        graph_traits<GraphBasePCSF>::edge_descriptor edge = *it;
        COO_Edges[count][0] = boost::source(edge, graph_boost);
        COO_Edges[count][1] = boost::target(edge, graph_boost);
        Weights[count]      = edge_weight_map[edge];

        count++;
        std::cout << boost::source(edge, graph_boost)<<" "<< boost::target(edge, graph_boost)<< " "<< edge_weight_map[edge] << std::endl;
    }
    GraphSTD::ToCSR();


}*/

void GraphWeight::read(const char* File, const int nof_lines) {
    dimacs9 = false;
    GraphBase::read(File, nof_lines);

    if (!dimacs9) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        //std::uniform_int_distribution<weight_t> distribution (1, 100);
        std::uniform_real_distribution<weight_t> distribution (1.0f, 100.0f);
        std::generate(Weights, Weights + E, [&](){ return distribution(generator); });
        //for (int i = 0; i < E; i++)
        //    std::cout <<  Weights[i] << " ";
        for (int i = 0; i < E; i++)
             Weights[i] = i;
    }
}

void GraphWeight::readMatrixMarket(std::ifstream& fin, const int nof_lines) {
    GraphSTD::readMatrixMarket(fin, nof_lines);
}

void GraphWeight::readDimacs10(std::ifstream& fin) {
	GraphSTD::readDimacs10(fin);
}

void GraphWeight::readSnap(std::ifstream& fin, const int nof_lines) {
	GraphSTD::readSnap(fin, nof_lines);
}

void GraphWeight::readDimacs9(std::ifstream& fin, const int nof_lines) {
	fUtil::Progress progress(nof_lines);

	char c;
	int lines = 0;
	std::string nil;
	while ((c = fin.peek()) != EOF) {
		if (c == 'a') {
			node_t index1, index2;
			fin >> nil >> index1 >> index2 >> Weights[lines];

            COO_Edges[lines][0] = index1 - 1;
            COO_Edges[lines][1] = index2 - 1;

			lines++;
			progress.next(lines + 1);
		}
		fileUtil::skipLines(fin);
	}
	COOSize = lines;
    GraphSTD::ToCSR();
    dimacs9 = true;
}

#if __linux__

#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>

void GraphWeight::readBinary(const char* File) {

}

#endif

} //@graph
