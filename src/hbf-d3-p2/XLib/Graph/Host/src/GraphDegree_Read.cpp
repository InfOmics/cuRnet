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
#include <iostream>
#include <sstream>
#include <iterator>
#include <XLib.hpp>
#include "../include/GraphDegree.hpp"

namespace graph {

void GraphDegree::readMatrixMarket(std::ifstream& fin, const int nof_lines) {
	fUtil::Progress progress(nof_lines);

	while (fin.peek() == '%')
		fileUtil::skipLines(fin);
	fileUtil::skipLines(fin);

	for (int lines = 0; lines < nof_lines; ++lines) {
		int index1, index2;
		fin >> index1 >> index2;
		fileUtil::skipLines(fin);
		index1--;
		index2--;

		OutDegrees[index1]++;
		if (Direction == EdgeType::UNDIRECTED)
			OutDegrees[index2]++;
		progress.next(lines + 1);
	}
}

void GraphDegree::readDimacs9(std::ifstream& fin, const int nof_lines) {
	fUtil::Progress progress(nof_lines);

	char c = fin.peek();
	int lines = 0;
	std::string nil;
	while (c != EOF) {

		if (c == 'a') {
			int index1, index2;
			fin >> nil >> index1 >> index2;
			index1--;
			index2--;

			OutDegrees[index1]++;
			if (Direction == EdgeType::UNDIRECTED)
				OutDegrees[index2]++;
			lines++;
			progress.next(lines + 1);
		}
		fileUtil::skipLines(fin);
		c = fin.peek();
	}
}

void GraphDegree::readDimacs10(std::ifstream& fin) {
	fUtil::Progress progress(V);

	while (fin.peek() == '%')
		fileUtil::skipLines(fin);
	fileUtil::skipLines(fin);

	for (int lines = 0; lines < V; lines++) {
		std::string str;
		std::getline(fin, str);

		std::istringstream stream(str);
		std::istream_iterator<std::string> iis(stream >> std::ws);

		OutDegrees[lines] = std::distance(iis, std::istream_iterator<std::string>());
		progress.next(lines + 1);
	}
}

void GraphDegree::readSnap(std::ifstream& fin, const int nof_lines) {
	fUtil::Progress progress(nof_lines);

	while (fin.peek() == '#')
		fileUtil::skipLines(fin);

	fUtil::UniqueMap<int, int> Map;
	for (int lines = 0; lines < nof_lines; lines++) {
		int ID1, ID2;
		fin >> ID1 >> ID2;

		int index1 = Map.insertValue(ID1);
		int index2 = Map.insertValue(ID2);

		OutDegrees[index1]++;
		if (Direction == EdgeType::UNDIRECTED)
			OutDegrees[index2]++;
		progress.next(lines + 1);
	}
}

void GraphDegree::readBinary(const char* File) {

}

} //@graph
