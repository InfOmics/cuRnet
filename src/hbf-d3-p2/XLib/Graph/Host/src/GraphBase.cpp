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
#include "XLib.hpp"
#include "../include/GraphBase.hpp"

namespace graph {

void GraphBase::read(const char* File, const int nof_lines) {
	std::cout << "Reading Graph File..." << std::flush;

	if (fileUtil::extractFileExtension(File).compare(".bin") == 0) {
		readBinary(File);
		return;
	}

	std::ifstream fin(File);
	std::string s;
	fin >> s;
	fin.seekg(std::ios::beg);

	//MatrixMarket
	if (s.compare("%%MatrixMarket") == 0)
		readMatrixMarket(fin, nof_lines);
	//Dimacs10
	else if (s.compare("%") == 0 || fUtil::isDigit(s))
		readDimacs10(fin);
	//Dimacs9
	else if (s.compare("c") == 0 || s.compare("p") == 0)
		readDimacs9(fin, nof_lines);
	//SNAP
	else if (s.compare("#") == 0)
		readSnap(fin, nof_lines);

	fin.close();
}

} //@graph
