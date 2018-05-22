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
#include "../include/GraphNamespace.hpp"
#include "XLib.hpp"

#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>

namespace graph {

namespace {
	void getMatrixMarketHeader(std::ifstream& fin, node_t &V, int &nof_lines, EdgeType &FileDirection);
	void getDimacs9Header	  (std::ifstream& fin, node_t &V, int &nof_lines);
	void getDimacs10Header	  (std::ifstream& fin, node_t &V, int &nof_lines, EdgeType &FileDirection);
	void getSnapHeader		  (std::ifstream& fin, node_t &V, int &nof_lines, EdgeType &FileDirection);
	void getPGSolverHeader	  (std::ifstream& fin, node_t &V, int &nof_lines, EdgeType &FileDirection);
	void getBinaryHeader	  (const char* File, node_t &V, edge_t &E, EdgeType &FileDirection);
}

void readHeader(const char* File, node_t &V, edge_t &E, int &nof_lines, EdgeType& UserDirection) {
	fileUtil::checkRegularFile(File);
	long long int size = fileUtil::fileSize(File);

	StreamModifier::thousandSep();

	std::cout << std::endl << "Read Header:\t" << fileUtil::extractFileName(File) << "\tSize: " << size / (1024 * 1024) << " MB" << std::endl;
	std::ifstream fin(File);
	std::string s;
	fin >> s;
	fin.seekg(std::ios::beg);

	EdgeType FileDirection = EdgeType::UNDEF_EDGE_TYPE;

	bool binFile = false;
	if (fileUtil::extractFileExtension(File).compare(".bin") == 0) {
		getBinaryHeader(File, V, E, FileDirection);
		binFile = true;
	}
	else if (s.compare("c") == 0 || s.compare("p") == 0)
		getDimacs9Header(fin, V, nof_lines);
	else if (s.compare("##") == 0)
		getPGSolverHeader(fin, V, nof_lines, FileDirection);
	else if (s.compare("%%MatrixMarket") == 0)
		getMatrixMarketHeader(fin, V, nof_lines, FileDirection);
	else if (s.compare("#") == 0)
		getSnapHeader(fin, V, nof_lines, FileDirection);
	else if (s.compare("%") == 0 || fUtil::isDigit(s.c_str()))
		getDimacs10Header(fin, V, nof_lines, FileDirection);
	else
		__ERROR(" Error. Graph Type not recognized: " << File << " " << s)

	fin.close();
	bool undirectedFlag = UserDirection == EdgeType::UNDIRECTED || (UserDirection == EdgeType::UNDEF_EDGE_TYPE && FileDirection == EdgeType::UNDIRECTED);

	if (!binFile)
		E = undirectedFlag ? nof_lines * 2 : nof_lines;
	std::string graphDir = undirectedFlag ? "GraphType: Undirected " : "GraphType: Directed ";

	if (UserDirection != EdgeType::UNDEF_EDGE_TYPE)
		graphDir.append("(User Def)");
	else if (FileDirection != EdgeType::UNDEF_EDGE_TYPE) {
		graphDir.append("(File Def)");
		UserDirection = FileDirection;
	} else {
		graphDir.append("(UnDef)");
		UserDirection = EdgeType::DIRECTED;
	}

	std::cout	<< std::endl << "\tNodes: " << V << "\tEdges: " << E << '\t' << graphDir
				<< "\tDegree AVG: " << std::fixed << std::setprecision(1) << (float) E  / V << std::endl << std::endl;
	StreamModifier::resetSep();
}

namespace {

void getPGSolverHeader(std::ifstream& fin, node_t &V, int &nof_lines, EdgeType& FileDirection) {
	fileUtil::skipLines(fin);
	FileDirection = EdgeType::DIRECTED;

	node_t p0, p1;
	fin >> p0 >> p1 >> nof_lines;
	V = p0 + p1;
}

void getMatrixMarketHeader(std::ifstream& fin, node_t &V, int &nof_lines, EdgeType& FileDirection) {
	std::string MMline;
	std::getline(fin, MMline);
	FileDirection = MMline.find("symmetric") != std::string::npos ? EdgeType::UNDIRECTED : EdgeType::DIRECTED;
    /*if (MMline.find("real") != std::string::npos)
        FileAttributeType = AttributeType::REAL;
    else if (MMline.find("integer") != std::string::npos)
        FileAttributeType = AttributeType::INTEGER;
    else
        FileAttributeType = AttributeType::BINARY;*/
	while (fin.peek() == '%')
		fileUtil::skipLines(fin);

	fin >> V >> MMline >> nof_lines;
}

void getDimacs9Header(std::ifstream& fin, node_t &V, int &nof_lines) {
	while (fin.peek() == 'c')
		fileUtil::skipLines(fin);

	std::string nil;
	fin >> nil >> nil >> V >> nof_lines;
    //FileAttributeType = AttributeType::INTEGER;
}

void getDimacs10Header(std::ifstream& fin, node_t &V, int &nof_lines, EdgeType &FileDirection) {
	while (fin.peek() == '%')
		fileUtil::skipLines(fin);

	std::string str;
	fin >> V >> nof_lines >> str;
	FileDirection = str.compare("100") == 0 ? EdgeType::DIRECTED : EdgeType::UNDIRECTED;
    //FileAttributeType = AttributeType::BINARY;
}

void getSnapHeader(std::ifstream& fin, int &V, int &nof_lines, EdgeType& FileDirection) {
	std::string tmp;
	fin >> tmp >> tmp;
	FileDirection = tmp.compare("Undirected") == 0 ? EdgeType::UNDIRECTED : EdgeType::DIRECTED;
	fileUtil::skipLines(fin);

	while (fin.peek() == '#') {
		std::getline(fin, tmp);
		if (tmp.substr(2, 6).compare("Nodes:") == 0) {
			std::istringstream stream(tmp);
			stream >> tmp >> tmp >> V >> tmp >> nof_lines;
			break;
		}
	}
    fileUtil::skipLines(fin);
    std::string MMline;
	std::getline(fin, MMline);
    //FileAttributeType = MMline.find("Sign") != std::string::npos ?
    //                        AttributeType::SIGN :: AttributeType::BINARY;
}


void getBinaryHeader(const char* File, int &V, int &E, EdgeType& FileDirection) {
	const int fileSize = fileUtil::fileSize(File);
	FILE *fp = fopen(File, "r");

	int* memory_mapped = (int*) mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fp->_fileno, 0);
	if (memory_mapped == MAP_FAILED)
		__ERROR("memory_mapped error");

	V =	memory_mapped[0];
	E = memory_mapped[1];
	FileDirection = static_cast<EdgeType>(memory_mapped[2]);

	munmap(memory_mapped, fileSize);
	fclose(fp);
}

} //@anonimous
} //@graph
