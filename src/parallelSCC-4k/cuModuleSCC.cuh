#pragma once

#include <base_graph/base_graph.hpp>

#include "cudaUtil.cuh"

void cu_scc(
	curnet::base_graph &graph, 
	std::vector<int> &colors 
	);
	
void cu_bfs(
	curnet::base_graph *graph, 
	std::vector<std::vector<float>> &distance,
	std::vector<int> source
	);
