#include "base_graph.hpp"
#include <memory>
#include <algorithm>
#include <numeric>

#include <iostream>
#include <deque>
#include <stack>

namespace curnet
{
	base_graph::base_graph(int p_n_edges, enum direction p_direction)
	:
		is_coo(true),
		is_csr(false),
		direction(p_direction),
		n_e(p_n_edges),
		edges_coo( p_n_edges ),
		edges_weights_coo( p_n_edges ),
		edges_in_csr(p_n_edges),
		edges_in_weights_csr(p_n_edges),
		edges_out_csr(p_n_edges),
		edges_out_weights_csr(p_n_edges),
		edges_out_weights(p_n_edges)
	{
	}
	/*8
	base_graph::base_graph(const base_graph& g)
	:
		node_id(g.node_id),
		n_v(g.n_v),
		n_e(g.n_e),
		vertices_weights(g.vertices_weights),
		is_coo(g.is_coo),
		edges_coo(g.edges_coo),
		edges_weights_coo(g.edges_weights_coo),
		
		edges_in_degree(g.edges_in_degree),
		edges_out_degree(g.edges_out_degree),

		// CSR representation
		// OUT nodes
		is_csr(g.is_csr),
		vertices_out_csr(g.vertices_out_csr),
		edges_out_csr(g.edges_out_csr),               // int  because CSR format
		edges_out_weights_csr(g.edges_out_weights_csr), // weights differs between formats
		
		// IN nodes
		vertices_in_csr(g.vertices_in_csr),
		edges_in_csr(g.edges_in_csr),
		edges_in_weights_csr(g.edges_in_weights_csr),
		
		edges_out_weights(g.edges_out_weights),
		
		direction(g.direction)
	{
	}*/
	
	base_graph::~base_graph()
	{
	}
	
	bool base_graph::csr()
	{
		return is_csr;
	}
	
	bool base_graph::coo()
	{
		return is_coo;
	}
	
	bool base_graph::to_csr()
	{
		// already done?
		if(csr()) return true;

		// at this moment, only COO->CSR supported
		if(!coo()) return false;
		
		// build help structures
		for (auto v : edges_coo) {
			edges_out_degree[v.x]++;
			
			if (direction == Undirected)
				edges_out_degree[v.y]++;
			else
				edges_in_degree[v.y]++;
		}
		
		std::unique_ptr<int[]> tmp(new int[n_v]());
		{
			// compute OUT structure for CSR
			edges_out_csr[0] = 0;
			std::partial_sum(edges_out_degree.data(), edges_out_degree.data() + n_v, vertices_out_csr.data() + 1);
			for (int i = 0; i < edges_coo.size(); ++i)
			{
				int2 v = edges_coo[i];

				edges_out_weights_csr[ vertices_out_csr[v.x] + tmp[v.x] ] = edges_weights_coo[i];
				edges_out_weights[ vertices_out_csr[v.x] + tmp[v.x] ].x = v.y;
				edges_out_weights[ vertices_out_csr[v.x] + tmp[v.x] ].y = edges_weights_coo[i];
				edges_out_csr[ vertices_out_csr[v.x] + tmp[v.x]++ ] = v.y;
				if (direction == Undirected)
				{
					edges_out_weights_csr[ vertices_out_csr[v.y] + tmp[v.y] ] = edges_weights_coo[i];
					edges_out_weights[ vertices_out_csr[v.y] + tmp[v.y] ].x = v.x;
					edges_out_weights[ vertices_out_csr[v.y] + tmp[v.y] ].y = edges_weights_coo[i];
					edges_out_csr[ vertices_out_csr[v.y] + tmp[v.y]++ ] = v.x;
				}
			}
		}

		if (direction == Directed)
		{
			// compute IN structure for CSR
			// reset values
			std::fill(tmp.get(), tmp.get() + n_v, 0);
			
			edges_in_csr[0] = 0;
			std::partial_sum(edges_in_degree.data(), edges_in_degree.data() + n_v, vertices_in_csr.data() + 1);
			
			for (int i = 0; i < edges_coo.size(); ++i)
			{
				int2 v = edges_coo[i];
				edges_out_weights[ vertices_in_csr[v.y] + tmp[v.y] ].x = v.x;
				edges_out_weights[ vertices_in_csr[v.y] + tmp[v.y] ].y = edges_weights_coo[i];
				
				edges_in_weights_csr[ vertices_in_csr[v.y] + tmp[v.y] ] = edges_weights_coo[i];
				edges_in_csr[ vertices_in_csr[v.y] + tmp[v.y]++ ] = v.x;
			}
		}
		
		return true;
	}
	
	int2* base_graph::edges()
	{
		return edges_coo.data();
	}
	float* base_graph::weights()
	{
		return edges_weights_coo.data();
	}
	
	void base_graph::v(int n_v)
	{
		//TODO: reset?
		this->n_v = n_v;
		
		edges_out_degree.resize(n_v);
		vertices_out_csr.resize(n_v + 1);
		edges_in_degree.resize(n_v);
		vertices_in_csr.resize(n_v + 1);
	}
	
	int base_graph::v()
	{
		return n_v;
	}
	
	int base_graph::e()
	{
		return n_e;
	}
	
	int* base_graph::v_out()
	{
		return vertices_out_csr.data();
	}
	int* base_graph::e_out()
	{
		return edges_out_csr.data();
	}
	int* base_graph::w_out()
	{
		return edges_out_weights_csr.data();
	}
	
	int* base_graph::v_in()
	{
		return vertices_in_csr.data();
	}
	int* base_graph::e_in()
	{
		return edges_in_csr.data();
	}
	int* base_graph::w_in()
	{
		return edges_in_weights_csr.data();
	}
	Edge_T* base_graph::weights_out()
	{
		return edges_out_weights.data();
	}
	
	std::unordered_map< string, int>& base_graph::node_to_id() { return map_node_to_id; }
	std::vector< string>& base_graph::id_to_node() { return vector_id_to_node; }
//	const std::unordered_map< string, int>& base_graph::node_ids() const { return node_id; }	

	std::vector<int> base_graph::bfs(std::vector<int> sources)
	{
		std::deque<int> frontier(sources.begin(), sources.end());
		const int INF = std::numeric_limits<unsigned>::max();
		std::vector<int> distances(v(), INF);
		int max_distance = 0;
		for(auto v : sources) distances[v] = 0;
		
		while(!frontier.empty())
		{
			int v = frontier.front();
			frontier.pop_front();
			for(int i = vertices_out_csr[v]; i < vertices_out_csr[v+1]; ++i)
			{
				int v2 = edges_out_csr[i];
				if(distances[v2] == INF)
				{
					distances[v2] = distances[v] + 1;
					max_distance = max(distances[v2], max_distance);
					frontier.push_back(v2);
				}
			}
		}
		
		return distances;
	}
	
	std::vector<int> base_graph::scc()
	{
		std::vector<int> SCC(v(), 0);
		
		std::vector<int> nodes_idx(v(), 0);
		std::vector<int> min_dist(v(), 0);
		std::vector<bool> is_in_SCC(v(), false);

		std::stack<int> stack;
		std::stack<int> explored;

		//keep tracking of which neighbor we visited: when todo[i] == v_out[i+i], then all neighbors visited!
		std::vector<int> todo(vertices_out_csr.begin(), vertices_out_csr.end());
		int idxDist = 1;
		for(int w = 0; w < v(); w++)
		{
			if(nodes_idx[w] == 0) // not visited yet
			{
				//starting point
				explored.push(w);
				min_dist[w] = nodes_idx[w] = idxDist++;
				stack.push(w);
				is_in_SCC[w] = true;
				
				while(!explored.empty())
				{
					int h = explored.top();
					if(todo[h] < vertices_out_csr[h+1])//not visited all neighbors yet!
					{
						int s = edges_out_csr[todo[h]++];//get the neighbor
						if(nodes_idx[s] == 0)
						{
							//not visited: update distance
							min_dist[s] = nodes_idx[s] = idxDist++;
							//push on the stack
							stack.push(s);
							//push on the explored queue
							explored.push(s);
							
							//keep track that is on processing
							is_in_SCC[s] = true;
						}
						else if(nodes_idx[s] < nodes_idx[h] && is_in_SCC[s])
						{
							min_dist[h] = std::min(min_dist[h], nodes_idx[s]);
						}
					}
					else
					{
						//all neighbors visited, stop the search!
						explored.pop();
						if(min_dist[h] == nodes_idx[h])
						{
							//std::cout << "SCC FOUND!: = { ";
							while(!stack.empty() && nodes_idx[stack.top()] >= nodes_idx[h])
							{
								is_in_SCC[stack.top()] = false;
								SCC[stack.top()] = nodes_idx[h];// get unique ID
								//std::cout << " " << stack.top();
								stack.pop();
							}
							//std::cout << " } " << std::endl;
						}
						if(!explored.empty()) min_dist[explored.top()] = std::min(min_dist[explored.top()], min_dist[h]);
					}
				}
			}
		}

		return SCC;
	}

}
