#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
//#include "../hbf-d3-p2/XLib/Graph/EdgeT.cuh"

namespace curnet
{	
	using namespace std;

	enum direction { Directed, Undirected };

	struct int2
	{
		int x;
		int y;
	};
	
	struct Edge_T
	{
		int x;
		float y;
	};
	
	// think about going into template-way, MAYBE later.
	//template <class T, class W>
	class base_graph {
	public:
		base_graph(int n_edges, direction direction);
		//base_graph(size_t n_vertices, size_t n_edges, direction direction);
		//base_graph(int* vertices, size_t n_vertices, int* edges, size_t n_edges, direction direction);

		// Three-way rule: copy-constructor, assignment constructor and destructor
		base_graph(const base_graph& graph);
		base_graph& operator= (const base_graph& rhs);
		~base_graph();

		bool csr();
		bool coo();
		bool to_csr();
		bool to_coo();
		
		void v(int v);
		int v();
		int e();
		int2* edges();
		float* weights();
		
		int* v_out();
		int* e_out();
		int* w_out();
		
		int* v_in();
		int* e_in();
		int* w_in();
		
		std::unordered_map< string, int> & node_to_id();
		std::vector< string> & id_to_node();
		//const std::unordered_map< string, int> & node_ids() const;
		
		std::vector<int> bfs(std::vector<int> sources);
		std::vector<int> scc();
		
		Edge_T* weights_out();

		

	private:
		std::unordered_map< std::string, int > map_node_to_id;
		std::vector< std::string > vector_id_to_node;

		int n_v;
		int n_e;
		std::vector<int> vertices_weights;

		bool is_coo;
		std::vector<int2> edges_coo;// int2 because COO format
		std::vector<float> edges_weights_coo;// weights differs between formats
		
		std::vector<int> edges_in_degree;
		std::vector<int> edges_out_degree;

		// CSR representation
		// OUT nodes
		bool is_csr;
		std::vector<int> vertices_out_csr;
		std::vector<int> edges_out_csr;        // int  because CSR format
		std::vector<int> edges_out_weights_csr; // weights differs between formats
		
		// IN nodes
		std::vector<int> vertices_in_csr;
		std::vector<int> edges_in_csr;         // int  because CSR format
		std::vector<int> edges_in_weights_csr;  // weights differs between formats
		
		vector<Edge_T> edges_out_weights;  // weights differs between formats
		

		enum direction direction;
	};

}
