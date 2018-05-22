#pragma once

#include <limits>

const int N_OF_TESTS  =	1;	// NUMBER OF TESTS
#define          SCC_MAX_CONCURR_TH		45056
#define	               BLOCKDIM		256		// (important: >= 256 for INTER_BLOCK_SYNC=1)

// -------------------------------------------------------------------------------------
typedef unsigned dist_t;			// ONLY UNSIGNED TYPE	(suggested unsigned short for high-diameter graph, unsigned char for low-diamter graph)

/*typedef struct color_struct
{
	int color;
	int colorF;
	int colorB;
} color_t;*/
typedef int color_t;

enum VisitType
{
	SCC_Decomposition,
	ColorUpdate,
	Coloring,
	BFS
};

//#define   	     		 MIN_VW		4		// Min Virtual Warp
//#define   	             MAX_VW		32		// Max Virtual Warp (suggested: 32)

const int             REG_QUEUE  = 	32;
#define                    SAFE		1		// Check for overflow in REG_QUEUE	(suggested: 1 for high-degree graph, 0 otherwise)

//const bool     DUPLICATE_REMOVE  =	0;		// Remove duplicate vertices in the frontier
const bool     INTER_BLOCK_SYNC  = 	0;		// Inter-block Synchronization	(suggested: 1 for high-diameter, 0 otherwise)

#define STORE_MODE (FrontierWrite::SHARED_WARP)		// SIMPLE, SHARED_WARP, SHARED_BLOCK

// ------------------------------ DYNAMIC PARALLELISM -----------------------------------

//const bool DYNAMIC_PARALLELISM  =	0;	 	//(suggested: 1 for high-degree graph, 0 otherwise)
const int           THRESHOLD_G =	400000;	// degree threshold to active DYNAMIC_PARALLELISM
//const int           THRESHOLD_G =	1;	// degree threshold to active DYNAMIC_PARALLELISM
const int 	   RESERVED_BLOCKS  =   2;
const int 	   LAUNCHED_BLOCKS  =   2;

// ----------------------------- ADVANCED CONFIGURATION -----------------------

#define                  ATOMICCAS	0

const int         ITEM_PER_WARP  = 	1;		// Iteration per Warp in the Frontier

const bool 	          BLOCK_BFS  =	0;
const int  BLOCK_FRONTIER_LIMIT  =  4096;	//2031 max

// ---------------------------- DEBUG and help constant ---------------------------------------

const bool            COUNT_DUP  =	0;		// count the number of duplicates found with the hash table

const bool CHECK_TRAVERSED_EDGES = 	true;
const bool        PRINT_FRONTIER = 	false;

const dist_t INF = std::numeric_limits<dist_t>::max();

#define LOG_TIMES_TO_FILE       0
#define DEEP_LOG        0
//#define MIN_VW 4
//const bool DYNAMIC_PARALLELISM = 0;
//const int DUPLICATE_REMOVE = 0;
