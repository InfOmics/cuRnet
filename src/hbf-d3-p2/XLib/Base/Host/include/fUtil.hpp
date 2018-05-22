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
#pragma once

#if __linux__
    #include <unistd.h>
    #include <sys/syscall.h>
    #include <sys/resource.h>
#endif

#include <iostream>
#include <locale>			// numpunct<char>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>      	// setprecision
#include <cstdlib>			// exit
#include <ostream>			// color
#include <algorithm>		// sorting
#include <exception>		// sorting
#include <unordered_map>

#if !defined(__NVCC__)
	#define PRINT_ERR(ERR) "\n\n\033[91m--> "#ERR "\033[97m\n"
    #define PRINT_MSG(MSG) "\n\n\033[96m--> "#MSG "\033[97m\n"
#else
    #define PRINT_ERR(ERR) "\n\n--> "#ERR "\n"
    #define PRINT_MSG(MSG) "\n\n--> "#MSG "\n"
#endif

#define __ENABLE(VAL, EXPR) {       \
    if (VAL)  {                     \
        EXPR                        \
    }                               \
}

#define __PRINT(msg)  {             \
    std::cout << msg << std::endl;  \
}

#define __ERROR(msg)  {                                                         \
    std::cerr << std::endl << " ! ERROR : " << msg << std::endl << std::endl;   \
    std::exit(EXIT_FAILURE);                                                    \
}

#define __ERROR_LINE(msg)	{                                                               \
                            std::cerr << std::endl << " ! ERROR : " << msg                  \
                                      << " in " << __FILE__ << " : " << __func__            \
                                      << " (line: " << __LINE__ << ")" << endl << endl;     \
                            std::exit(EXIT_FAILURE);                                        \
                        }

#define	def_SWITCH(x)	switch (x) {			\
							case -1:			\
							case 0:	fun(1)		\
								break;			\
							case 1: fun(2)		\
								break;			\
							case 2: fun(4)		\
								break;			\
							case 3: fun(8)		\
								break;			\
							case 4: fun(16)		\
								break;			\
							case 5: fun(32)		\
								break;			\
							case 6: fun(64)		\
								break;			\
							case 7: fun(128)	\
								break;			\
							default: fun(32)	\
						}

#define	def_SWITCHB(x)	switch (x) {			\
							case -1:			\
							case 0:	funB(1)		\
								break;			\
							case 1: funB(2)		\
								break;			\
							case 2: funB(4)		\
								break;			\
							case 3: funB(8)		\
								break;			\
							case 4: funB(16)	\
								break;			\
							case 5: funB(32)	\
								break;			\
							case 6: funB(64)	\
								break;			\
							case 7: funB(128)	\
								break;			\
							default: fun(32)	\
						}

/// @namespace StreamModifier provide modifiers and support methods for std:ostream
namespace StreamModifier {
    /**
     * @enum Color change the color of the output stream
     */
    enum Color {
                           /** <table border=0><tr><td><div> Red </div></td><td><div style="background:#FF0000;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_RED       = 31, /** <table border=0><tr><td><div> Green </div></td><td><div style="background:#008000;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_GREEN     = 32, /** <table border=0><tr><td><div> Yellow </div></td><td><div style="background:#FFFF00;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_YELLOW    = 33, /** <table border=0><tr><td><div> Blue </div></td><td><div style="background:#0000FF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_BLUE      = 34, /** <table border=0><tr><td><div> Magenta </div></td><td><div style="background:#FF00FF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_MAGENTA   = 35, /** <table border=0><tr><td><div> Cyan </div></td><td><div style="background:#00FFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_CYAN      = 36, /** <table border=0><tr><td><div> Light Gray </div></td><td><div style="background:#D3D3D3;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_L_GRAY    = 37, /** <table border=0><tr><td><div> Dark Gray </div></td><td><div style="background:#A9A9A9;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_D_GREY    = 90, /** <table border=0><tr><td><div> Light Red </div></td><td><div style="background:#DC143C;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_L_RED     = 91, /** <table border=0><tr><td><div> Light Green </div></td><td><div style="background:#90EE90;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_L_GREEN   = 92, /** <table border=0><tr><td><div> Light Yellow </div></td><td><div style="background:#FFFFE0;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_L_YELLOW  = 93, /** <table border=0><tr><td><div> Light Blue </div></td><td><div style="background:#ADD8E6;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_L_BLUE    = 94, /** <table border=0><tr><td><div> Light Magenta </div></td><td><div style="background:#EE82EE;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_L_MAGENTA = 95, /** <table border=0><tr><td><div> Light Cyan </div></td><td><div style="background:#E0FFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_L_CYAN    = 96, /** <table border=0><tr><td><div> White </div></td><td><div style="background:#FFFFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
        FG_WHITE	 = 97, /** Default */
        FG_DEFAULT  = 39
    };

    /**
     * @enum Emph
     */
    enum Emph {
        SET_BOLD      = 1,
        SET_DIM       = 2,
		SET_UNDERLINE = 4,
        SET_RESET     = 0,
    };

    /// @cond
	std::ostream& operator<<(std::ostream& os, const Color& mod);
	std::ostream& operator<<(std::ostream& os, const Emph& mod);
    //struct myseps;
    /// @endcond

	void thousandSep();
	void resetSep();
    void fixedFloat();
    void scientificFloat();
}

namespace fUtil {

    template<bool PRINT>
    void print_info(const char* msg) {
        if (PRINT)
            std::cout << msg << std::endl;
    }

	template<typename T>
	std::string typeStringObj(T Obj);
	template<typename T>
	std::string typeString();

	void memInfoPrint(size_t total, size_t free, size_t Req);
	void memInfoHost(int Req);

	template<bool FAULT = true, typename T, typename R>
	bool Compare(T* ArrayA, R* ArrayB, const int size);

    /*template<class iterator_type>
    void dumpStrings(iterator_type it, iterator_type end)
    {
        while (it != end) {
            cout << *(it++) << endl;
        }
        for (; itr != end; ++itr) {
             process(*itr);
         }
    }*/

    template<bool FAULT = true, typename T, typename R>
    bool Compare(T* ArrayA, R* ArrayB, const int size, bool (*equalFunction)(T, R));

	template<bool FAULT = true, typename T, typename R>
	bool CompareAndSort(T* ArrayA, R* ArrayB, const int size);

	bool isDigit(std::string str);

	class Progress {
		private:
			long long int progressC, nextChunk, total;
			double fchunk;
		public:
			Progress(long long int total);
			~Progress();
			void next(long long int progress);
			void perCent(long long int progress);
	};

    template<typename T, typename R>
	class UniqueMap : public std::unordered_map<T, R> {
		public:
			R insertValue(T id);
	};
}

namespace fileUtil {

	void checkRegularFile(const char* File);
    void checkRegularFile(std::ifstream& fin);
	std::string extractFileName(std::string s);
	std::string extractFileExtension(std::string str);
	std::string extractFilePath(std::string str);

	long long int fileSize(const char* File);
	void skipLines(std::istream& fin, const int nof_lines = 1);
}

#include "impl/fUtil.i.hpp"
