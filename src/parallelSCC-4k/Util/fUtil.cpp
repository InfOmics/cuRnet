#include "fUtil.h"

namespace scc4k{

namespace Color {
	std::ostream& operator<<(std::ostream& os, const Code& mod) {
		return os << "\033[" << (int) mod << "m";
	}
}

namespace SetFormat {
	std::ostream& operator<<(std::ostream& os, const CodeS& mod) {
		return os << "\033[" << (int) mod << "m";
	}
}

namespace fUtil {

	void memInfoPrint(size_t total, size_t free, size_t Req) {
			std::cout	<< "  Total Memory:\t\t" << (total >> 20)	<< " MB\n"
						<< "   Free Memory:\t\t" << (free >> 20)	<< " MB\n"
						<< "Request memory:\t\t" << (Req >> 20)		<< " MB\n"
						<< "   Request (%):\t\t" << ((Req >> 20) * 100) / (total >> 20) << " %\n\n";
		if (Req > free)
			error(" ! Memory too low");
	}

	void memInfoHost(int Req) {
		long pages = sysconf(_SC_PHYS_PAGES);
		long page_size = sysconf(_SC_PAGE_SIZE);
		memInfoPrint(pages * page_size, pages * page_size - 100 * (1 << 20), Req);
	}

// --------------------------- PRINT ---------------------------------------------------

	std::string extractFileName(std::string s) {
		std::string path(s, s.length());
		int found = path.find_last_of(".");
		std::string name2 = path.substr(0, found);

		found = name2.find_last_of("/");
		if (found >= 0)
			name2 = name2.substr(found + 1);

		return name2;
	}

// --------------------------- MATH ---------------------------------------------------

	inline unsigned nearestPower2(unsigned v) {
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return v;
	}

	unsigned log_2(unsigned n) {
		if (n <= 1)
			return 0;
		return 31 - __builtin_clz(n);
	}

	inline bool isPositiveInteger(const std::string& s) {
		for (unsigned i = 0; i < s.size(); i++) {
			if (!isdigit(s.at(i)))
				return false;
		}
		return true;
	}

	float perCent(int part, int max) {
		return ((float) part / max) * 100;
	}
}

}
