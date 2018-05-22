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
namespace data_movement {

	template<typename T>
	__device__ __forceinline__ void RegInsert(T* Queue, const T item, int& size) {
		switch (size) {
			case 0: Queue[0] = item; break;
            case 1: Queue[1] = item; break;
            case 2: Queue[2] = item; break;
            case 3: Queue[3] = item; break;
            case 4: Queue[4] = item; break;
            case 5: Queue[5] = item; break;
            case 6: Queue[6] = item; break;
            case 7: Queue[7] = item; break;
            case 8: Queue[8] = item; break;
            case 9: Queue[9] = item; break;
            case 10: Queue[10] = item; break;
            case 11: Queue[11] = item; break;
            case 12: Queue[12] = item; break;
            case 13: Queue[13] = item; break;
            case 14: Queue[14] = item; break;
            case 15: Queue[15] = item; break;
            case 16: Queue[16] = item; break;
            case 17: Queue[17] = item; break;
            case 18: Queue[18] = item; break;
            case 19: Queue[19] = item; break;
            case 20: Queue[20] = item; break;
            case 21: Queue[21] = item; break;
            case 22: Queue[22] = item; break;
            case 23: Queue[23] = item; break;
            case 24: Queue[24] = item; break;
            case 25: Queue[25] = item; break;
            case 26: Queue[26] = item; break;
            case 27: Queue[27] = item; break;
            case 28: Queue[28] = item; break;
            case 29: Queue[29] = item; break;
            case 30: Queue[30] = item; break;
            case 31: Queue[31] = item; break;
            case 32: Queue[32] = item; break;
            case 33: Queue[33] = item; break;
            case 34: Queue[34] = item; break;
            case 35: Queue[35] = item; break;
            case 36: Queue[36] = item; break;
            case 37: Queue[37] = item; break;
            case 38: Queue[38] = item; break;
            case 39: Queue[39] = item; break;
            case 40: Queue[40] = item; break;
            case 41: Queue[41] = item; break;
            case 42: Queue[42] = item; break;
            case 43: Queue[43] = item; break;
            case 44: Queue[44] = item; break;
            case 45: Queue[45] = item; break;
            case 46: Queue[46] = item; break;
            case 47: Queue[47] = item; break;
            case 48: Queue[48] = item; break;
            case 49: Queue[49] = item; break;
            case 50: Queue[50] = item; break;
            case 51: Queue[51] = item; break;
            case 52: Queue[52] = item; break;
            case 53: Queue[53] = item; break;
            case 54: Queue[54] = item; break;
            case 55: Queue[55] = item; break;
            case 56: Queue[56] = item; break;
            case 57: Queue[57] = item; break;
            case 58: Queue[58] = item; break;
            case 59: Queue[59] = item; break;
            case 60: Queue[60] = item; break;
            case 61: Queue[61] = item; break;
            case 62: Queue[62] = item; break;
            case 63: Queue[63] = item; break;
        }
        size++;
    }
} //@data_movement
