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
namespace parsing {
namespace {

template<int N, int MUL, int INDEX = 0>
struct fastStringToIntStr {
    static inline int Aux(const char* str) {
        return (str[INDEX] - '0') * MUL + fastStringToIntStr<N - 1, INDEX + 1, MUL / 10>::Aux(str);
    }
};
template<int MUL, int INDEX>
struct fastStringToIntStr<1, MUL, INDEX> {
    static inline int Aux(const char* str) { return (str[INDEX] - '0'); }
};

} //@anonymous

inline int fastStringToInt(char* str, const int len) {
    switch(len) {
        case 10: return fastStringToIntStr<10, 1000000000>::Aux(str);
        case 9: return fastStringToIntStr<9, 100000000>::Aux(str);
        case 8: return fastStringToIntStr<8, 10000000>::Aux(str);
        case 7: return fastStringToIntStr<7, 1000000>::Aux(str);
        case 6: return fastStringToIntStr<6, 100000>::Aux(str);
        case 5: return fastStringToIntStr<5, 10000>::Aux(str);
        case 4: return fastStringToIntStr<4, 1000>::Aux(str);
        case 3: return fastStringToIntStr<3, 100>::Aux(str);
        case 2: return fastStringToIntStr<2, 10>::Aux(str);
        case 1: return fastStringToIntStr<1, 1>::Aux(str);
        default: return 0;
    }
}

namespace {

template<int LENGHT>
struct fastIntToStr {
    static inline void Aux(const unsigned int value, char* Buffer) {
        static const char digits[201] = "0001020304050607080910111213141516171819"
                                        "2021222324252627282930313233343536373839"
                                        "4041424344454647484950515253545556575859"
                                        "6061626364656667686970717273747576777879"
                                        "8081828384858687888990919293949596979899";

        const unsigned int i = LENGHT <= 2 ? (value << 1) : ((value % 100u) << 1);
        Buffer[LENGHT - 1] = digits[i + 1];
        Buffer[LENGHT - 2] = digits[i];
        fastIntToStr<LENGHT - 2>::Aux(value / 100, Buffer);
    }
};
template<>
struct fastIntToStr<1> {
    static inline void Aux(const int value, char* Buffer) {
        static const char digits[11] = "0123456789";
        Buffer[0] = digits[value];
    }
};
template<>
struct fastIntToStr<0> {
    static inline void Aux(__attribute__((unused)) const int value, __attribute__((unused)) char* Buffer) {}
};

} //@anonymous

inline void fastIntToString(const unsigned int value, char*& Buffer) {
    const int len = value < 10 ? 1 :
                    (value < 100) ? 2 :
                    (value < 1000) ? 3 :
                    (value < 10000) ? 4 :
                    (value < 100000) ? 5 :
                    (value < 1000000) ? 6 :
                    (value < 10000000) ? 7 :
                    (value < 100000000) ? 8 :
                    (value < 1000000000) ? 9 : 10;

    switch(len) {
    case 10: fastIntToStr<10>::Aux(value, Buffer); break;
        case 9: fastIntToStr<9>::Aux(value, Buffer); break;
        case 8: fastIntToStr<8>::Aux(value, Buffer); break;
        case 7: fastIntToStr<7>::Aux(value, Buffer); break;
        case 6: fastIntToStr<6>::Aux(value, Buffer); break;
        case 5: fastIntToStr<5>::Aux(value, Buffer); break;
        case 4: fastIntToStr<4>::Aux(value, Buffer); break;
        case 3: fastIntToStr<3>::Aux(value, Buffer); break;
        case 2: fastIntToStr<2>::Aux(value, Buffer); break;
        case 1: fastIntToStr<1>::Aux(value, Buffer);
    }
    Buffer += len;
}

} //@parsing
