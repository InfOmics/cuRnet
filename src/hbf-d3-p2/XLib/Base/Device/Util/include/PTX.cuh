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

/** \namespace PTX
 *  provide simple interfaces for low-level PTX instructions
 */
namespace PTX {
// ---------------------------- THREAD PTX -------------------------------------

/** \fn unsigned int LaneID()
 *  \brief return the lane ID within the current warp
 *
 *  Provide the thread ID within the current warp (called lane).
 *  \return identification ID in the range 0 &le; ID &le; 31
 */
__device__ __forceinline__ unsigned int LaneID();

/** \fn void ThreadExit()
 *  \brief terminate the current thread
 */
__device__ __forceinline__ void ThreadExit();

// --------------------------------- MATH --------------------------------------

/** \fn unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
 *  \brief sum three operands with one instruction
 *
 *  Sum three operand with one instruction. Only in Maxwell architecture
 *  IADD3 is implemented in hardware, otherwise involves multiple instructions.
 *  \return x + y + z
 */
__device__ __forceinline__
unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z);

/** \fn unsigned int __fms(unsigned int word)
 *  \brief find most significant bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position in the range: 0 &le; bitPosition &le; 31.
 *  0xFFFFFFFF if no bit is found.
 */
__device__ __forceinline__ unsigned int __fms(unsigned int word);

/** \fn unsigned int __fms(unsigned long long int dword)
 *  \brief find most significant bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position in the range: 0 &le; bitPosition &le; 63.
 *          0xFFFFFFFF if no bit is found.
 */
__device__ __forceinline__ unsigned int __fms(unsigned long long int dword);

/** \fn unsigned int __fms(int word)
 *  \brief find most significant non-sign bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position of the most significant 0 for negative
 *          inputs and the most significant 1 for non-negative inputs
 *          in the range: 0 &le; bitPosition &le; 30.
 *          0xFFFFFFFF if no bit is found.
 */
__device__ __forceinline__ unsigned int __fms(int word);

/** \fn unsigned int __fms(long long int word)
 *  \brief find most significant non-sign bit
 *
 *  Calculate the bit position of the most significant 1.
 *  \return the bit position of the most significant 0 for negative
 *          inputs and the most significant 1 for non-negative inputs
 *          in the range: 0 &le; bitPosition &le; 62.
 *          0xFFFFFFFF if no bit is found.
 */
__device__ __forceinline__ unsigned int __fms(long long int dword);

/** \fn unsigned int LaneMaskEQ()
 *  \brief 32-bit mask with bit set in position equal to the thread's
 *         lane number in the warp
 *  \return 1 << laneid
 */
__device__ __forceinline__ unsigned int LaneMaskEQ();

/** \fn unsigned int LaneMaskLT()
 *  \brief 32-bit mask with bits set in positions less than the thread's lane
 *         number in the warp
 *  \return (1 << laneid) - 1
 */
__device__ __forceinline__ unsigned int LaneMaskLT();

/** \fn unsigned int LaneMaskLE()
 *  \brief 32-bit mask with bits set in positions less than or equal to the
 *         thread's lane number in the warp
 *  \return (1 << (laneid + 1)) - 1
 */
__device__ __forceinline__ unsigned int LaneMaskLE();

/** \fn unsigned int LaneMaskGT()
 *  \brief 32-bit mask with bit set in position equal to the thread's
 *         lane number in the warp
 *  \return ~((1 << (laneid + 1)) - 1)
 */
__device__ __forceinline__ unsigned int LaneMaskGT();

/** \fn unsigned int LaneMaskGE()
 *  \brief 32-bit mask with bits set in positions greater than or equal to the
 *         thread's lane number in the warp
 *  \return ~((1 << laneid) - 1)
 */
__device__ __forceinline__ unsigned int LaneMaskGE();

} //@PTX

#include "impl/PTX.i.cuh"
