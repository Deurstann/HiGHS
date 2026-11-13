#include "ipm/ipx/sparse_utils.h"
#include <cassert>

namespace ipx {

// Mark of a column visited by a failed AugmentingPath search.
static constexpr Int kDeadColumn = -2;

Int DepthFirstSearch(Int istart, const Int* Ap, const Int* Ai,
                     const Int* colmap, Int top, Int* istack, Int* marked,
                     Int marker, Int* work) {
    assert(marked[istart] != marker);
    Int* pstack = work;
    Int head = 0;
    istack[head] = istart;
    while (head >= 0) {
        const Int i = istack[head];
        const Int j = colmap ? colmap[i] : i;
        if (marked[i] != marker) {
            marked[i] = marker;
            pstack[head] = j >= 0 ? Ap[j] : 0;
        }
        bool done = true;
        const Int pbeg = pstack[head];
        const Int pend = j >= 0 ? Ap[j+1] : 0;
        for (Int p = pbeg; p < pend; p++) {
            Int inext = Ai[p];
            if (marked[inext] == marker)
                continue;
            pstack[head] = p+1;
            istack[++head] = inext;
            done = false;
            break;
        }
        if (done) {
            head--;
            istack[--top] = i;
        }
    }
    return top;
}

bool AugmentingPath(Int jstart, const Int* Ap, const Int* Ai, Int* jmatch,
                    Int* cheap, Int* marked, Int* work, Int* work2, Int* work3,
                    Int* work4) {
    // istack holds row indices without duplicates.
    // jstack holds jstart and up to m indices from jmatch.
    // pstack holds the corresponding column pointers.
    // visited holds jstart and the matched columns visited, at most m+1.
    Int* istack = work;
    Int* jstack = work2;
    Int* pstack = work3;
    Int* visited = work4;
    Int num_visited = 0;
    bool found = false;
    Int head = 0;
    jstack[head] = jstart;
    while (head >= 0) {
        const Int j = jstack[head];
        if (marked[j] != jstart) { // first time j visited in this path
            marked[j] = jstart;
            visited[num_visited++] = j;
            Int i, p;
            for (p = cheap[j]; p < Ap[j+1] && !found; p++) {
                i = Ai[p];
                found = jmatch[i] == -1;
            }
            cheap[j] = p;
            if (found) {
                istack[head] = i;
                break;
            }
            pstack[head] = Ap[j]; // start depth-first search from j
        }
        const Int pbeg = pstack[head];
        const Int pend = Ap[j+1];
        Int p;
        for (p = pbeg; p < pend; p++) {
            Int i = Ai[p];
            if (jmatch[i] < -1) // row to be ignored
                continue;
            assert(jmatch[i] >= 0);
            if (marked[jmatch[i]] == jstart ||
                marked[jmatch[i]] == kDeadColumn)
                continue;
            pstack[head] = p+1;
            istack[head] = i;
            jstack[++head] = jmatch[i]; // continue augmenting path here
            break;
        }
        if (p == Ap[j+1])
            head--;
    }
    if (found) {
        for (Int p = head; p >= 0; p--)
            jmatch[istack[p]] = jstack[p];
    } else {
        for (Int k = 0; k < num_visited; k++)
            marked[visited[k]] = kDeadColumn;
    }
    return found;
}

}  // namespace ipx
