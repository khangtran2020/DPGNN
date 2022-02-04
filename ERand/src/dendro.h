// ****************************************************************************************************
// *** COPYRIGHT NOTICE *******************************************************************************
// dendro.h - hierarchical random graph (hrg) data structure
// Copyright (C) 2005-2009 Aaron Clauset
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// See http://www.gnu.org/licenses/gpl.txt for more details.
//
// ****************************************************************************************************
// Author       : Aaron Clauset  ( aaronc@santafe.edu | http://www.santafe.edu/~aaronc/ )
// Collaborators: Cristopher Moore and Mark E.J. Newman
// Project      : Hierarchical Random Graphs
// Location     : University of New Mexico, Dept. of Computer Science AND Santa Fe Institute
// Created      : 26 October 2005 - 7 December 2005
// Modified     : 23 December 2007 (cleaned up for public consumption)
//
// ****************************************************************************************************
//
// Maximum likelihood dendrogram data structure. This is the heart of the HRG algorithm: all
// manipulations are done here and all data is stored here. The data structure uses the separate
// graph data structure to store the basic adjacency information (in a dangerously mutable way).
//
// ****************************************************************************************************
// ****************************************************************************************************
// Author       : XIAO Qian  ( xiaoqiannus@gmail.com )
// Collaborators: CHEN Rui and TAN Kian-Lee
// Project      : Differentially Private Network Structural Inference
// Location     : National University of Singapore
// Modified     : 14 Feb 2014
// 		        : 9 Oct 2014    (cleaned up for public consumption)
//
// ****************************************************************************************************
// ****************************************************************************************************
// ****************************************************************************************************
// Author       : Khang Tran ( kt36@njit.edu )
// Collaborators: Hai Phan
// Project      : Privacy Preserving for Graph Neural Network
// Location     : New Jersey Institute of Technology
// Modified     : 04 Feb 2022
// ****************************************************************************************************


#if !defined(dendro_INCLUDED)
#define dendro_INCLUDED


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>

#include "MersenneTwister.h"
#include "graph.h"
#include "rbtree.h"
#include "laplace.h"
#include <cmath>

using namespace std;

// ********************************************************************************************************
// ******** Basic Structures ******************************************************************************



enum {
    DENDRO, GRAPH, LEFT, RIGHT
};
struct block {
    double x;
    int y;
};
struct ipair {
    int x;
    int y;
    short int t;
    string sp;
};

// ********************************************************************************************************
// ******** Internal Edge Class ***************************************************************************
// The usefulness of this data structure is to provide an easy to way maintain the set of internal edges,
// in the dendrogram D. It allows for the selection of a random internal edge in O(1) time, and it takes
// O(1) time to update its structure given an internal move.

class interns {
private:
    ipair *edgelist;                    // list of internal edges
    int **indexLUT;                    // table of indices of internal edges in edgelist
    int q;                        // number of internal edges
    int count;                    // (for adding edges) edgelist index of new edge to add
    MTRand mtr;                        // Mersenne Twister random number generator instance

public:
    interns(const int);

    ~interns();

    bool addEdge(const int, const int, const short int);    // add an internal edge, O(1)
    ipair *getEdge(const int);                            // returns the ith edge of edgelist, O(1)
    ipair *getRandomEdge();                            // returns a uniformly random internal edge, O(1)
    void printEdgeList();                            // writes edgelist to terminal
    bool swapEdges(const int, const int, const short int, const int, const int, const short int);
    // swaps two edges, O(1)
};

// ********************************************************************************************************

interns::interns(const int n) {
    q = n;
    count = 0;
    edgelist = new ipair[q];
    indexLUT = new int *[q + 1];
    for (int i = 0; i < (q + 1); i++) {
        indexLUT[i] = new int[2];
        indexLUT[i][0] = indexLUT[i][1] = -1;
    }
}

interns::~interns() {
    delete[] edgelist;
    for (int i = 0; i < (q + 1); i++) { delete[] indexLUT[i]; }
    delete[] indexLUT;
}

// ********************************************************************************************************

// NOTE: Returns an address to another object -- do not deallocate
ipair *interns::getEdge(const int i) { return &edgelist[i]; }

// ********************************************************************************************************

// NOTE: Returns an address to another object -- do not deallocate
ipair *interns::getRandomEdge() { return &edgelist[(int) (floor((double) (q) * mtr.randExc()))]; }

// ********************************************************************************************************

bool interns::addEdge(const int new_x, const int new_y, const short int new_type) {
    // This function adds a new edge (i,j,t,sp) to the list of internal edges. After checking that the inputs
    // fall in the appropriate range of values, it records the new edgelist index in the indexLUT and then
    // puts the input values into that edgelist location.
    if (count < q and new_x >= 0 and new_x < (q + 1) and new_y >= 0 and new_y < (q + 2) and
        (new_type == LEFT or new_type == RIGHT)) {
        if (new_type == LEFT) { indexLUT[new_x][0] = count; } else { indexLUT[new_x][1] = count; }
        edgelist[count].x = new_x;
        edgelist[count].y = new_y;
        edgelist[count].t = new_type;
        count++;
        return true;
    } else { return false; }
}

// ********************************************************************************************************

void interns::printEdgeList() {
    for (int i = 0; i < q; i++) {
        cout << "(" << edgelist[i].x << " " << edgelist[i].y << " ";
        if (edgelist[i].t == LEFT) { cout << "L) "; }
        else if (edgelist[i].t == RIGHT) { cout << "R) "; }
        else {
            cout << "?) ";
        }
    }
    cout << endl;
    return;
}

// ********************************************************************************************************

bool interns::swapEdges(const int one_x, const int one_y, const short int one_type, const int two_x,
                        const int two_y, const short int two_type) {
    // The moves on the dendrogram always swap edges, either of which (or both, or neither) can by internal
    // edges. So, this function mirrors that operation for the internal edgelist and indexLUT.

    int index, jndex, temp;
    bool one_isInternal = false;
    bool two_isInternal = false;

    if (one_x >= 0 and one_x < (q + 1) and two_x >= 0 and two_x < (q + 1) and
        (two_type == LEFT or two_type == RIGHT) and
        one_y >= 0 and one_y < (q + 2) and two_y >= 0 and two_y < (q + 2) and (one_type == LEFT or one_type == RIGHT)) {

        if (one_type == LEFT) { temp = 0; } else { temp = 1; }
        if (indexLUT[one_x][temp] > -1) { one_isInternal = true; }
        if (two_type == LEFT) { temp = 0; } else { temp = 1; }
        if (indexLUT[two_x][temp] > -1) { two_isInternal = true; }

        if (one_isInternal and two_isInternal) {
            if (one_type == LEFT) { index = indexLUT[one_x][0]; } else { index = indexLUT[one_x][1]; }
            if (two_type == LEFT) { jndex = indexLUT[two_x][0]; } else { jndex = indexLUT[two_x][1]; }
            temp = edgelist[index].y;
            edgelist[index].y = edgelist[jndex].y;
            edgelist[jndex].y = temp;

        } else if (one_isInternal) {
            if (one_type == LEFT) {
                index = indexLUT[one_x][0];
                indexLUT[one_x][0] = -1;
            } else {
                index = indexLUT[one_x][1];
                indexLUT[one_x][1] = -1;
            }
            edgelist[index].x = two_x;
            edgelist[index].t = two_type;
            if (two_type == LEFT) { indexLUT[two_x][0] = index; } else { indexLUT[two_x][1] = index; } // add new

        } else if (two_isInternal) {
            if (two_type == LEFT) {
                index = indexLUT[two_x][0];
                indexLUT[two_x][0] = -1;
            } else {
                index = indexLUT[two_x][1];
                indexLUT[two_x][1] = -1;
            }
            edgelist[index].x = one_x;
            edgelist[index].t = one_type;
            if (one_type == LEFT) { indexLUT[one_x][0] = index; } else { indexLUT[one_x][1] = index; } // add new
        } else {} // else neither is internal

        return true;
    } else { return false; }
}

// ********************************************************************************************************
// ******** Tree elementd Class ***************************************************************************

class elementd {
public:
    short int type;                // either DENDRO or GRAPH
    double logL_pub;                // log-likelihood contribution of this internal node
    double logL_priv;
    double p;                    // probability p_i that an edge exists between L and R subtrees
    double p_priv;
    long long e;                    // number of edges between L and R subtrees
    long long e_priv;
    int n;                    // number of leafs in subtree rooted here
    int n_priv;
    int label;                // subtree label: smallest leaf index
    int index;                // index in containing array
    double noisy_p;
    double noise_scale;

    elementd *M;                    // pointer to parent node
    elementd *L;                    // pointer for L subtree
    elementd *R;                    // pointer for R subtree

    elementd();

    ~elementd();
};

elementd::elementd() {
    type = DENDRO;
    logL_pub = logL_priv = p = p_priv = 0.0;
    noisy_p = noise_scale = 0.0;
    e = 0;
    e_priv = 0;
    n = 0;
    n_priv = 0;
    label = index = -1;
    M = L = R = NULL;
}

elementd::~elementd() {}

// ********************************************************************************************************
// ******** Dendrogram Class ******************************************************************************

class dendro {

private:
    elementd *root;            // root of the dendrogram
    elementd *internal;            // array of n-1 internal vertices (the dendrogram D)
    elementd *leaf;            // array of n   leaf vertices (the graph G)
    int n;                // number of leaf vertices to allocate
    interns *d;                // list of internal edges of dendrogram D
    list **paths;            // array of path-lists from root to leaf
    double L;                // log-likelihood of graph G given dendrogram D
    MTRand mtr;                // Mersenne Twister random number generator instance
    rbtree subtreeL, subtreeR;    // trees for computeEdgeCount() function

    void binarySearchInsert(elementd *,
                            elementd *);                            // insert node i according to binary search property
    list *binarySearchFind(const double);                                    // return path to root from leaf
    long long computeEdgeCount(const int, const short int, const int, const short int,
                               const bool);  // compute number of edges between two internal subtrees
    elementd *findCommonAncestor(list **, const int,
                                 const int);                        // find internal node of D that is common ancestor of i,j
    void printSubTree(elementd *);                                            // display the subtree rooted at z
    list *
    reversePathToRoot(const int);                                        // return reverse of path to leaf from root
    void QsortMain(block *, int, int);                                        // ` functions
    int QsortPartition(block *, int, int, int);

    long long countEdgeBelow(elementd *,
                             bool);                                            //count edges in graph in the subtree
    void updateNoisyPr(elementd *, double, Laplace *,
                       bool);                    //update connection probabilities with laplacian noise in the subtree
    void setNoisyPrBelow(elementd *, double);


public:
    graph *g;                                    // underlying G (dangerously accessible)

    dendro();

    ~dendro();                                // constructor / destructor
    void buildDendrogram();                        // build dendrogram from g
    double getLikelihood();                        // return likelihood of G given D
    bool importDendrogramStructure(const string);    // read dendrogram structure from file
    void makeRandomGraph(graph *, double);                        // make random G from D
    bool monteCarloMove(double &, bool &, double, bool);            // make single MCMC move
    void recordDendrogramStructure(const string);    // record D structure to file
    void recordGraphStructure(const string, graph *);        // record G structure to file
    void refreshLikelihood();                    // force refresh of log-likelihood value
    void printDendrogram();                        // write dendrogram structure to terminal
    void makeNoisyRandomGraph(graph *, double, bool);                //genenerate noisy random graph
    int getNumofNodes();                        //return number of nodes


};

// ******** Dendrogram Methods ****************************************************************************

dendro::dendro() {
    root = NULL;
    internal = NULL;
    leaf = NULL;
    d = NULL;
    paths = NULL;
    g = NULL;
}

dendro::~dendro() {
    list *curr, *prev;
    if (g != NULL) {
        delete g;
        g = NULL;
    }    // O(m)
    if (internal != NULL) {
        delete[] internal;
        internal = NULL;
    }    // O(n)
    if (leaf != NULL) {
        delete[] leaf;
        leaf = NULL;
    }    // O(n)
    if (d != NULL) {
        delete d;
        d = NULL;
    }    // O(n)
    if (paths != NULL) {
        for (int i = 0; i < n; i++) {
            curr = paths[i];
            while (curr != NULL) {
                prev = curr;
                curr = curr->next;
                delete prev;
                prev = NULL;
            }
            paths[i] = NULL;
        }
        delete[] paths;
    }
    paths = NULL;
}

// ********************************************************************************************************
int dendro::getNumofNodes() {
    return n;
}

void dendro::binarySearchInsert(elementd *x, elementd *y) {
    if (y->p < x->p) {        // go to left subtree
        if (x->L == NULL) { // check if left subtree is empty
            x->L = y;        // make x left child
            y->M = x;        // make y parent of child
            return;
        } else { binarySearchInsert(x->L, y); }
    } else {                // go to right subtree
        if (x->R == NULL) { // check if right subtree is empty
            x->R = y;        // make x right child
            y->M = x;        // make y parent of child
            return;
        } else { binarySearchInsert(x->R, y); }
    }
    return;
}

// ********************************************************************************************************

list *dendro::binarySearchFind(const double v) {
    list *head = nullptr, *tail = nullptr, *newlist;
    elementd *current = root;
    bool flag_stopSearch = false;

    while (!flag_stopSearch) {                // continue until we're finished
        newlist = new list;                // add this node to the path
        newlist->x = current->label;
        if (current == root) {
            head = newlist;
            tail = head;
        } else {
            tail->next = newlist;
            tail = newlist;
        }
        if (v < current->p) {                // now try left subtree
            if (current->L->type == GRAPH) { flag_stopSearch = true; }
            else { current = current->L; }
        } else {                            // else try right subtree
            if (current->R->type == GRAPH) { flag_stopSearch = true; }
            else { current = current->R; }
        }
    }
    return head;
}

// ********************************************************************************************************

void dendro::buildDendrogram() {
    if (g == NULL) {
        cout << "Error: cannot build dendrogram without a graph structure.\n";
        return;
    }

/* the initialization of the dendrogram structure goes like this:
 * 1) we allocate space for the n-1 internal nodes of the dendrogram, and then the n leaf nodes
 * 2) we build a random binary tree structure out of the internal nodes by assigning each
 *    a uniformly random value over [0,1] and then inserting it into the tree according to the
 *    binary-search rule.
 * 3) next, we make a random permutation of the n leaf nodes and add them to the dendrogram D by
 *    replacing the emptpy spots in-order
 * 4) then, we compute the path from the root to each leaf and store that in each leaf (this is
 *    prep work for the next step)
 * 5) finally, we compute the values for nL, nR, e (and thus p) and the label for each internal
 *    node by allocating each of the m edges in g to the appropriate internal node
 */

    // --- Initialization and memory allocation for data structures
    // After allocating the memory for D and G, we need to mark the nodes for G as being
    // non-internal vertices, and then insert them into a random binary tree structure.
    // For simplicity, we make the first internal node in the array the root.

    bool flag_debug = true;
    n = g->numNodes();        // size of graph
    leaf = new elementd[n];        // allocate memory for G, O(n)
    internal = new elementd[n - 1];    // allocate memory for D, O(n)
    d = new interns(n - 2);      // allocate memory for internal edges of D, O(n)
    for (int i = 0; i < n; i++) {        // initialize leaf nodes
        leaf[i].type = GRAPH;
        leaf[i].label = i;
        leaf[i].index = i;
        if (g->getType(i) == 0) {
            leaf[i].n_priv = 0;
            leaf[i].n = 1;
        } else if (g->getType(i) == 1) {
            leaf[i].n_priv = 1;
            leaf[i].n = 0;
        } else if (g->getType(i) == 2) {
            leaf[i].n_priv = 1;
            leaf[i].n = 1;
        }
    }
//    for (int i = 0; i < n; i++) cout << i << " " << leaf[i].label << " " << leaf[i].n_priv << endl;
    if (flag_debug) { cout << ">> dendro: allocated memory for internal and leaf arrays" << endl; }
    root = &internal[0];        // initialize internal nodes
    root->label = 0;
    root->index = 0;
    root->p = mtr.randExc();
    for (int i = 1; i < (n - 1); i++) {        // insert remaining internal vertices, O(n log n)
        internal[i].label = i;
        internal[i].index = i;
        internal[i].p = mtr.randExc();
        internal[i].p_priv = internal[i].p;
        binarySearchInsert(root, &internal[i]);
    }
    if (flag_debug) { cout << ">> dendro: inserted internal vertices into random binary tree" << endl; }

    // --- Hang leaf nodes off end of dendrogram O(n log n)
    // To impose this random hierarchical relationship on G, we first take a random permutation
    // of the leaf vertices and then replace the NULLs at the bottom of the tree in-order with
    // the leafs. As a hack to ensure that we can find the leafs later using a binary search,
    // we assign each of them the p value of their parent, perturbed slightly so as to preserve
    // the binary search property.

    block *array;
    array = new block[n];
    for (int i = 0; i < n; i++) {
        array[i].x = mtr.randExc();
        array[i].y = i;
    }
    QsortMain(array, 0, n - 1);

    int k = 0;                        // replace NULLs with leaf nodes, and
    for (int i = 0; i < (n - 1); i++) {        //    maintain binary search property, O(n)
        if (internal[i].L == NULL) {
            internal[i].L = &leaf[array[k].y];
            leaf[array[k].y].M = &internal[i];
            leaf[array[k].y].p = internal[i].p - 0.0000000000001;
            leaf[array[k++].y].p_priv = internal[i].p_priv - 0.0000000000001;
        }
        if (internal[i].R == NULL) {
            internal[i].R = &leaf[array[k].y];
            leaf[array[k].y].M = &internal[i];
            leaf[array[k].y].p = internal[i].p + 0.0000000000001;
            leaf[array[k++].y].p_priv = internal[i].p_priv + 0.0000000000001;
        }
    }
    delete[] array;
    if (flag_debug) { cout << ">> dendro: replaced NULLs in bin-tree with leaf nodes" << endl; }

    // --- Compute the path from root -> leaf for each leaf O(n log n)
    // Using the binary search property, we can find each leaf node in O(log n) time. The
    // binarySearchFind() function returns the list of internal node indices that the search
    // crossed, in the order of root -> ... -> leaf, for use in the subsequent few operations.

    if (paths != NULL) {
        list *curr, *prev;
        for (int i = 0; i < n; i++) {
            curr = paths[i];
            while (curr != NULL) {
                prev = curr;
                curr = curr->next;
                delete prev;
                prev = NULL;
            }
            paths[i] = NULL;
        }
        delete[] paths;
    }
    paths = NULL;
    paths = new list *[n];
    for (int i = 0; i < n; i++) { paths[i] = binarySearchFind(leaf[i].p); }

    if (flag_debug) { cout << ">> dendro: computed paths from root to leafs" << endl; }

    // --- Count e for each internal node O(m)
    // To count the number of edges that span the L and R subtrees for each internal node,
    // we use the path information we just computed. Then, we loop over all edges in G
    // and find the common ancestor in D of the two endpoints and increment that internal
    // node's e count. This process takes O(m) time because in a roughly balanced binary
    // tree (given by our random dendrogram), the vast majority of vertices take basically
    // constant time to find their common ancestor. Note that because our adjacency list
    // is symmetric, we overcount each e by a factor of 2, so we need to correct this after.

    elementd *ancestor;
    edge *curr;
    int degree;
    for (int i = 0; i < (n - 1); i++) {
        internal[i].e = 0;
        internal[i].label = -1;
    }
    for (int i = 0; i < n; i++) {
        curr = g->getNeighborList(i, false);
        while (curr != NULL) {
            ancestor = findCommonAncestor(paths, i, curr->x);
            ancestor->e += 1;
            curr = curr->next;
        }
    }
    for (int i = 0; i < n; i++) {
        curr = g->getNeighborList(i, true);
        while (curr != NULL) {
            ancestor = findCommonAncestor(paths, i, curr->x);
            ancestor->e_priv += 1;
            curr = curr->next;
        }
    }
    for (int i = 0; i < (n - 1); i++) {
        internal[i].e /= 2;
        internal[i].e_priv /= 2;
    }
    if (flag_debug) { cout << ">> dendro: finished common ancestor computation" << endl; }

    // --- Count n for each internal node O(n log n)
    // To tabulate the number of leafs in each subtree rooted at an internal node,
    // we use the path information computed above.

    for (int i = 0; i < n; i++) {
        ancestor = &leaf[i];
        int type_priv = leaf[i].n_priv;
        int type_pub = leaf[i].n;
        ancestor = ancestor->M;
        while (ancestor != NULL) {
//            cout << ancestor->label << " " << ancestor->n << endl;
            if (type_priv > 0) ancestor->n_priv++;
            if (type_pub > 0) ancestor->n++;
            ancestor = ancestor->M;
        }
    }
    if (flag_debug) { cout << ">> dendro: computed subtree sizes" << endl; }

    // --- Label all internal vertices O(n log n)
    // We want to label each internal vertex with the smallest leaf index of its children.
    // This will allow us to collapse many leaf-orderings into a single dendrogram structure
    // that is independent of child-exhanges (since these have no impact on the likelihood
    // of the hierarchical structure). To do this, we loop over the leaf vertices from
    // smallest to largest and walk along that leaf's path from the root. If we find an
    // unlabeled internal node, then we mark it with this leaf's index.


    for (int i = 0; i < n; i++) {
        ancestor = &leaf[i];
        while (ancestor != NULL) {
            if (ancestor->label == -1 || ancestor->label > leaf[i].label) { ancestor->label = leaf[i].label; }
            ancestor = ancestor->M;
        }
    }

    if (flag_debug) { cout << ">> dendro: labeled all internal vertices" << endl; }

    // --- Exchange children to enforce order-property O(n)
    // We state that the order-property requires that an internal node's label is the
    // smallest index of its left subtree. The dendrogram so far doesn't reflect this, so we
    // need to step through each internal vertex and make that adjustment (swapping nL and nR
    // if we make a change).

    int temp;
    elementd *tempe;
    for (int i = 0; i < (n - 1); i++) {
        if (internal[i].L->label > internal[i].label) {
            tempe = internal[i].L;
            internal[i].L = internal[i].R;
            internal[i].R = tempe;
        }
    }
    if (flag_debug) { cout << ">> dendro: enforced order-property" << endl; }

    // --- Tabulate internal dendrogram edges O(n^2)
    // For the MCMC moves later on, we'll need to be able to choose, uniformly at random, an
    // internal edge of the dendrogram to manipulate. There are always n-2 of them, and we can
    // find them simply by scanning across the internal vertices and observing which have children
    // that are also internal vertices. Note: very important that the order property be enforced
    // before this step is taken; otherwise, the internal edges wont reflect the actual dendrogram
    // structure.

    for (int i = 0; i < (n - 1); i++) {
        if (internal[i].L->type == DENDRO) { d->addEdge(i, internal[i].L->index, LEFT); }
        if (internal[i].R->type == DENDRO) { d->addEdge(i, internal[i].R->index, RIGHT); }
    }

    if (flag_debug) { cout << ">> dendro: tabulated internal dendrogram edges" << endl; }

    // --- Clear memory for paths O(n log n)
    // Now that we're finished using the paths, we need to deallocate them manually.

    list *current, *previous;
    for (int i = 0; i < n; i++) {
        current = paths[i];
        while (current != NULL) {
            previous = current;
            current = current->next;
            delete previous;
            previous = NULL;
        }
        paths[i] = NULL;
    }
    delete[] paths;
    paths = NULL;
    if (flag_debug) { cout << ">> dendro: cleared memory for paths" << endl; }

    // --- Compute p_i for each internal node O(n)
    // Each internal node's p_i = e_i / (nL_i*nR_i), and now that we have each of those
    // pieces, we may calculate this value for each internal node. Given these, we can then
    // calculate the log-likelihood of the entire dendrogram structure
    // \log(L) = \sum_{i=1}^{n} ( ( e_i \log[p_i] ) + ( (nL_i*nR_i - e_i) \log[1-p_i] ) )

    L = 0.0;
    double dL_pub, dL_priv;
    double nL_nR, ei;
    double nL_nR_priv, ei_priv;
    for (int i = 0; i < (n - 1); i++) {
        nL_nR = ((double) internal[i].L->n) * ((double) internal[i].R->n);
        ei = (double) internal[i].e;
        nL_nR_priv = ((double) internal[i].L->n_priv) * ((double) internal[i].R->n_priv);
        ei_priv = (double) internal[i].e_priv;
        if (ei != 0 && nL_nR == 0 || ei_priv != 0 && nL_nR_priv == 0)
            cout << nL_nR << " " << ei << " " << nL_nR_priv << " " << ei_priv << endl;
        if (nL_nR != 0) internal[i].p = (ei) / (nL_nR);
        else internal[i].p = 0;
        if (nL_nR_priv != 0)internal[i].p_priv = (ei_priv) / (nL_nR_priv);
        else internal[i].p_priv = 0;

        if (ei == 0 or ei == nL_nR or internal[i].p == 0) { dL_pub = 0.0; }
        else { dL_pub = ei * log(internal[i].p) + (nL_nR - ei) * log(1.0 - internal[i].p); }
        if (ei_priv == 0 or ei_priv == nL_nR_priv or internal[i].p_priv == 0) { dL_priv = 0.0; }
        else { dL_pub = ei_priv * log(internal[i].p_priv) + (nL_nR_priv - ei_priv) * log(1.0 - internal[i].p_priv); }
        internal[i].logL_pub = dL_pub;
        internal[i].logL_priv = dL_priv;
        L += dL_pub + dL_priv;
    }
//    printDendrogram();
    if (flag_debug) {
        cout << ">> dendro: computed internal node probability value" << endl;
//		if (n<100) { printDendrogram(); }
        cout << "Log-Likelihood = " << L << endl;
    }
    char pauseme;
    for (int i = 0; i < (n - 1); i++) {
        if (internal[i].label > internal[i].L->label) {
            tempe = internal[i].L;
            internal[i].L = internal[i].R;
            internal[i].R = tempe;
            cout << "#### WARNING - order property violated by internal[" << i << "] (fixed)" << endl;
            cin >> pauseme;
        }
    }

    // --- Dendrogram is now built
    if (flag_debug) { cout << ">> dendro: build dendrogram complete" << endl; }

    return;
}

// ********************************************************************************************************

long long
dendro::computeEdgeCount(const int a, const short int atype, const int b, const short int btype, const bool edge_type) {
    // This function computes the number of edges that cross between the subtree internal[a]
    // and the subtree internal[b].
    // To do this, we use an array A[1..n] integers which take values -1 if A[i] is in the
    // subtree defined by internal[a], +1 if A[i] is in the subtree internal[b], and 0
    // otherwise. Taking the smaller of the two sets, we then scan over the edges attached
    // to that set of vertices and count the number of endpoints we see in the other set.

    bool flag_debug = true;
    bool flag_go = true;
    int nA, nB;
    long long count = 0;
    const short int k = 1 + DENDRO + GRAPH;

    elementd *curr;

    // --- First, we push the leaf nodes in the L and R subtrees into balanced binary tree
    //     structures so that we can search them quickly later on.
    if (atype == GRAPH) {                    // default case, subtree A is size 1
        subtreeL.insertItem(a, -1);            // insert single node as member of left subtree
        nA = 1;                    //
    } else {
        curr = &internal[a];            // explore subtree A, O(|A|)
        curr->type = k + 1;                    //
        nA = 0;                    //
        while (flag_go) {

            if (curr->index == internal[a].M->index) {
                internal[a].type = DENDRO;
                flag_go = false;
            } else {
                if (curr->type == k + 1 and
                    curr->L->type == GRAPH) {        // - is it time, and is left child a graph node?
                    subtreeL.insertItem(curr->L->index, -1);
                    curr->type = k + 2;  //
                    nA++;                    //
                }
                if (curr->type == k + 2 and
                    curr->R->type == GRAPH) {        // - is it time, and is right child a graph node?
                    subtreeL.insertItem(curr->R->index, -1);
                    curr->type = k + 3;  //
                    nA++;                    //
                }
                if (curr->type == k + 1) {            // - go left
                    curr->type = k + 2;            //
                    curr = curr->L;        //
                    curr->type = k + 1;
                } else if (curr->type == k + 2) {        // - else go right
                    curr->type = k + 3;            //
                    curr = curr->R;        //
                    curr->type = k + 1;
                } else {                        // - else go up a level
                    curr->type = DENDRO;        //
                    curr = curr->M;        //
                    if (curr == NULL) {
                        flag_go = false;
//						cout << "A exit: reached null parent" << endl;
                    }
                }
            }
            if (nA > n) {
                cout << "error! nA > n\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n" << endl;
                break;
            }
        }
    }

    if (btype == GRAPH) {                    // default case, subtree A is size 1
        subtreeR.insertItem(b, 1);            // insert node as single member of right subtree
        nB = 1;                    //
    } else {
        flag_go = true;
        curr = &internal[b];            // explore subtree B, O(|B|)
        curr->type = k + 1;                    //
        nB = 0;                    //
        while (flag_go) {

            if (curr->index == internal[b].M->index) {
                internal[b].type = DENDRO;
                flag_go = false;
            } else {
                if (curr->type == k + 1 and
                    curr->L->type == GRAPH) {        // - is it time, and is left child a graph node?
                    subtreeR.insertItem(curr->L->index, 1);
                    curr->type = k + 2;  //
                    nB++;                    //
                }
                if (curr->type == k + 2 and
                    curr->R->type == GRAPH) {        // - is it time, and is right child a graph node?
                    subtreeR.insertItem(curr->R->index, 1);
                    curr->type = k + 3;  //
                    nB++;                    //
                }
                if (curr->type == k + 1) {            // - look left
                    curr->type = k + 2;            //
                    curr = curr->L;        //
                    curr->type = k + 1;
                } else if (curr->type == k + 2) {        // - look right
                    curr->type = k + 3;            //
                    curr = curr->R;        //
                    curr->type = k + 1;
                } else {                        // - else go up a level
                    curr->type = DENDRO;        //
                    curr = curr->M;        //
                    if (curr == NULL) {
                        flag_go = false;
                    }
                }
            }
            if (nB > n) {
                cout << "error! nB > n \n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n#\n" << endl;
                break;
            }
        }
    }

    // --- Now, we take the smaller subtree and ask how many of its emerging edges have their
    //     partner in the other subtree. O(|A| log |A|) time
    edge *current;
    int *treeList;
    if (nA < nB) {
        treeList = subtreeL.returnArrayOfKeys();    // subtreeL is smaller
        for (int i = 0; i < nA; i++) {
            current = g->getNeighborList(treeList[i], edge_type);
            while (current != NULL) {            // loop over each of its neighbors v_j
                if (subtreeR.findItem(current->x) != NULL) { count++; }
                current = current->next;            // to see if v_j is in A
            }                                //
            subtreeL.deleteItem(treeList[i]);
        }
        delete[] treeList;
        treeList = subtreeR.returnArrayOfKeys();
        for (int i = 0; i < nB; i++) { subtreeR.deleteItem(treeList[i]); }
        delete[] treeList;
    } else {
        treeList = subtreeR.returnArrayOfKeys();    // subtreeR is smaller
        for (int i = 0; i < nB; i++) {
            current = g->getNeighborList(treeList[i], edge_type);
            while (current != NULL) {            // loop over each of its neighbors v_j
                if (subtreeL.findItem(current->x) != NULL) { count++; }
                current = current->next;            // to see if v_j is in B
            }                                //
            subtreeR.deleteItem(treeList[i]);
        }
        delete[] treeList;
        treeList = subtreeL.returnArrayOfKeys();
        for (int i = 0; i < nA; i++) { subtreeL.deleteItem(treeList[i]); }
        delete[] treeList;
    }

    return count;
}

// ********************************************************************************************************

elementd *dendro::findCommonAncestor(list **paths, const int i, const int j) {
    list *headOne = paths[i];
    list *headTwo = paths[j];
    elementd *lastStep;
    while (headOne->x == headTwo->x) {
        lastStep = &internal[headOne->x];
        headOne = headOne->next;
        headTwo = headTwo->next;
        if (headOne == NULL or headTwo == NULL) { break; }
    }
    return lastStep;            // Returns address of an internal node; do not deallocate
}

// ********************************************************************************************************

double dendro::getLikelihood() { return L; }

// ********************************************************************************************************

bool dendro::importDendrogramStructure(const string in_file) {
    string bracketL, bracketR, sL, sR, sLtype, sRtype, sp, se, sn;
    int sindex, sLindex, sRindex, snume, snumn;
    double sprob;
    bool safeExit = true;
    bool flag_debug = true;
    n = 1;

    ifstream fscan(in_file.c_str(), ios::in);
    while (fscan >> sn) { if (sn == "[") { n++; }}
    fscan.close();

    leaf = new elementd[n];        // allocate memory for G, O(n)
    internal = new elementd[n - 1];    // allocate memory for D, O(n)
    d = new interns(n - 2);        // allocate memory for internal edges of D, O(n)
    for (int i = 0; i < n; i++) {        // initialize leaf nodes
        leaf[i].type = GRAPH;
        leaf[i].label = i;
        leaf[i].index = i;
        leaf[i].n = 1;
    }
    root = &internal[0];        // initialize internal nodes
    root->label = 0;
    for (int i = 1; i < (n - 1); i++) {
        internal[i].index = i;
        internal[i].label = -1;
    }
    if (flag_debug) { cout << ">> dendro: allocated memory for internal and leaf arrays" << endl; }

    // --- Import basic structure from file O(n)
    ifstream fin(in_file.c_str(), ios::in);
    while (fin >> bracketL >> sindex >> bracketR >> sL >> sLindex >> sLtype >> sR >> sRindex >> sRtype >> sp >> sprob
               >> se >> snume >> sn >> snumn) {
        cout << bracketL << " " << sindex << " " << bracketR << " " << sL << " " << sLindex << " " << sLtype << " "
             << sR << " " << sRindex << " " << sRtype << " " << sp << " " << sprob << " " << se << " " << snume << " "
             << sn << " " << snumn << endl;
        if (sLtype == "(D)") {
            internal[sindex].L = &internal[sLindex];
            internal[sLindex].M = &internal[sindex];
        } else if (sLtype == "(G)") {
            internal[sindex].L = &leaf[sLindex];
            leaf[sLindex].M = &internal[sindex];
        } else {
            cout << "Error: " << bracketL << sindex << bracketR << sL << sLindex << sLtype << sR << sRindex << sRtype
                 << sp << sprob << se << snume << sn << snumn << endl;
            safeExit = false;
            break;
        }
        if (sRtype == "(D)") {
            internal[sindex].R = &internal[sRindex];
            internal[sRindex].M = &internal[sindex];
        } else if (sRtype == "(G)") {
            internal[sindex].R = &leaf[sRindex];
            leaf[sRindex].M = &internal[sindex];
        } else {
            cout << "Error: " << bracketL << sindex << bracketR << sL << sLindex << sLtype << sR << sRindex << sRtype
                 << sp << sprob << se << snume << sn << snumn << endl;
            safeExit = false;
            break;
        }
        internal[sindex].p = sprob;
        if (sprob < 0.0 || sprob > 1.0) {
            cout << "Error: " << bracketL << sindex << bracketR << sL << sLindex << sLtype << sR << sRindex << sRtype
                 << sp << sprob << se << snume << sn << snumn << endl;
            safeExit = false;
            break;
        }
        internal[sindex].e = snume;
        internal[sindex].n = snumn;
        internal[sindex].index = sindex;
    }
    fin.close();
    if (!safeExit) { return false; }
    if (flag_debug) { cout << ">> dendro: imported basic structure" << endl; }

    // --- Label all internal vertices O(n log n)
    elementd *curr;
    for (int i = 0; i < n; i++) {
        curr = &leaf[i];
        while (curr != NULL) {
            if (curr->label == -1 || curr->label > leaf[i].label) { curr->label = leaf[i].label; }
            curr = curr->M;
        }
    }
    if (flag_debug) { cout << ">> dendro: labeled all internal vertices" << endl; }

    // --- Exchange children to enforce order-property O(n)
    int temp;
    elementd *tempe;
    for (int i = 0; i < (n - 1); i++) {
        if (internal[i].L->label > internal[i].label) {
            tempe = internal[i].L;
            internal[i].L = internal[i].R;
            internal[i].R = tempe;
        }
    }
    if (flag_debug) { cout << ">> dendro: enforced order-property" << endl; }

    // --- Tabulate internal dendrogram edges O(n)
    int k = 0;
    for (int i = 0; i < (n - 1); i++) {
        if (internal[i].L->type == DENDRO) { d->addEdge(i, internal[i].L->index, LEFT); }
        if (internal[i].R->type == DENDRO) { d->addEdge(i, internal[i].R->index, RIGHT); }
    }
    if (flag_debug) { cout << ">> dendro: tabulated internal dendrogram edges" << endl; }

    // --- Compute p_i for each internal node O(n)
    // Each internal node's p_i = e_i / (nL_i*nR_i), and now that we have each of those
    // pieces, we may calculate this value for each internal node. Given these, we can then
    // calculate the log-likelihood of the entire dendrogram structure
    // \log(L) = \sum_{i=1}^{n} ( ( e_i \log[p_i] ) + ( (nL_i*nR_i - e_i) \log[1-p_i] ) )
    L = 0.0;
    double dL_pub, dL_priv;
    double nL_nR, ei;
    double nL_nR_priv, ei_priv;
    for (int i = 0; i < (n - 1); i++) {
        nL_nR = ((double) internal[i].L->n) * ((double) internal[i].R->n);
        ei = (double) internal[i].e;
        nL_nR_priv = ((double) internal[i].L->n_priv) * ((double) internal[i].R->n_priv);
        ei_priv = (double) internal[i].e_priv;
        if (ei != 0 && nL_nR == 0 || ei_priv != 0 && nL_nR_priv == 0)
            cout << nL_nR << " " << ei << " " << nL_nR_priv << " " << ei_priv << endl;
        if (nL_nR != 0) internal[i].p = (ei) / (nL_nR);
        else internal[i].p = 0;
        if (nL_nR_priv != 0)internal[i].p_priv = (ei_priv) / (nL_nR_priv);
        else internal[i].p_priv = 0;

        if (ei == 0 or ei == nL_nR or internal[i].p == 0) { dL_pub = 0.0; }
        else { dL_pub = ei * log(internal[i].p) + (nL_nR - ei) * log(1.0 - internal[i].p); }
        if (ei_priv == 0 or ei_priv == nL_nR_priv or internal[i].p_priv == 0) { dL_priv = 0.0; }
        else { dL_pub = ei_priv * log(internal[i].p_priv) + (nL_nR_priv - ei_priv) * log(1.0 - internal[i].p_priv); }
        internal[i].logL_pub = dL_pub;
        internal[i].logL_priv = dL_priv;
        L += dL_pub + dL_priv;
    }
    if (flag_debug) {
        cout << ">> dendro: computed log-likelihood" << endl;
        cout << "   Log-Likelihood = " << L << endl;
    }

    // --- Dendrogram is now built
    if (flag_debug) { cout << ">> dendro: build dendrogram complete" << endl; }

    return true;
}

long long dendro::countEdgeBelow(elementd *curr, bool edge_type) {
    // This function counts the number of edges in the subtree rooted at node curr.
    if (curr != NULL) {
        if (curr->type == GRAPH) {
            return 0;
        } else if (curr->type == DENDRO) {
            if (edge_type == true)
                return curr->e_priv + countEdgeBelow(curr->L, edge_type) + countEdgeBelow(curr->R, edge_type);
            else return curr->e + countEdgeBelow(curr->L, edge_type) + countEdgeBelow(curr->R, edge_type);
        }
    }
    return -1;
}

void dendro::setNoisyPrBelow(elementd *curr, double p) {
    // This function sets all the connection probabilities in the subtree to be p
    if (curr != NULL) {
        if (curr->type == GRAPH) {
            return;
        } else if (curr->type == DENDRO) {
            curr->noisy_p = p;
            setNoisyPrBelow(curr->L, p);
            setNoisyPrBelow(curr->R, p);
            return;
        }
    }
    return;
}


void dendro::updateNoisyPr(elementd *curr, double epsilon, Laplace *lap, bool flag_control = true) {
    //This function

    bool flag_debug = false;
    if (flag_debug) {
        cout << ">> dendro: begin update noisy p" << endl;
    }
    // --- Top-down procedure to add noise into the connection probabilities of each internal node
    if (curr != NULL) {
        // --- Stop at leaf GRAPH nodes
        if (curr->type == GRAPH) {
            if (flag_debug) { cout << "update graph node" << curr->index << endl; }
            return;
        } else if (curr->type == DENDRO) {

            if (flag_debug) { cout << "update dendro node" << curr->index << endl; }

            double sen = 1.0;                //sensitivity for counting queries are one.
            double e = (double) curr->e_priv;
            double nL_nR = ((double) curr->L->n_priv) * ((double) curr->R->n_priv);
            double nL = (double) curr->L->n_priv;
            double nR = (double) curr->R->n_priv;
            double scale = sen / (nL_nR *
                                  epsilon);            //compute Laplacian noise scale if updating only this internal node
            double total = (nL + nR) * (nL + nR - 1) / 2.0;            //the number of total possible edges

            double community_scale =
                    1.0 / (epsilon * total);    //compute Laplacian noise scale if using Erdős–Rényi random graph model

            if (flag_control) {
                // --- Set threshold scale = 0.05 for safely keeping updating internal node probs and
                //         threshold community_scale = 0.01 for safely using Erdős–Rényi random graph model
                if (scale >= 0.05 and community_scale >= 0.01) {
                    // --- Noise scale is beyond threshold
                    // use Erdős–Rényi random graph model to approximate
                    double num_of_edges_below = (double) countEdgeBelow(curr, true);

                    double noise = lap->get_laplacian_random_variable(sen / epsilon);
                    double np = (num_of_edges_below + noise) / total;
                    setNoisyPrBelow(curr, np);        //set all connection probabilities in the subtree to be np
                    return;

                } else {
                    // --- Noise is under control
                    double noise = lap->get_laplacian_random_variable(sen / epsilon);

                    curr->noisy_p = (e + noise) / nL_nR;
                    // --- Keep going below to update probs of internal nodes
                    updateNoisyPr(curr->L, epsilon, lap, flag_control);
                    updateNoisyPr(curr->R, epsilon, lap, flag_control);
                    return;
                }
            } else {
                //update noisy probabilities without Erdős–Rényi random graph model approximation
                double noise = lap->get_laplacian_random_variable(sen / epsilon);
                curr->noisy_p = (e + noise) / nL_nR;
                updateNoisyPr(curr->L, epsilon, lap, flag_control);
                updateNoisyPr(curr->R, epsilon, lap, flag_control);
                return;

            }


        }
    } else {
        cout << "error!" << curr->index << endl;
    }
}

void dendro::makeNoisyRandomGraph(graph *random_g, double epsilon, bool flag_control = true) {
    //This function generates random graph with noisy probabilities
    bool flag_debug = true;
    bool flag_add_noise;
    elementd *commonAncestor;
//
    if (flag_debug) { cout << ">> dendro: making random graph from dendrogram" << endl; }


    if (epsilon) {
        flag_add_noise = true;
    } else {
        flag_add_noise = false;
    }
    if (flag_add_noise) {
        Laplace *lap = new Laplace;
        updateNoisyPr(root, epsilon, lap,
                      flag_control);        //add noise to the connection probs in the entire dendrogram
        cout << ">> dendro: udpate noisy p complete" << endl;
        if (flag_debug) {                                //if debug, print all noisy probabilities
            double *new_noise = new double[n - 1];
            double *p_array;
            p_array = new double[n - 1];

            for (int i = 0; i < (n - 1); i++) {
                new_noise[i] = internal[i].noisy_p;
                p_array[i] = internal[i].p_priv;

            }

            delete[] new_noise;
            new_noise = NULL;
            delete[] p_array;
            p_array = NULL;
        }

    }


    list *curr, *prev;
    if (paths != NULL) {
        for (int i = 0; i < n; i++) {
            curr = paths[i];
            while (curr != NULL) {
                prev = curr;
                curr = curr->next;
                delete prev;
                prev = NULL;
            }
            paths[i] = NULL;
        }
        delete[] paths;
    }
    paths = NULL;
    paths = new list *[n];                // build paths from root O(n d)
    for (int i = 0; i < n; i++) { paths[i] = reversePathToRoot(i); }

    for (int i = 0; i <
                    n; i++) {            //randomly generate each link between each pair of nodes in graph according to the connection probabilities
        if (g->getType(i) == 0) continue;
        for (int j = (i + 1); j < n; j++) {
            commonAncestor = findCommonAncestor(paths, i, j);
            if (g->getType(j) == 0) continue;
            double prob;
            if (flag_add_noise) {
                prob = commonAncestor->noisy_p;
            } else {
                prob = commonAncestor->p;
            }
            if (mtr.randExc() < prob) {

                if (!(random_g->doesLinkExist(i, j, false))) {
                    if (!(random_g->addLink(i, j, false))) {
                        cout << "Error: (" << j << " " << i << ")" << endl;
                    }
                }
                if (!(random_g->doesLinkExist(j, i, false))) {
                    if (!(random_g->addLink(j, i, false))) {
                        cout << "Error: (" << j << " " << i << ")" << endl;
                    }
                }
            }
        }
    }
//	random_g->printPairs();

    for (int i = 0; i < n; i++) {
        curr = paths[i];
        while (curr != NULL) {
            prev = curr;
            curr = curr->next;
            delete prev;
            prev = NULL;
        }
        paths[i] = NULL;
    }
    delete[] paths;                    // delete paths data structure O(n log n)
    paths = NULL;

    return;
}


// ********************************************************************************************************

void dendro::makeRandomGraph(graph *random_g, double epsilon) {
    bool flag_debug = true;
    if (flag_debug) { cout << ">> dendro: making random graph from dendrogram" << endl; }

    list *curr, *prev;
    if (paths != NULL) {
        for (int i = 0; i < n; i++) {
            curr = paths[i];
            while (curr != NULL) {
                prev = curr;
                curr = curr->next;
                delete prev;
                prev = NULL;
            }
            paths[i] = NULL;
        }
        delete[] paths;
    }
    paths = NULL;
    paths = new list *[n];                // build paths from root O(n d)
    for (int i = 0; i < n; i++) { paths[i] = reversePathToRoot(i); }

    elementd *commonAncestor;
    Laplace lap;
    double sen = 1.0;
    for (int i = 0; i < n; i++) {            // O((h+d)*n^2) - h: height of D; d: average degree in G
        for (int j = (i + 1); j < n; j++) {        // decide neighbors of v_i
            commonAncestor = findCommonAncestor(paths, i, j);
            double prob;
            if (epsilon) {
                double noise = lap.get_laplacian_random_variable(sen / epsilon);
                double e = (double) commonAncestor->e_priv;
                double p = commonAncestor->p_priv;
                double noisy_p = 0;
                if (e == 0 or p == 0) {
                    noisy_p = noise;
                } else {
                    noisy_p = p * (1 + noise / ((double) e));
                }
                prob = noisy_p;

            } else {
                prob = commonAncestor->p_priv;
            }


//            if (mtr.randExc() < commonAncestor->p) {
            if (mtr.randExc() < prob) {

                if (!(random_g->doesLinkExist(i, j, false))) {
                    if (!(random_g->addLink(i, j, false))) {
                        cout << "Error: (" << j << " " << i << ")" << endl;
                    }
                }
                if (!(random_g->doesLinkExist(j, i, false))) {
                    if (!(random_g->addLink(j, i, false))) {
                        cout << "Error: (" << j << " " << i << ")" << endl;
                    }
                }
            }
        }
    }
//	random_g->printPairs();

    for (int i = 0; i < n; i++) {
        curr = paths[i];
        while (curr != NULL) {
            prev = curr;
            curr = curr->next;
            delete prev;
            prev = NULL;
        }
        paths[i] = NULL;
    }
    delete[] paths;                    // delete paths data structure O(n log n)
    paths = NULL;

    return;
}

// ********************************************************************************************************

bool dendro::monteCarloMove(double &delta, bool &ftaken, double T, bool type_move) {
    // A single MC move begins with the selection of a random internal edge (a,b) of the
    // dendrogram. This also determines the three subtrees i, j, k that we will rearrange,
    // and we choose uniformly from among the options.
    //
    // If (a,b) is a left-edge, then we have ((i,j),k), and moves
    // ((i,j),k) -> ((i,k),j)								(alpha move)
    //           -> (i,(j,k)) + enforce order-property for (j,k)	(beta move)
    //
    // If (a,b) is a right-edge, then we have (i,(j,k)), and moves
    // (i,(j,k)) -> ((i,k),j)								(alpha move)
    //           -> ((i,j),k)								(beta move)
    //
    // For each of these moves, we need to know what the change in likelihood will be, so
    // that we can determine with what probability we execute the move.

    elementd *temp, *tempe;
    ipair *tempPair;
    int x, y, token, n_i, n_j, n_k, tempi, n_i_priv, n_j_priv, n_k_priv;
    double n_x, n_y, n_x_priv, n_y_priv;
    double e_x, e_x_priv, e_y, e_y_priv;
    short int t;
    double p_x, p_x_priv, p_y, p_y_priv, L_x, L_x_priv, L_y, L_y_priv, L_new, dLogL, dLogL_pub, dLogL_priv;
    char pauseme;
//    printDendrogram();
    delta = 0.0;
    ftaken = false;
    tempPair = d->getRandomEdge();    // returns address; no need to deallocate
    x = tempPair->x;            // copy contents of referenced random edge
    y = tempPair->y;            //    into local variables
    t = tempPair->t;
//    cout << t << endl;
    if (t == LEFT) {                                //
        if (mtr.randExc() < 0.5) {                    // ## LEFT ALPHA move: ((i,j),k) -> ((i,k),j)
//             I need to calculate the change in the likelihood that would result from
//             this move. Most of the information needed to do this is already available,
//             the exception being e_ik, the number of edges that span the i and k subtrees.
//             I use a slow algorithm O(n) to do this, since I don't know of a better way at
//             this point.

            n_i = internal[y].L->n;
            n_j = internal[y].R->n;
            n_k = internal[x].R->n;

            n_i_priv = internal[y].L->n_priv;
            n_j_priv = internal[y].R->n_priv;
            n_k_priv = internal[x].R->n_priv;

            n_y = ((double) n_i) * ((double) n_k);
            n_y_priv = ((double) n_i_priv) * ((double) n_k_priv);

            e_y = (double) computeEdgeCount(internal[y].L->index, internal[y].L->type, internal[x].R->index,
                                            internal[x].R->type, false);   // e_ik
            e_y_priv = (double) computeEdgeCount(internal[y].L->index, internal[y].L->type, internal[x].R->index,
                                                 internal[x].R->type, true);   // e_i

            if (n_y != 0) p_y = (e_y) / (n_y); else p_y = 0.0;
            if (n_y_priv != 0) p_y_priv = (e_y_priv) / (n_y_priv); else p_y_priv = 0.0;
            if (e_y == 0 or e_y == n_y or p_y == 0.0) { L_y = 0.0; }
            else { L_y = (e_y) * log(p_y) + (n_y - e_y) * log(1.0 - p_y); }
            if (p_y_priv == 0.0 || p_y_priv == 1.0) { L_y_priv = 0.0; }
            else {
                L_y_priv = (e_y_priv) * log(p_y_priv) + (n_y_priv - e_y_priv) * log(1.0 - p_y_priv);
            }

            n_x = ((double) n_i + (double) n_k) * ((double) n_j);
            n_x_priv = ((double) n_i_priv + (double) n_k_priv) * ((double) n_j_priv);
            e_x = (double) internal[x].e + (double) internal[y].e - (double) e_y;                        // e_yj
            e_x_priv = (double) computeEdgeCount(internal[y].L->index, internal[y].L->type, internal[y].R->index,
                                                 internal[y].R->type, true) +
                       (double) computeEdgeCount(internal[y].R->index, internal[y].R->type, internal[x].R->index,
                                                 internal[x].R->type, true);
            if (e_y != 0 && n_y == 0 || e_y_priv != 0 && n_y_priv == 0)
                cout << "LEFT 1 Y: " << e_y << " " << n_y << " " << e_y_priv << " " << n_y_priv << endl;
            if (e_x != 0 && n_x == 0 || e_x_priv != 0 && n_x_priv == 0)
                cout << "LEFT 1 X: " << e_x << " " << n_x << " " << e_x_priv << " " << n_x_priv << endl;
            if (n_x != 0) p_x = (e_x) / (n_x); else p_x = 0;
            if (n_x_priv != 0) p_x_priv = (e_x_priv) / (n_x_priv); else p_x_priv = 0;
            if (e_x == 0 or e_x == n_x || p_x == 0) { L_x = 0.0; }
            else { L_x = (e_x) * log(p_x) + (n_x - e_x) * log(1.0 - p_x); }
            if (p_x_priv == 0 || p_x_priv == 1) { L_x_priv = 0.0; }
            else {
                L_x_priv = (e_x_priv) * log(p_x_priv) + (n_x_priv - e_x_priv) * log(1.0 - p_x_priv);
            }
/*            if (isnan(L_x_priv) || isnan(L_y_priv)) {
                if (isnan(internal[x].logL_priv) || isnan(internal[y].logL_priv)) cout << "YES:\t";
                cout << "L1 X: " << isnan(L_x_priv) << " " << p_x << " " << p_x_priv << " Y: " << isnan(L_y_priv) << " "
                    << p_y << " " << p_y_priv << endl;
                return false;
            }
*/
            dLogL_pub = (L_x - internal[x].logL_pub) + (L_y - internal[y].logL_pub);
            dLogL_priv = (L_x_priv - internal[x].logL_priv) + (L_y_priv - internal[y].logL_priv);
            if (type_move == true) {
                dLogL = dLogL_pub;
            } else {
                dLogL = T * dLogL_priv;
            }

            if ((dLogL > 0.0) or (mtr.randExc() < exp(dLogL))) {  // make LEFT ALPHA move
                ftaken = true;
                d->swapEdges(x, internal[x].R->index, RIGHT, y, internal[y].R->index, RIGHT);
                temp = internal[x].R;            // - swap j and k
                internal[x].R = internal[y].R;            //
                internal[y].R = temp;                    //
                internal[x].R->M = &internal[x];            // - adjust parent pointers
                internal[y].R->M = &internal[y];            //
                internal[y].n = n_i + n_k;
                internal[y].n_priv = n_i_priv + n_k_priv;
                internal[x].e = (long long) e_x;                    // - update e_i for [x] and [y]
                internal[x].e_priv = (long long) e_x_priv;                    // - update e_i for [x] and [y]
                internal[y].e = (long long) e_y;                    //
                internal[y].e_priv = (long long) e_y_priv;                    //
                internal[x].p = p_x;                    // - update p_i for [x] and [y]
                internal[x].p_priv = p_x_priv;
                internal[y].p = p_y;                    //
                internal[y].p_priv = p_y_priv;
                internal[x].logL_pub = L_x;                    // - update L_i for [x] and [y]
                internal[x].logL_priv = L_x_priv;
                internal[y].logL_pub = L_y;                    //
                internal[y].logL_priv = L_y_priv;
                // - order-property maintained
//                L += dLogL_pub + dLogL_priv;                // - update LogL
                dendro::refreshLikelihood();
                delta = dLogL;                //

                // TRAP: Catches violations of the ordering property
                if (internal[x].label > internal[x].L->label or internal[y].label > internal[y].L->label) {
//                    printDendrogram();
                    if (internal[x].label > internal[x].L->label) {
                        cout << "**** WARNING - order property violated by internal[" << x << "]" << endl;
                        cout << "x    (p = " << internal[x].p << "\te = " << internal[x].e << "\tnL = "
                             << internal[x].L->n << "\tnR = " << internal[x].R->n << "\tlabel = " << internal[x].label
                             << ")\tinternal[" << x << "]\t(D)" << endl;
                        if (internal[x].L->type == GRAPH) {
                            cout << "x->L [" << internal[x].L->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->L (p = " << internal[x].L->p << "\te = " << internal[x].L->e << "\tnL = "
                                 << internal[x].L->L->n << "\tnR = " << internal[x].L->R->n << "\tlabel = "
                                 << internal[x].L->label << ")\tinternal[" << internal[x].L->index << "]\t(D)" << endl;
                        }
                        if (internal[x].R->type == GRAPH) {
                            cout << "x->R [" << internal[x].R->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->R (p = " << internal[x].R->p << "\te = " << internal[x].R->e << "\tnL = "
                                 << internal[x].R->L->n << "\tnR = " << internal[x].R->R->n << "\tlabel = "
                                 << internal[x].R->label << ")\tinternal[" << internal[x].R->index << "]\t(D)" << endl;
                        }
                    }
                    if (internal[y].label > internal[y].L->label) {
                        cout << "**** WARNING - order property violated by internal[" << y << "]" << endl;
                        cout << "y    (p = " << internal[y].p << "\te = " << internal[y].e << "\tnL = "
                             << internal[y].L->n << "\tnR = " << internal[y].R->n << "\tlabel = " << internal[y].label
                             << ")\tinternal[" << y << "]\t(D)" << endl;
                        if (internal[y].L->type == GRAPH) {
                            cout << "y->L [" << internal[y].L->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->L (p = " << internal[y].L->p << "\te = " << internal[y].L->e << "\tnL = "
                                 << internal[y].L->L->n << "\tnR = " << internal[y].L->R->n << "\tlabel = "
                                 << internal[y].L->label << ")\tinternal[" << internal[y].L->index << "]\t(D)" << endl;
                        }
                        if (internal[y].R->type == GRAPH) {
                            cout << "y->R [" << internal[y].R->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->R (p = " << internal[y].R->p << "\te = " << internal[y].R->e << "\tnL = "
                                 << internal[y].R->L->n << "\tnR = " << internal[y].R->R->n << "\tlabel = "
                                 << internal[y].R->label << ")\tinternal[" << internal[y].R->index << "]\t(D)" << endl;
                        }
                    }
                    return false;
                }
            }
        } else {                                    // ## LEFT BETA move:  ((i,j),k) -> (i,(j,k))
            n_i = internal[y].L->n;
            n_j = internal[y].R->n;
            n_k = internal[x].R->n;

            n_i_priv = internal[y].L->n_priv;
            n_j_priv = internal[y].R->n_priv;
            n_k_priv = internal[x].R->n_priv;

            n_y = (double) n_j * n_k;
            n_y_priv = (double) n_j_priv * n_k_priv;
            e_y = (double) computeEdgeCount(internal[y].R->index, internal[y].R->type, internal[x].R->index,
                                            internal[x].R->type, false);
            e_y_priv = (double) computeEdgeCount(internal[y].R->index, internal[y].R->type, internal[x].R->index,
                                                 internal[x].R->type, true);
            if (n_y != 0) p_y = (e_y) / (n_y); else p_y = 0;
            if (n_y_priv != 0) p_y_priv = (e_y_priv) / (n_y_priv); else p_y_priv = 0;
            if (e_y == 0 or e_y == n_y or p_y == 0) { L_y = 0.0; }
            else { L_y = (e_y) * log(p_y) + (n_y - e_y) * log(1.0 - p_y); }
            if (p_y_priv == 0.0 || p_y_priv == 1.0)  { L_y_priv = 0.0; }
            else {
                L_y_priv = (e_y_priv) * log(p_y_priv) + (n_y_priv - e_y_priv) * log(1.0 - p_y_priv);
            }
            // ((i,j),k) -> (i,(j,k))
            n_x = (double) (n_j + n_k) * n_i;
            n_x_priv = (double) (n_j_priv + n_k_priv) * n_i_priv;
            e_x = (double) internal[x].e + (double) internal[y].e - (double) e_y;                        // e_yj
            e_x_priv = (double) computeEdgeCount(internal[y].L->index, internal[y].L->type, internal[y].R->index,
                                                 internal[y].R->type, true) +
                       (double) computeEdgeCount(internal[y].L->index, internal[y].L->type, internal[x].R->index,
                                                 internal[x].R->type, true);
            if (e_y != 0 && n_y == 0 || e_y_priv != 0 && n_y_priv == 0)
                cout << "LEFT 2 Y: " << e_y << " " << n_y << " " << e_y_priv << " " << n_y_priv << endl;
            if (e_x != 0 && n_x == 0 || e_x_priv != 0 && n_x_priv == 0)
                cout << "LEFT 2 X: " << e_x << " " << n_x << " " << e_x_priv << " " << n_x_priv << endl;
            if (n_x != 0) p_x = (e_x) / (n_x); else p_x = 0;
            if (n_x_priv != 0) p_x_priv = (e_x_priv) / (n_x_priv); else p_x_priv = 0;
            if (e_x == 0 or e_x == n_x || p_x == 0) { L_x = 0.0; }
            else { L_x = (e_x) * log(p_x) + (n_x - e_x) * log(1.0 - p_x); }
            if (p_x_priv == 0 || p_x_priv == 1) { L_x_priv = 0.0; }
            else {
                L_x_priv = (e_x_priv) * log(p_x_priv) + (n_x_priv - e_x_priv) * log(1.0 - p_x_priv);
            }
/*
            if (isnan(L_x_priv) || isnan(L_y_priv)) {
                if (isnan(internal[x].logL_priv) || isnan(internal[y].logL_priv)) cout << "YES:\t";
                cout << "L2 X: " << isnan(L_x_priv) << " " << p_x << " " << p_x_priv << " Y: " << isnan(L_y_priv) << " "
                     << p_y << " " << p_y_priv << endl;
                return false;
            }
*/
            dLogL_pub = (L_x - internal[x].logL_pub) + (L_y - internal[y].logL_pub);
            dLogL_priv = (L_x_priv - internal[x].logL_priv) + (L_y_priv - internal[y].logL_priv);
            if (type_move == true) {
                dLogL = dLogL_pub;
            } else {
                dLogL = T * dLogL_priv;
            }
            if ((dLogL > 0.0) or (mtr.randExc() < exp(dLogL))) {  // make LEFT BETA move
                ftaken = true;
                d->swapEdges(y, internal[y].L->index, LEFT, y, internal[y].R->index, RIGHT);
                temp = internal[y].L;            // - swap L and R of [y]
                internal[y].L = internal[y].R;            //
                internal[y].R = temp;                    //
                d->swapEdges(x, internal[x].R->index, RIGHT, y, internal[y].R->index, RIGHT);
                temp = internal[x].R;            // - swap i and k
                internal[x].R = internal[y].R;            //
                internal[y].R = temp;                    //
                internal[x].R->M = &internal[x];            // - adjust parent pointers
                internal[y].R->M = &internal[y];            //
                d->swapEdges(x, internal[x].L->index, LEFT, x, internal[x].R->index, RIGHT);
                temp = internal[x].L;            // - swap L and R of [x]
                internal[x].L = internal[x].R;            //
                internal[x].R = temp;                    //
                internal[y].n = n_j + n_k;                // - update n
                internal[y].n_priv = n_j_priv + n_k_priv;
                internal[x].e = (long long) e_x;                    // - update e_i for [x] and [y]
                internal[x].e_priv = (long long) e_x_priv;                    // - update e_i for [x] and [y]
                internal[y].e = (long long) e_y;                    //
                internal[y].e_priv = (long long) e_y_priv;                    //
                internal[x].p = p_x;                    // - update p_i for [x] and [y]
                internal[x].p_priv = p_x_priv;
                internal[y].p = p_y;                    //
                internal[y].p_priv = p_y_priv;
                internal[x].logL_pub = L_x;                    // - update L_i for [x] and [y]
                internal[x].logL_priv = L_x_priv;
                internal[y].logL_pub = L_y;                    //
                internal[y].logL_priv = L_y_priv;
                if (internal[y].R->label < internal[y].L->label) {
                    d->swapEdges(y, internal[y].L->index, LEFT, y, internal[y].R->index, RIGHT);
                    temp = internal[y].L;    // - enforce order-property if necessary
                    internal[y].L = internal[y].R;    //
                    internal[y].R = temp;            //
                }                                    //
                internal[y].label = internal[y].L->label;    //
//                L += dLogL_pub + dLogL_priv;                // - update LogL
                dendro::refreshLikelihood();
                delta = dLogL;                //


                // TRAP: Catches violations of the ordering property
                if (internal[x].label > internal[x].L->label or internal[y].label > internal[y].L->label) {
//                    printDendrogram();
                    if (internal[x].label > internal[x].L->label) {
                        cout << "**** WARNING - order property violated by internal[" << x << "]" << endl;
                        cout << "x    (p = " << internal[x].p << "\te = " << internal[x].e << "\tnL = "
                             << internal[x].L->n << "\tnR = " << internal[x].R->n << "\tlabel = " << internal[x].label
                             << ")\tinternal[" << x << "]\t(D)" << endl;
                        if (internal[x].L->type == GRAPH) {
                            cout << "x->L [" << internal[x].L->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->L (p = " << internal[x].L->p << "\te = " << internal[x].L->e << "\tnL = "
                                 << internal[x].L->L->n << "\tnR = " << internal[x].L->R->n << "\tlabel = "
                                 << internal[x].L->label << ")\tinternal[" << internal[x].L->index << "]\t(D)" << endl;
                        }
                        if (internal[x].R->type == GRAPH) {
                            cout << "x->R [" << internal[x].R->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->R (p = " << internal[x].R->p << "\te = " << internal[x].R->e << "\tnL = "
                                 << internal[x].R->L->n << "\tnR = " << internal[x].R->R->n << "\tlabel = "
                                 << internal[x].R->label << ")\tinternal[" << internal[x].R->index << "]\t(D)" << endl;
                        }
                    }
                    if (internal[y].label > internal[y].L->label) {
                        cout << "**** WARNING - order property violated by internal[" << y << "]" << endl;
                        cout << "y    (p = " << internal[y].p << "\te = " << internal[y].e << "\tnL = "
                             << internal[y].L->n << "\tnR = " << internal[y].R->n << "\tlabel = " << internal[y].label
                             << ")\tinternal[" << y << "]\t(D)" << endl;
                        if (internal[y].L->type == GRAPH) {
                            cout << "y->L [" << internal[y].L->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->L (p = " << internal[y].L->p << "\te = " << internal[y].L->e << "\tnL = "
                                 << internal[y].L->L->n << "\tnR = " << internal[y].L->R->n << "\tlabel = "
                                 << internal[y].L->label << ")\tinternal[" << internal[y].L->index << "]\t(D)" << endl;
                        }
                        if (internal[y].R->type == GRAPH) {
                            cout << "y->R [" << internal[y].R->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->R (p = " << internal[y].R->p << "\te = " << internal[y].R->e << "\tnL = "
                                 << internal[y].R->L->n << "\tnR = " << internal[y].R->R->n << "\tlabel = "
                                 << internal[y].R->label << ")\tinternal[" << internal[y].R->index << "]\t(D)" << endl;
                        }
                    }
                    return false;
                }
            }
        }
    } else {                                        // right-edge: t == RIGHT
        if (mtr.randExc() < 0.5) {                    // alpha move: (i,(j,k)) -> ((i,k),j)
            n_i = internal[x].L->n;
            n_j = internal[y].L->n;
            n_k = internal[y].R->n;

            n_i_priv = internal[x].L->n_priv;
            n_j_priv = internal[y].L->n_priv;
            n_k_priv = internal[y].R->n_priv;

            n_y = (double) n_i * n_k;
            n_y_priv = (double) n_i_priv * n_k_priv;
            e_y = (double) computeEdgeCount(internal[x].L->index, internal[x].L->type, internal[y].R->index,
                                            internal[y].R->type, false);   // e_ik
            e_y_priv = (double) computeEdgeCount(internal[x].L->index, internal[x].L->type, internal[y].R->index,
                                                 internal[y].R->type, true);
            if (n_y != 0) p_y = (double) (e_y) / (double) (n_y); else p_y = 0;
            if (n_y_priv != 0) p_y_priv = (double) (e_y_priv) / (double) (n_y_priv); else p_y_priv = 0;
            if (e_y == 0 or e_y == n_y or p_y == 0) { L_y = 0.0; }
            else { L_y = (double) (e_y) * log(p_y) + (double) (n_y - e_y) * log(1.0 - p_y); }
            if (p_y_priv == 0.0 || p_y_priv == 1.0) { L_y_priv = 0.0; }
            else {
                L_y_priv = (double) (e_y_priv) * log(p_y_priv) + (double) (n_y_priv - e_y_priv) * log(1.0 - p_y_priv);
            }

            n_x = (double) (n_i + n_k) * n_j;
            n_x_priv = (double) (n_i_priv + n_k_priv) * n_j_priv;
            e_x = (double) internal[x].e + (double) internal[y].e -
                  (double) e_y;                        // alpha move: (i,(j,k)) -> ((i,k),j)
            e_x_priv = (double) computeEdgeCount(internal[x].L->index, internal[x].L->type, internal[y].L->index,
                                                 internal[y].L->type, true) +
                       (double) computeEdgeCount(internal[y].L->index, internal[y].L->type, internal[y].R->index,
                                                 internal[y].R->type, true);

            if (e_y != 0 && n_y == 0 || e_y_priv != 0 && n_y_priv == 0)
                cout << "RIGHT 1 Y: " << e_y << " " << n_y << " " << e_y_priv << " " << n_y_priv << endl;
            if (e_x != 0 && n_x == 0 || e_x_priv != 0 && n_x_priv == 0)
                cout << "RIGHT 1 X: " << e_x << " " << n_x << " " << e_x_priv << " " << n_x_priv << endl;
            if (n_x != 0) p_x = (double) (e_x) / (double) (n_x); else p_x = 0;
            if (n_x_priv != 0) p_x_priv = (double) (e_x_priv) / (double) (n_x_priv); else p_x_priv = 0;
            if (e_x == 0 or e_x == n_x || p_x == 0) { L_x = 0.0; }
            else { L_x = (double) (e_x) * log(p_x) + (double) (n_x - e_x) * log(1.0 - p_x); }
            if (p_x_priv == 0 || p_x_priv == 1) { L_x_priv = 0.0; }
            else {
                L_x_priv = (double) (e_x_priv) * log(p_x_priv) + (double) (n_x_priv - e_x_priv) * log(1.0 - p_x_priv);
            }
            if (e_x != 0 && n_x == 0 || e_y != 0 && n_y == 0)
                cout << t << " " << n_x << " " << e_x << " " << n_y << " " << e_y << endl;
/*
            if (isnan(L_x_priv) || isnan(L_y_priv)) {
                if (isnan(internal[x].logL_priv) || isnan(internal[y].logL_priv)) cout << "YES:\t";
                cout << "R1 X: " << isnan(L_x_priv) << " " << p_x << " " << p_x_priv << " Y: " << isnan(L_y_priv) << " "
                     << p_y << " " << p_y_priv << endl;
                return false;
            }
*/
            dLogL_pub = (L_x - internal[x].logL_pub) + (L_y - internal[y].logL_pub);
            dLogL_priv = (L_x_priv - internal[x].logL_priv) + (L_y_priv - internal[y].logL_priv);
            if (type_move == true) {
                dLogL = dLogL_pub;
            } else {
                dLogL = T * dLogL_priv;
            }
            if ((dLogL > 0.0) or (mtr.randExc() < exp(dLogL))) {  // make RIGHT ALPHA move
                ftaken = true;
                d->swapEdges(x, internal[x].L->index, LEFT, x, internal[x].R->index, RIGHT);
                temp = internal[x].L;            // - swap L and R of [x]
                internal[x].L = internal[x].R;            //
                internal[x].R = temp;                //
                d->swapEdges(y, internal[y].L->index, LEFT, x, internal[x].R->index, RIGHT);
                temp = internal[y].L;            // - swap i and j
                internal[y].L = internal[x].R;            //
                internal[x].R = temp;                //
                internal[x].R->M = &internal[x];            // - adjust parent pointers
                internal[y].L->M = &internal[y];            //
                internal[y].n = n_i + n_k;            // - update n
                internal[y].n_priv = n_i_priv + n_k_priv;
                internal[x].e = (long long) e_x;                    // - update e_i for [x] and [y]
                internal[x].e_priv = (long long) e_x_priv;                    // - update e_i for [x] and [y]
                internal[y].e = (long long) e_y;                    //
                internal[y].e_priv = (long long) e_y_priv;                    //
                internal[x].p = p_x;                    // - update p_i for [x] and [y]
                internal[x].p_priv = p_x_priv;
                internal[y].p = p_y;                    //
                internal[y].p_priv = p_y_priv;
                internal[x].logL_pub = L_x;                    // - update L_i for [x] and [y]
                internal[x].logL_priv = L_x_priv;
                internal[y].logL_pub = L_y;                    //
                internal[y].logL_priv = L_y_priv;
                internal[y].label = internal[x].label;        // - update order property
//                L += dLogL_pub + dLogL_priv;                // - update LogL
                dendro::refreshLikelihood();
                delta = dLogL;                //

                // TRAP: Catches violations of the ordering property
                if (internal[x].label > internal[x].L->label or internal[y].label > internal[y].L->label) {
//                    printDendrogram();
                    if (internal[x].label > internal[x].L->label) {
                        cout << "**** WARNING - order property violated by internal[" << x << "]" << endl;
                        cout << "x    (p = " << internal[x].p << "\te = " << internal[x].e << "\tnL = "
                             << internal[x].L->n << "\tnR = " << internal[x].R->n << "\tlabel = " << internal[x].label
                             << ")\tinternal[" << x << "]\t(D)" << endl;
                        if (internal[x].L->type == GRAPH) {
                            cout << "x->L [" << internal[x].L->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->L (p = " << internal[x].L->p << "\te = " << internal[x].L->e << "\tnL = "
                                 << internal[x].L->L->n << "\tnR = " << internal[x].L->R->n << "\tlabel = "
                                 << internal[x].L->label << ")\tinternal[" << internal[x].L->index << "]\t(D)" << endl;
                        }
                        if (internal[x].R->type == GRAPH) {
                            cout << "x->R [" << internal[x].R->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->R (p = " << internal[x].R->p << "\te = " << internal[x].R->e << "\tnL = "
                                 << internal[x].R->L->n << "\tnR = " << internal[x].R->R->n << "\tlabel = "
                                 << internal[x].R->label << ")\tinternal[" << internal[x].R->index << "]\t(D)" << endl;
                        }
                    }
                    if (internal[y].label > internal[y].L->label) {
                        cout << "**** WARNING - order property violated by internal[" << y << "]" << endl;
                        cout << "y    (p = " << internal[y].p << "\te = " << internal[y].e << "\tnL = "
                             << internal[y].L->n << "\tnR = " << internal[y].R->n << "\tlabel = " << internal[y].label
                             << ")\tinternal[" << y << "]\t(D)" << endl;
                        if (internal[y].L->type == GRAPH) {
                            cout << "y->L [" << internal[y].L->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->L (p = " << internal[y].L->p << "\te = " << internal[y].L->e << "\tnL = "
                                 << internal[y].L->L->n << "\tnR = " << internal[y].L->R->n << "\tlabel = "
                                 << internal[y].L->label << ")\tinternal[" << internal[y].L->index << "]\t(D)" << endl;
                        }
                        if (internal[y].R->type == GRAPH) {
                            cout << "y->R [" << internal[y].R->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->R (p = " << internal[y].R->p << "\te = " << internal[y].R->e << "\tnL = "
                                 << internal[y].R->L->n << "\tnR = " << internal[y].R->R->n << "\tlabel = "
                                 << internal[y].R->label << ")\tinternal[" << internal[y].R->index << "]\t(D)" << endl;
                        }
                    }
                    return false;
                }

            }
        } else {                                    // beta move:  (i,(j,k)) -> ((i,j),k)
            n_i = internal[x].L->n;
            n_j = internal[y].L->n;
            n_k = internal[y].R->n;

            n_i_priv = internal[x].L->n_priv;
            n_j_priv = internal[y].L->n_priv;
            n_k_priv = internal[y].R->n_priv;

            n_y = (double) n_i * n_j;
            n_y_priv = (double) n_i_priv * n_j_priv;
            e_y = (double) computeEdgeCount(internal[x].L->index, internal[x].L->type, internal[y].L->index,
                                            internal[y].L->type, false);   // e_ij
            e_y_priv = (double) computeEdgeCount(internal[x].L->index, internal[x].L->type, internal[y].L->index,
                                                 internal[y].L->type, true);
            if (n_y != 0) p_y = (double) (e_y) / (double) (n_y); else p_y = 0;
            if (n_y_priv != 0) p_y_priv = (double) (e_y_priv) / (double) (n_y_priv); else p_y_priv = 0;
            if (e_y == 0 or e_y == n_y or p_y == 0) { L_y = 0.0; }
            else { L_y = (double) (e_y) * log(p_y) + (double) (n_y - e_y) * log(1.0 - p_y); }
            if (p_y_priv == 0.0 || p_y_priv == 1.0) { L_y_priv = 0.0; }
            else {
                L_y_priv = (double) (e_y_priv) * log(p_y_priv) + (double) (n_y_priv - e_y_priv) * log(1.0 - p_y_priv);
            }

            n_x = (double) (n_i + n_j) * n_k;
            n_x_priv = (double) (n_i_priv + n_j_priv) * n_k_priv;
            e_x = (double) internal[x].e + (double) internal[y].e -
                  (double) e_y;                         // beta move:  (i,(j,k)) -> ((i,j),k)
            e_x_priv = (double) computeEdgeCount(internal[x].L->index, internal[x].L->type, internal[y].R->index,
                                                 internal[y].R->type, true) +
                       (double) computeEdgeCount(internal[y].L->index, internal[y].L->type, internal[y].R->index,
                                                 internal[y].R->type, true);
            if (e_y != 0 && n_y == 0 || e_y_priv != 0 && n_y_priv == 0)
                cout << "RIGHT 2 Y: " << e_y << " " << n_y << " " << e_y_priv << " " << n_y_priv << endl;
            if (e_x != 0 && n_x == 0 || e_x_priv != 0 && n_x_priv == 0)
                cout << "RIGHT 2 X: " << e_x << " " << n_x << " " << e_x_priv << " " << n_x_priv << endl;
            if (n_x != 0) p_x = (double) (e_x) / (double) (n_x); else p_x = 0;
            if (n_x_priv != 0) p_x_priv = (double) (e_x_priv) / (double) (n_x_priv); else p_x_priv = 0;
            if (e_x == 0 or e_x == n_x || p_x == 0) { L_x = 0.0; }
            else { L_x = (double) (e_x) * log(p_x) + (double) (n_x - e_x) * log(1.0 - p_x); }
            if (p_x_priv == 0 || p_x_priv == 1) { L_x_priv = 0.0; }
            else {
                L_x_priv = (double) (e_x_priv) * log(p_x_priv) + (double) (n_x_priv - e_x_priv) * log(1.0 - p_x_priv);
            }
//            if (e_x != 0 && n_x == 0 || e_y != 0 && n_y == 0)
//                cout << t << " " << n_x << " " << e_x << " " << n_y << " " << e_y << endl;
/*
            if (isnan(L_x_priv) || isnan(L_y_priv)) {
                if (isnan(internal[x].logL_priv) || isnan(internal[y].logL_priv)) cout << "YES:\t";
                cout << "R2 X: " << isnan(L_x_priv) << " " << p_x << " " << p_x_priv << " Y: " << isnan(L_y_priv) << " "
                     << p_y << " " << p_y_priv << endl;
                return false;
            }
*/
            dLogL_pub = (L_x - internal[x].logL_pub) + (L_y - internal[y].logL_pub);
            dLogL_priv = (L_x_priv - internal[x].logL_priv) + (L_y_priv - internal[y].logL_priv);
//            if (isnan(dLogL_pub) || isnan(dLogL_priv)) cout << "R2 X: " << p_x << " " << p_x_priv << " Y: " << p_y << " " << p_y_priv << endl;
            if (type_move == true) {
                dLogL = dLogL_pub;
            } else {
                dLogL = T * dLogL_priv;
            }
            if ((dLogL > 0.0) or (mtr.randExc() < exp(dLogL))) {  // make RIGHT BETA move
                ftaken = true;
                d->swapEdges(x, internal[x].L->index, LEFT, x, internal[x].R->index, RIGHT);
                temp = internal[x].L;            // - swap L and R of [x]
                internal[x].L = internal[x].R;            //
                internal[x].R = temp;                //
                d->swapEdges(x, internal[x].R->index, RIGHT, y, internal[y].R->index, RIGHT);
                temp = internal[x].R;            // - swap i and k
                internal[x].R = internal[y].R;            //
                internal[y].R = temp;                //
                internal[x].R->M = &internal[x];            // - adjust parent pointers
                internal[y].R->M = &internal[y];            //
                d->swapEdges(y, internal[y].L->index, LEFT, y, internal[y].R->index, RIGHT);
                temp = internal[y].L;            // - swap L and R of [y]
                internal[y].L = internal[y].R;            //
                internal[y].R = temp;                //
                internal[y].n = n_i + n_j;            // - update n
                internal[y].n_priv = n_i_priv + n_j_priv;
//                internal[x].n_priv = n_x_priv;
                internal[x].e = (long long) e_x;                    // - update e_i for [x] and [y]
                internal[x].e_priv = (long long) e_x_priv;                    // - update e_i for [x] and [y]
                internal[y].e = (long long) e_y;                    //
                internal[y].e_priv = (long long) e_y_priv;                    //
                internal[x].p = p_x;                    // - update p_i for [x] and [y]
                internal[x].p_priv = p_x_priv;
                internal[y].p = p_y;                    //
                internal[y].p_priv = p_y_priv;
                internal[x].logL_pub = L_x;                    // - update L_i for [x] and [y]
                internal[x].logL_priv = L_x_priv;
                internal[y].logL_pub = L_y;                    //
                internal[y].logL_priv = L_y_priv;                    //
                internal[y].label = internal[x].label;        // - order-property
//                L += dLogL_pub + dLogL_priv;                // - update LogL
                dendro::refreshLikelihood();
                delta = dLogL;                //

                // TRAP: Catches violations of the ordering property
                if (internal[x].label > internal[x].L->label or internal[y].label > internal[y].L->label) {
//                    printDendrogram();
                    if (internal[x].label > internal[x].L->label) {
                        cout << "**** WARNING - order property violated by internal[" << x << "]" << endl;
                        cout << "x    (p = " << internal[x].p << "\te = " << internal[x].e << "\tnL = "
                             << internal[x].L->n << "\tnR = " << internal[x].R->n << "\tlabel = " << internal[x].label
                             << ")\tinternal[" << x << "]\t(D)" << endl;
                        if (internal[x].L->type == GRAPH) {
                            cout << "x->L [" << internal[x].L->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->L (p = " << internal[x].L->p << "\te = " << internal[x].L->e << "\tnL = "
                                 << internal[x].L->L->n << "\tnR = " << internal[x].L->R->n << "\tlabel = "
                                 << internal[x].L->label << ")\tinternal[" << internal[x].L->index << "]\t(D)" << endl;
                        }
                        if (internal[x].R->type == GRAPH) {
                            cout << "x->R [" << internal[x].R->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->R (p = " << internal[x].R->p << "\te = " << internal[x].R->e << "\tnL = "
                                 << internal[x].R->L->n << "\tnR = " << internal[x].R->R->n << "\tlabel = "
                                 << internal[x].R->label << ")\tinternal[" << internal[x].R->index << "]\t(D)" << endl;
                        }
                    }
                    if (internal[y].label > internal[y].L->label) {
                        cout << "**** WARNING - order property violated by internal[" << y << "]" << endl;
                        cout << "y    (p = " << internal[y].p << "\te = " << internal[y].e << "\tnL = "
                             << internal[y].L->n << "\tnR = " << internal[y].R->n << "\tlabel = " << internal[y].label
                             << ")\tinternal[" << y << "]\t(D)" << endl;
                        if (internal[y].L->type == GRAPH) {
                            cout << "y->L [" << internal[y].L->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->L (p = " << internal[y].L->p << "\te = " << internal[y].L->e << "\tnL = "
                                 << internal[y].L->L->n << "\tnR = " << internal[y].L->R->n << "\tlabel = "
                                 << internal[y].L->label << ")\tinternal[" << internal[y].L->index << "]\t(D)" << endl;
                        }
                        if (internal[y].R->type == GRAPH) {
                            cout << "y->R [" << internal[y].R->index << "]\t(G)" << endl;
                        } else {
                            cout << "i->R (p = " << internal[y].R->p << "\te = " << internal[y].R->e << "\tnL = "
                                 << internal[y].R->L->n << "\tnR = " << internal[y].R->R->n << "\tlabel = "
                                 << internal[y].R->label << ")\tinternal[" << internal[y].R->index << "]\t(D)" << endl;
                        }
                    }
                    return false;
                }
                if (internal[x].label > internal[x].L->label) {
                    tempe = internal[x].L;
                    internal[x].L = internal[x].R;
                    internal[x].R = tempe;
                }
                if (internal[y].label > internal[y].L->label) {
                    tempe = internal[y].L;
                    internal[y].L = internal[y].R;
                    internal[y].R = tempe;
                }

            }
        }
    }
    return true;
}

// ********************************************************************************************************

void dendro::printDendrogram() {
    cout << "\nLEAFS = " << n << endl << "# ";
    printSubTree(root);
    return;
}

void dendro::printSubTree(elementd *z) {
    if (z != NULL) {
        if (z->type == GRAPH) {
            cout << "[" << z->label << "] " << z->n_priv << " " << z->n
                 << endl; //"\t(" << z->L << " " << z->R << ") - " << z->M <<  endl;
            return;
        } else if (z->type == DENDRO) {
            cout << "(p = " << z->p << "\tp_priv = " << z->p_priv << "\te = " << z->e << "\te_priv = " << z->e_priv
                 << "\tnL = " << z->L->n << "\tnR = " << z->R->n << "\tnL_priv = " << z->L->n_priv << "\tnR_priv = "
                 << z->R->n_priv << "\tlabel = "
                 << z->label << ")\tinternal[" << z->index << "]"
                 << endl; // "\t(" << z->L << " " << z->R << ") - " << z->M <<  endl;
            cout << "L ";
            printSubTree(z->L);
            cout << endl;
            cout << "R ";
            printSubTree(z->R);
            cout << endl;
        } else {
            cout << "(p = " << z->p << "\te = " << z->e << "\tnL = " << z->L->n << "\tnR = " << z->R->n << "\tlabel = "
                 << z->label << ")\tinternal[" << z->index << "] " << z->type << endl;
            cout << "L ";
            printSubTree(z->L);
            cout << endl;
            cout << "R ";
            printSubTree(z->R);
            cout << endl;
        }
    }
    return;
}

// ********************************************************************************************************

void dendro::refreshLikelihood() {
    // recalculates the log-likelihood of the dendrogram structure
    L = 0.0;
    double dL_pub, dL_priv;
    double nL_nR, ei;
    double nL_nR_priv, ei_priv;
    for (int i = 0; i < (n - 1); i++) {
        nL_nR = ((double) internal[i].L->n) * ((double) internal[i].R->n);
        ei = (double) internal[i].e;
        nL_nR_priv = ((double) internal[i].L->n_priv) * ((double) internal[i].R->n_priv);
        ei_priv = (double) internal[i].e_priv;
        if (nL_nR != 0) internal[i].p = (ei) / (nL_nR);
        else internal[i].p = 0;
        if (nL_nR_priv != 0)internal[i].p_priv = (ei_priv) / (nL_nR_priv);
        else internal[i].p_priv = 0;
        if (ei == 0 or ei == nL_nR or internal[i].p == 0) { dL_pub = 0.0; }
        else { dL_pub = ei * log(internal[i].p) + (nL_nR - ei) * log(1.0 - internal[i].p); }
        if (ei_priv == 0 or ei_priv == nL_nR_priv or internal[i].p_priv == 0) { dL_priv = 0.0; }
        else { dL_pub = ei_priv * log(internal[i].p_priv) + (nL_nR_priv - ei_priv) * log(1.0 - internal[i].p_priv); }
        internal[i].logL_pub = dL_pub;
        internal[i].logL_priv = dL_priv;
        L += dL_pub + dL_priv;
    }
    return;
}

// ********************************************************************************************************

void dendro::QsortMain(block *array, int left, int right) {
    if (right > left) {
        int pivot = left;
        int part = QsortPartition(array, left, right, pivot);
        QsortMain(array, left, part - 1);
        QsortMain(array, part + 1, right);
    }
    return;
}

int dendro::QsortPartition(block *array, int left, int right, int index) {
    block p_value, temp;
    p_value.x = array[index].x;
    p_value.y = array[index].y;

    // swap(array[p_value], array[right])
    temp.x = array[right].x;
    temp.y = array[right].y;
    array[right].x = array[index].x;
    array[right].y = array[index].y;
    array[index].x = temp.x;
    array[index].y = temp.y;

    int stored = left;
    for (int i = left; i < right; i++) {
        if (array[i].x <= p_value.x) {
            // swap(array[stored], array[i])
            temp.x = array[i].x;
            temp.y = array[i].y;
            array[i].x = array[stored].x;
            array[i].y = array[stored].y;
            array[stored].x = temp.x;
            array[stored].y = temp.y;
            stored++;
        }
    }
    // swap(array[right], array[stored])
    temp.x = array[stored].x;
    temp.y = array[stored].y;
    array[stored].x = array[right].x;
    array[stored].y = array[right].y;
    array[right].x = temp.x;
    array[right].y = temp.y;

    return stored;
}

// ********************************************************************************************************

void dendro::recordDendrogramStructure(const string out_file) {

    ofstream fout(out_file.c_str(), ios::trunc);
    for (int i = 0; i < (n - 1); i++) {
        fout << "[ " << i << " ] ";
        fout << "L= " << internal[i].L->index << " ";
        if (internal[i].L->type == DENDRO) { fout << "(D) "; } else { fout << "(G) "; }
        fout << "R= " << internal[i].R->index << " ";
        if (internal[i].R->type == DENDRO) { fout << "(D) "; } else { fout << "(G) "; }
        fout << "p= " << internal[i].p << " ";
        fout << "e= " << internal[i].e << " ";
        fout << "n= " << internal[i].n << "\n";
    }
    fout.close();

    return;
}

// ********************************************************************************************************

void dendro::recordGraphStructure(const string out_file, graph *random_g) {
    edge *curr;
    string thisName;
    bool flag_debug = true;
    if (flag_debug) { cout << ">> dendro: writing random graph to file" << endl; }

    ofstream fout(out_file.c_str(), ios::trunc);
    for (int i = 0; i < n; i++) {
        curr = random_g->getNeighborList(i, false);
        thisName = g->getName(i);      //get name from original graph
        while (curr != NULL) {
            if (thisName == "") { fout << i << "\t" << curr->x << "\n"; }
            else { fout << thisName << "\t" << g->getName(curr->x) << "\n"; }
            curr = curr->next;
        }
    }
    fout.close();

    return;
}

// ********************************************************************************************************

list *dendro::reversePathToRoot(const int leafIndex) {
    list *head, *subhead, *newlist;
    head = subhead = newlist = NULL;
    elementd *current = &leaf[leafIndex];

    while (current != NULL) {                // continue until we're finished
        newlist = new list;            // add this node to the path
        newlist->x = current->index;
        newlist->next = NULL;
        if (head == NULL) { head = newlist; }
        else {
            subhead = head;
            head = newlist;
            head->next = subhead;
        }
        current = current->M;
    }
    return head;
}

// ********************************************************************************************************
// ********************************************************************************************************

#endif
