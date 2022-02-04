// ****************************************************************************************************
// *** COPYRIGHT NOTICE *******************************************************************************
// graph.h - graph data structure for hierarchical random graphs
// Copyright (C) 2005-2008 Aaron Clauset
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
// You should have received a copy of the GNU General Public License;
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
// Created      : 8 November 2005
// Modified     : 23 December 2007 (cleaned up for public consumption)
//
// ****************************************************************************************************
// 
// Graph data structure for hierarchical random graphs. The basic structure is an adjacency list of
// edges; however, many additional pieces of metadata are stored as well. Each node stores its
// external name, its degree and (if assigned) its group index.
// 
// ****************************************************************************************************
// ****************************************************************************************************
// Author       : Khang Tran ( kt36@njit.edu )
// Collaborators: Hai Phan
// Project      : Privacy Preserving for Graph Neural Network
// Location     : New Jersey Institute of Technology
// Modified     : 04 Feb 2022
// ****************************************************************************************************

#if !defined(graph_INCLUDED)
#define graph_INCLUDED

#include <stdio.h>
#include <string>
#include "stdlib.h"

#include "rbtree.h"

using namespace std;

// ******** Basic Structures ******************************************************************************

#if !defined(edge_INCLUDED)
#define edge_INCLUDED

class edge {
public:
    int x;                // stored integer value  (edge terminator)
    edge *next;            // pointer to next elementd
    bool priv;
    edge();~edge();
};

edge::edge() {x = -1;next = NULL;priv = false;}
edge::~edge() {}

#endif

#if !defined(vert_INCLUDED)
#define vert_INCLUDED
class vert {
public:
    string name;            // (external) name of vertex
    long long degree;            // degree of this vertex
    int priv;
    vert();~vert();
};

vert::vert() {
    name = "";
    degree = 0;
    priv = -1;
}
vert::~vert() {}
#endif

// ******** Graph Class with Edge Statistics *************************************************************

class graph {
public:
    graph(const int);
    ~graph();

    bool addLink(const int, const int, const bool);                    // add (i,j) to graph
    bool doesLinkExist(const int, const int, const bool);                // true if (i,j) is already in graph
    long long getDegree(const int);                            // returns degree of vertex i
    string getName(const int);                                // returns name of vertex i
    int getType(const int);
    edge *getNeighborList(const int, const bool);                        // returns edge list of vertex i
    long long numLinks();                                    // returns m
    int numNodes();                                    // returns n
    void printPairs();                                    // prints all edges in graph
    int numNodePrivate();
    bool setName(const int, const string);                    // set name of vertex i

private:
    vert* nodes;            // list of nodes
    edge** nodeLink;            // linked list of neighbors to vertex
    edge** nodeLinkPrivate;     // linked list of private neighbor
    edge** nodeLinkTail;        // pointers to tail of neighbor list
    edge** nodeLinkTailPrivate;
    int n;                // number of vertices
    int n_priv;
    long m;                // number of directed edges
};

// ******** Constructor / Destructor **********************************************************************

graph::graph(const int size) {
    n = size;
    n_priv = 0;
    m = 0;
    nodes = new vert[n];
    nodeLink = new edge* [n];
    nodeLinkPrivate = new edge* [n];
    nodeLinkTail = new edge* [n];
    nodeLinkTailPrivate = new edge* [n];
    for (int i = 0; i < n; i++) {
        nodeLink[i] = NULL;
        nodeLinkPrivate[i] = NULL;
        nodeLinkTail[i] = NULL;
        nodeLinkTailPrivate[i] = NULL;
    }
}

graph::~graph() {
    edge *curr, *prev;
    for (int i = 0; i < n; i++) {
        curr = nodeLink[i];
        while (curr != NULL) {
            prev = curr;
            curr = curr->next;
            delete prev;
        }
    }
    for (int i = 0; i < n; i++) {
        curr = nodeLinkPrivate[i];
        while (curr != NULL) {
            prev = curr;
            curr = curr->next;
            delete prev;
        }
    }
    delete[] nodeLink;
    delete[] nodeLinkPrivate;
    nodeLink = NULL;
    nodeLinkPrivate = NULL;
    delete[] nodeLinkTail;
    delete[] nodeLinkTailPrivate;
    nodeLinkTail = NULL;
    nodeLinkTailPrivate = NULL;
    delete[] nodes;
    nodes = NULL;
}

// ********************************************************************************************************

bool graph::addLink(const int i, const int j, const bool type) {
    // Adds the directed edge (i,j) to the adjacency list for v_i
    edge *newedge;
    if (i >= 0 and i < n and j >= 0 and j < n) {
        newedge = new edge;
        newedge->x = j;
        newedge->priv = type;
        if (type == false) {
            if (nodeLink[i] == NULL && nodeLinkPrivate[i] == NULL) {            // first neighbor
                nodeLink[i] = newedge;
                nodeLinkTail[i] = newedge;
                nodes[i].priv = 0;
                nodes[i].degree = 1;
            }
            else if (nodeLink[i] != NULL && nodeLinkPrivate[i] == NULL) {
                nodeLinkTail[i]->next = newedge;
                nodeLinkTail[i]       = newedge;
                nodes[i].degree++;
            }
            else if (nodeLink[i] == NULL && nodeLinkPrivate[i] != NULL) {
                nodeLink[i] = newedge;
                nodeLinkTail[i] = newedge;
                nodes[i].degree++;
                nodes[i].priv = 2;
            }
            else {
                nodeLinkTail[i]->next = newedge;
                nodeLinkTail[i] = newedge;
                nodes[i].degree++;
            }
            m++;                                // increment edge count
        } else {
            if (nodeLink[i] == NULL && nodeLinkPrivate[i] == NULL) {            // first neighbor
                nodeLinkPrivate[i] = newedge;
                nodeLinkTailPrivate[i] = newedge;
                nodes[i].priv = 1;
                nodes[i].degree = 1;
            }
            else if (nodeLink[i] == NULL && nodeLinkPrivate[i] != NULL) {
                nodeLinkTailPrivate[i]->next = newedge;
                nodeLinkTailPrivate[i] = newedge;
                nodes[i].degree++;
            }
            else if (nodeLink[i] != NULL && nodeLinkPrivate[i] == NULL) {
                nodeLinkPrivate[i] = newedge;
                nodeLinkTailPrivate[i] = newedge;
                nodes[i].degree++;
                nodes[i].priv = 2;
            }
            else {
                nodeLinkTailPrivate[i]->next = newedge;
                nodeLinkTailPrivate[i] = newedge;
                nodes[i].degree++;
            }
            m++;                                // increment edge count
        }

        return true;
    } else { return false; }
}

// ********************************************************************************************************

bool graph::doesLinkExist(const int i, const int j, const bool type) {
    // This function determines if the edge (i,j) already exists in the adjacency list of v_i
    edge *curr;
    if (i >= 0 and i < n and j >= 0 and j < n) {
        if (type == false) curr = nodeLink[i];
        else curr = nodeLinkPrivate[i];
        while (curr != NULL) {
            if (curr->x == j) { return true; }
            curr = curr->next;
        }
    }
    return false;
}

// ********************************************************************************************************

long long graph::getDegree(const int i) { if (i >= 0 and i < n) { return nodes[i].degree; } else { return -1; }}

string graph::getName(const int i) { if (i >= 0 and i < n) { return nodes[i].name; } else { return ""; }}

// NOTE: Returns address; deallocation of returned object is dangerous
edge* graph::getNeighborList(const int i, const bool type) {
    if (i >= 0 and i < n) {
        if (type == false) return nodeLink[i];
        else return nodeLinkPrivate[i];
    } else { return NULL; }
}

// ********************************************************************************************************

long long int graph::numLinks() {
    return m;
}

int graph::numNodePrivate() {
    n_priv = 0;
    for (int i = 0; i < n; i++) {
        if (nodes[i].priv == 1 || nodes[i].priv == 2) n_priv++;
    }
    return n_priv;
}

int graph::numNodes() {
    return n;
}

int graph::getType(const int i) { if (i >= 0 and i < n) { return nodes[i].priv; } else { return -1; }}

// ********************************************************************************************************

void graph::printPairs() {
    edge *curr;
    for (int i = 0; i < n; i++) {
        cout << "[" << i << "]\t";
        curr = nodeLink[i];
        while (curr != NULL) {
            cout << curr->x << "\t";
            curr = curr->next;
        }
        cout << "\n";
    }
    return;
}

// ********************************************************************************************************
// ********************************************************************************************************

bool graph::setName(const int i, const string text) {
    if (i >= 0 and i < n) {
        nodes[i].name = text;
        return true;
    } else { return false; }
}

// ********************************************************************************************************
// ********************************************************************************************************

#endif
