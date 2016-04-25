using namespace std;
#include "deptree.h"

DepEdge::DepEdge(unsigned head, unsigned modifier, unsigned relation) {
    this->head = head;
    this->modifier = modifier;
    this->relation = relation;
}

void DepEdge::print(cnn::Dict& depreldict) const {
    cerr << head << "-" << depreldict.Convert(relation) << "->" << modifier;
}

void DepEdge::print() const {
    cerr << head << "->" << modifier;
}

DepTree::DepTree(vector<unsigned> parents, vector<unsigned> deprels,
        vector<unsigned> sent) {
    this->numnodes = parents.size(); // = (length of the sentence + 1) to accommodate root
    this->parents = parents;
    this->sent = sent;
    this->deprels = deprels;

    // DO NOT change order -- ugh!!
    set_children_and_root();
    set_leaves();
    set_dfo();
    set_dfo_msgs();
    nummsgs = dfo_msgs.size();
    set_msg_neighbors();
    set_incoming_messages();

}

void DepTree::printTree(cnn::Dict& tokdict, cnn::Dict& depreldict) {
    cerr << "\nTree for sentence \"";
    for (unsigned int i = 1; i < numnodes; i++) {
        cerr << tokdict.Convert(sent[i]) << " ";
    }
    cerr << "\"" << endl;

    for (auto itr = children.begin(); itr != children.end(); ++itr) {
        unsigned node = itr->first;
        for (unsigned child : get_children(node)) {
            cerr << node << "-" << depreldict.Convert(deprels[child]) << "->"
                    << child << endl;
        }
    }

    cerr << "Leaves: ";
    for (unsigned leaf : leaves)
        cerr << leaf << " ";
    cerr << endl;

    cerr << "Depth-first Ordering:" << endl;
    for (unsigned node : dfo) {
        cerr << node << "...";
    }
    cerr << endl;

    for (DepEdge msg : dfo_msgs) {
        msg.print(depreldict);
        cerr << endl;
    }
    cerr << endl;
}

vector<unsigned> DepTree::get_children(unsigned& node) {
    if (children.find(node) == children.end()) {
        return vector<unsigned>();
    } else {
        return children[node];
    }

}

vector<DepEdge> DepTree::get_all_edges() {
    vector < DepEdge > edges;
    for (unsigned node = 1; node < parents.size(); ++node) {
        edges.push_back(DepEdge(parents[node], node, deprels[node]));
    }
    return edges;
}

void DepTree::set_children_and_root() {
    for (unsigned child = 1; child < numnodes; child++) {
        unsigned parent = parents[child];
        if (parent == DUMMY_ROOT) {
            root = child;
        }
        vector<unsigned> clist;
        if (children.find(parent) != children.end()) {
            clist = children[parent];
        }
        clist.push_back(child);
        //children.insert(make_pair(parent, clist));
        children[parent] = clist;
    }
}

void DepTree::set_leaves() {
    for (unsigned node = 0; node < numnodes; ++node) {
        if (get_children(node).size() == 0)
            leaves.insert(node);
    }
}

void DepTree::set_dfo() {
    vector<unsigned> stack;
    set<unsigned> seen;
    stack.push_back(DUMMY_ROOT);

    while (!stack.empty()) {
        int top = stack.back();

        if (children.find(top) != children.end()
                && seen.find(top) == seen.end()) {
            vector<unsigned> clist = children[top];
            for (auto itr2 = clist.rbegin(); itr2 != clist.rend(); ++itr2) {
                stack.push_back(*itr2);
            }
            seen.insert(top);
        } else if (children.find(top) != children.end()
                && seen.find(top) != seen.end()) {
            unsigned tobepopped = stack.back();
            dfo.push_back(tobepopped);
            stack.pop_back();
        } else {
            unsigned tobepopped = stack.back();
            dfo.push_back(tobepopped);
            stack.pop_back();
        }
    }
    assert(dfo.size() == numnodes);
    // not retaining the dummy root node in dfo
    assert(dfo.back() == DUMMY_ROOT);
    dfo.pop_back();
}

void DepTree::set_dfo_msgs() { // does not include root -> dummyroot and dummyroot -> root
    unsigned messageid = 0;
    for (unsigned node : dfo) {
        if (node == root) {
            continue;
        }
        DepEdge bottomup(node, parents[node], deprels[node]);
        dfo_msgs.push_back(bottomup);
        msgdict.insert(make_pair(bottomup, messageid));
        ++messageid;
    }
    for (auto itr = dfo.rbegin(); itr != dfo.rend(); itr++) {
        unsigned node = *itr;
        if (node == root) {
            continue;
        }
        DepEdge topdown(parents[node], node, deprels[node]);
        dfo_msgs.push_back(topdown);
        msgdict.insert(make_pair(topdown, messageid));
        ++messageid;
    }
    assert(dfo_msgs.size() == (2 * (numnodes - 2)));
}

void DepTree::set_msg_neighbors() {
    for (unsigned edgnum = 0; edgnum < dfo_msgs.size(); ++edgnum) {
        DepEdge edge = dfo_msgs[edgnum];
        vector < DepEdge > *neighbor_list = new vector<DepEdge>();

        neighbors.insert(make_pair(edge, neighbor_list));
        msg_nbrs.insert(
                pair<unsigned, vector<unsigned>>(edgnum, vector<unsigned>()));

        unsigned node = edge.head; // main node in question...
        unsigned avoid = edge.modifier;

        if (node != root && parents[node] != avoid) { // we don't want dummyroot -> root edge
            DepEdge np(parents[node], node, deprels[node]);
            auto nbr_id = msgdict.find(np);
            assert(nbr_id != msgdict.end());
            neighbor_list->push_back(np);
            msg_nbrs[edgnum].push_back(nbr_id->second);
        }

        vector<unsigned> children = get_children(node);
        for (unsigned child : children) {
            if (child == avoid) {
                continue;
            }
            DepEdge nc(child, node, deprels[child]);
            neighbor_list->push_back(nc);
            auto msgid = msgdict.find(nc);
            assert(msgid != msgdict.end());
            msg_nbrs[edgnum].push_back(msgid->second);
        }
        assert(neighbors.find(edge) != neighbors.end());
    }
    assert(neighbors.size() == dfo_msgs.size());
    assert(neighbors.size() == msg_nbrs.size());
}

void DepTree::set_incoming_messages() {
    for (unsigned node = 1; node < numnodes; ++node) {
        node_msg_nbrs.insert(
                pair<unsigned, vector<unsigned>>(node, vector<unsigned>()));
    }
    for (DepEdge msg : dfo_msgs) {
        auto t = msgdict.find(msg);
        assert(t != msgdict.end());
        node_msg_nbrs[msg.modifier].push_back(t->second);
    }
    assert(node_msg_nbrs.size() == numnodes - 1);
}

void test_tree(DepTree t, cnn::Dict tokdict, cnn::Dict depreldict) {
    assert(t.msg_nbrs.size() == t.dfo_msgs.size());
    if (t.numnodes <= 10) {
        t.printTree(tokdict, depreldict);
        cerr << "\nEdge DFO:\n";
        for (DepEdge e : t.dfo_msgs) {
            e.print(depreldict);
            cerr << "(";
            auto nid = t.msgdict.find(e);
            assert(nid != t.msgdict.end());
            cerr << nid->second << ")\t";
            cerr << endl;
        }

        cerr << "\nEdge Neighbors:\n";
        for (DepEdge e : t.dfo_msgs) {
            e.print();
            cerr << " has neighbors: ";
            auto mid = t.msgdict.find(e);
            assert(mid != t.msgdict.end());

            vector<unsigned> neighbors = t.msg_nbrs[mid->second];

            for (unsigned i : neighbors) {
                DepEdge ee = t.dfo_msgs[i];
                ee.print();
                cerr << "(";
                auto nid = t.msgdict.find(ee);
                assert(nid != t.msgdict.end());
                cerr << nid->second << ")\t";
            }
            cerr << endl;
        }
        cerr << endl;

        cerr << "\nNode - Incoming Messages:\n";
        for (unsigned n = 1; n < t.numnodes; ++n) {
            cerr << n << " will receive  ";
            vector<unsigned> neighbors = t.node_msg_nbrs[n];

            for (unsigned i : neighbors) {
                DepEdge ee = t.dfo_msgs[i];
                ee.print();
                cerr << "(";
                auto nid = t.msgdict.find(ee);
                assert(nid != t.msgdict.end());
                cerr << nid->second << ")\t";
            }
            cerr << endl;
        }
        cerr << endl;
    }
}

