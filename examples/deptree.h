#ifndef DEPTREE_H
#define DEPTREE_H
using namespace std;

const unsigned int DUMMY_ROOT = 0;
const string DUMMY_ROOT_STR = "ROOT";

struct DepEdge {
    unsigned head;
    unsigned modifier;
    unsigned relation;

    explicit DepEdge(unsigned head, unsigned modifier, unsigned relation);
    void print(cnn::Dict& depreldict) const;
    void print() const;

    bool operator <(const DepEdge& other) const {
        if (head < other.head)
            return true;
        else if (head > other.head)
            return false;
        if (modifier < other.modifier)
            return true;
        else if (modifier > other.modifier)
            return false;
        if (relation < other.relation)
            return true;
        return false;
    }

    bool operator ==(const DepEdge& other) const {
        return (head == other.head) && (modifier == other.modifier)
                && (relation == other.relation);
    }

};

struct DepTree {
    unsigned numnodes;
    unsigned root;
    vector<unsigned> parents;
    vector<unsigned> sent;
    vector<unsigned> deprels;
    set<unsigned> leaves;
    map<unsigned, vector<unsigned>> children;

    vector<unsigned> dfo; // depth-first ordering of the nodes
    vector<DepEdge> dfo_msgs; // depth-first ordering in terms of the edges (bottom-up + top-down)
    map<DepEdge, unsigned> msgdict; // given a message edge, gives the order/id associated with it
    map<unsigned, vector<unsigned>> msg_nbrs; // all incoming msg ids for a msg id, except destination
    unsigned nummsgs;

    map<unsigned, vector<unsigned>> node_msg_nbrs; // all incoming msg ids to a node

    explicit DepTree(vector<unsigned> parents, vector<unsigned> deprels,
            vector<unsigned> sent);

    void printTree(cnn::Dict& tokdict, cnn::Dict& depreldict);
    vector<unsigned> get_children(unsigned& node);
    vector<DepEdge> get_all_edges();

private:
    map<DepEdge, vector<DepEdge>*> neighbors;

    void set_children_and_root();
    void set_leaves();
    void set_dfo();
    void set_dfo_msgs();
    void set_msg_neighbors();
    void set_incoming_messages();
};
#endif //DEPTREE_H
