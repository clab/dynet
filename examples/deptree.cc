using namespace std;

const unsigned int DUMMY_ROOT = 0;
const string DUMMY_ROOT_STR = "ROOT";

struct DepEdge {
    unsigned head;
    unsigned modifier;
    unsigned relation;

    explicit DepEdge(unsigned head, unsigned modifier, unsigned relation) {
        this->head = head;
        this->modifier = modifier;
        this->relation = relation;
    }

    void print(cnn::Dict& depreldict) const {
        cerr << head << "-" << depreldict.Convert(relation) << "->" << modifier;
    }

    void print() const {
        cerr << head << "->" << modifier;
    }

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

    void printTree(cnn::Dict& tokdict, cnn::Dict& depreldict) {
        cerr << "\nTree for sentence \"";
        for (unsigned int i = 1; i < numnodes; i++) {
            cerr << tokdict.Convert(sent[i]) << " ";
        }
        cerr << "\"" << endl;

        for (auto itr = children.begin(); itr != children.end(); ++itr) {
            unsigned node = itr->first;
            for (unsigned child : get_children(node)) {
                cerr << node << "-" << depreldict.Convert(deprels[child])
                        << "->" << child << endl;
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

    vector<unsigned> get_children(unsigned& node) {
        if (children.find(node) == children.end()) {
            return vector<unsigned>();
        } else {
            return children[node];
        }

    }

    vector<DepEdge> get_all_edges() {
        vector<DepEdge> edges;
        for (unsigned node = 1; node < parents.size(); ++node) {
            edges.push_back(DepEdge(parents[node], node, deprels[node]));
        }
        return edges;
    }

private:
    map<DepEdge, vector<DepEdge>*> neighbors;

    void set_children_and_root() {
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

    void set_leaves() {
        for (unsigned node = 0; node < numnodes; ++node) {
            if (get_children(node).size() == 0)
                leaves.insert(node);
        }
    }

    void set_dfo() {
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

    void set_dfo_msgs() { // does not include root -> dummyroot and dummyroot -> root
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

    void set_msg_neighbors() {
        for (unsigned edgnum = 0; edgnum < dfo_msgs.size(); ++edgnum) {
            DepEdge edge = dfo_msgs[edgnum];
            vector<DepEdge>* neighbor_list = new vector<DepEdge>();

            neighbors.insert(make_pair(edge, neighbor_list));
            msg_nbrs.insert(
                    pair<unsigned, vector<unsigned>>(edgnum,
                            vector<unsigned>()));

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

    void set_incoming_messages() {
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
}
;

