using namespace std;

const unsigned int DUMMY_ROOT = 0;
const string DUMMY_ROOT_STR = "ROOT";

struct DepTree {
    unsigned numnodes;
    unsigned root;
    vector<unsigned> parents;
    vector<unsigned> sent;
    vector<unsigned> deprels;
    set<unsigned> leaves;

    vector<unsigned> dfo; // depth-first ordering of the nodes
    map<unsigned, vector<unsigned>> children;

    explicit DepTree(vector<unsigned> parents, vector<unsigned> deprels,
            vector<unsigned> sent) {
        this->numnodes = parents.size(); // = (length of the sentence + 1) to accommodate root
        this->parents = parents;
        this->sent = sent;
        this->deprels = deprels;

        set_children_and_root();
        set_leaves();
        set_dfo();
    }

    void printTree(cnn::Dict& tokdict, cnn::Dict& depreldict) {
        cerr << "Tree for sentence \"";
        for (unsigned int i = 0; i < numnodes; i++) {
            cerr << tokdict.Convert(sent[i]) << " ";
        }
        cerr << "\"" << endl;

        for (unsigned int i = 0; i < numnodes; i++) {
            cerr << i << "<-" << depreldict.Convert(deprels[i]) << "-"
                    << parents[i] << endl;
        }
        cerr << "Leaves: ";
        for (unsigned leaf : leaves)
            cerr << leaf << " ";
        cerr << endl;
        cerr << "Depth-first Ordering:" << endl;
        for (unsigned node : dfo) {
            cerr << node << "->";
        }
        cerr << endl;
    }

    vector<unsigned> get_children(unsigned node) {
        vector<unsigned> clist;
        if (children.find(node) == children.end()) {
            return clist;
        } else {
            return children[node];
        }
    }

private:

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
        // TODO: should we maintain root in dfo? No
        assert(dfo.back() == DUMMY_ROOT);
        dfo.pop_back();
    }
};
