from typing import List
from collections import Counter, defaultdict

from green_tsetlin.ruleset import RuleSet


class Node:
    def __init__(self, ta):
        self.ta = ta
        self.w = None
        self.childs = []

class TopographicalC:
    def __init__(self, rs: RuleSet, explanation="none", exclude_negative_clauses=False):       
        self.explanation = explanation
        self.exclude_negative_clauses = exclude_negative_clauses
        
        if self.exclude_negative_clauses is True:
            raise ValueError("Unsuported option, exclude_negative_clauses=True, Please set to Faluse to use this writer.")
        
        if self.explanation != "none":
            raise ValueError("Unsported explanation: '{}', please use a supported explanation.".format(explanation))
        
        self.rules : List[set] = None
        self.n_literals = rs.n_literals
        self.rules = [set(r) for r in rs.rules]
        self.weights = rs.weights
        self.n_classes = rs.n_classes
        
        self.rules_in_tree = set()
        self._root:Node = None
        
    def to_file(self, path_to_file:str):        
                
        self._generate_tree()                
        with open(path_to_file, "w") as fp:
            fp.write("#ifndef __TOPOGRAPHICAL_C_OUTPUT__H_\n")
            fp.write("#define __TOPOGRAPHICAL_C_OUTPUT__H_\n")
            fp.write("// Export from green_tsetlin:TopographicalC\n")
            fp.write("\n")
            self._write_header(fp)
            
            fp.write("\n")
            fp.write("// start clauses")
            fp.write("\n")
            
            self._export_tree2(fp, self._root, 0, [])
            
            self._write_find_output(fp)    
            fp.write("\n")
            fp.write("#endif // #ifndef __TOPOGRAPHICAL_C_OUTPUT__H_\n")
        
    
    def _write_header(self, fp):                
        fp.write("#include <inttypes.h>\n")
        fp.write("#include <string.h>\n")
        fp.write("\n")
        fp.write("#define NUM_CLASSES {}\n".format(self.n_classes))
        fp.write("#define DEFAULT_OUTPUT 0\n".format(0))
        
        fp.write("int predict_tm(uint8_t* x)\n")
        fp.write("{\n")
        fp.write("    static int16_t votes[NUM_CLASSES] = {0};\n")
        fp.write("    memset(votes, 0, sizeof(votes));\n")
        fp.write("\n")
        fp.write("\n")
        
    def _write_find_output(self, fp):
        default_output = 0
        fp.write("int output_class = DEFAULT_OUTPUT;\n")
        fp.write("int16_t most_votes = votes[DEFAULT_OUTPUT];\n")
        
        if default_output == 0:        
            fp.write("for(int i = 1; i < NUM_CLASSES; ++i)\n")
        else:
            fp.write("for(int i = 0; i < NUM_CLASSES; ++i)\n")
            
        fp.write("{\n")
        
        fp.write("    if(votes[i] > most_votes)\n")
        fp.write("    {\n")
        fp.write("        most_votes = votes[i];\n")
        fp.write("        output_class = i;\n")
        fp.write("    }\n")
        fp.write("")        
        fp.write("}\n")
        fp.write("    return output_class;\n")
        fp.write("}\n")
        
    def _generate_tree(self):               
        all_rules = set(range(len(self.rules)))
        self._root = self._subtree(None, all_rules, set())
        

    def _subtree(self, head:int, rules:set, path:set):          
        current_node = Node(head)
        
        rules = set(rules)
        while rules:
                        
            ta2r = defaultdict(set)
            for rule_k in rules:
                if self.rules[rule_k] == path:
                    self.rules_in_tree.add(rule_k)
                    current_node.w = self.weights[rule_k]
                    
                for ta in (self.rules[rule_k] - path):                
                    ta2r[ta].add(rule_k)
                                    
            if len(ta2r) < 1:
                return current_node                
                    
            next_head, next_rules = max(ta2r.items(), key=lambda r: len(r[1]))
            next_path = set(path)
            next_path.add(next_head)
            child_node = self._subtree(next_head, next_rules, next_path)
            current_node.childs.append(child_node)
            
            rules -= self.rules_in_tree
            
        return current_node        
    
    def print_tree(self, tree, d):
        print("-"*(d+1), "#", "ta:", tree.ta, "w:", tree.w)
        for c in tree.childs:
            self.print_tree(c, d+1)
        
        
    def _export_tree(self, fp, tree:Node, d):
        
        if tree.ta is not None:
            if tree.ta < self.n_literals:
                if_str = "if(x[{}] > 0){{\n".format(tree.ta)
            else:
                if_str = "if(x[{}] == 0){{\n".format(tree.ta - self.n_literals)
                    
            fp.write("  "*d)
            fp.write(if_str)
            
        if tree.w is not None:
            add_votes = " ".join(["votes[{}] += {};".format(k, w) for k, w in enumerate(tree.w)])
            fp.write("{}\n".format(add_votes))
            
        for c in tree.childs:
            self._export_tree(fp, c, d+1)
            
        if tree.ta is not None:
            fp.write("  "*d)
            fp.write("}\n")

    def _export_tree2(self, fp, tree:Node, d, if_conds:list):        
        if_conds = list(if_conds)
        if tree.ta is not None:
            if_conds.append(tree.ta)


        flush = False
        if tree.w is not None:
            if_list = []
            for ta_k in if_conds:
                if ta_k < self.n_literals:
                    if_list.append("x[{}] > 0".format(ta_k))
                else:
                    if_list.append("x[{}] == 0".format(ta_k - self.n_literals))
                    
            clause_statements = " && ".join(if_list)
            if_statment = "if({}){{".format(clause_statements)
            
            fp.write("  "*d)
            fp.write("{}\n".format(if_statment))
            if_conds = [] # reset

            add_votes = " ".join(["votes[{}] += {};".format(k, w) for k, w in enumerate(tree.w)])
            fp.write("{}\n".format(add_votes))

            flush = True

            d += 1

        
            
        for c in tree.childs:
            self._export_tree2(fp, c, d, if_conds)
            
        if tree.ta is not None and flush:
            fp.write("  "*d)
            fp.write("}\n")
        
        

class MockRuleset:
    def __init__(self):
        self.rules = [[1, 2, 3], [1, 2], [1,3] ,[3, 4]]
        self.weights = [[-21, 21], [24, -21], [-26, 28], [-26, 28]]
        self.n_literals = 8
        self.n_classes = 2
    
            
if __name__ == "__main__":

    rs = MockRuleset()
    w = TopographicalC(rs)
    
    w.to_file("tmp.h")
    
