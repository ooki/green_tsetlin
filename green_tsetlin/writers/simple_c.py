from green_tsetlin.ruleset import RuleSet




class SimpleC:
    def __init__(self, rs: RuleSet, explanation="none", exclude_negative_clauses=False):
        self.rs = rs         
        self.explanation = explanation
        self.exclude_negative_clauses = exclude_negative_clauses
        
        if self.exclude_negative_clauses is True:
            raise ValueError("Unsuported option, exclude_negative_clauses=True, Please set to Faluse to use this writer.")
        
        if self.explanation != "none":
            raise ValueError("Unsported explanation: '{}', please use a supported explanation.".format(explanation))
        
        
    def to_file(self, path_to_file:str):        
        
        with open(path_to_file, "w") as fp:
            fp.write("#ifndef __SIMPLE_C_OUTPUT__H_\n")
            fp.write("#define __SIMPLE_C_OUTPUT__H_\n")
            fp.write("// Export from green_tsetlin:SimpleC\n")
            fp.write("\n")
            self._write_header(fp)
            
            fp.write("\n")
            fp.write("// start clauses")
            fp.write("\n")
            
            for k in range(len(self.rs.rules)):
                self._write_rule(fp, k)
            
            self._write_find_output(fp)    
            fp.write("\n")
            fp.write("#endif // #ifndef __SIMPLE_C_OUTPUT__H_\n")
                    
    
    def _write_header(self, fp):                
        fp.write("#include <inttypes.h>\n")
        fp.write("#include <string.h>\n")
        fp.write("\n")
        fp.write("#define NUM_CLASSES {}\n".format(self.rs.n_classes))
        fp.write("#define DEFAULT_OUTPUT 0\n".format(0))
        
        fp.write("int predict_tm(uint8_t* x)\n")
        fp.write("{\n")
        fp.write("    static int16_t votes[NUM_CLASSES] = {0};\n")
        fp.write("    memset(votes, 0, sizeof(votes));\n")
        fp.write("\n")
        fp.write("\n")

        
    def _write_rule(self, fp, rule_k):
        
        weights = self.rs.weights[rule_k]
        rule = self.rs.rules[rule_k]
                
        
        if_list = []
        for ta_k in rule:
            if ta_k < self.rs.n_literals:
                if_list.append("x[{}] > 0".format(ta_k))
            else:
                if_list.append("x[{}] == 0".format(ta_k - self.rs.n_literals))
                
        clause_statements = " && ".join(if_list)
        if_statment = "if({}){{".format(clause_statements)
        add_votes = " ".join(["votes[{}] += {};".format(k, w) for k, w in enumerate(weights)])
        
        
        fp.write("{}\n".format(if_statment))
        fp.write("{}\n".format(add_votes))
        fp.write("}\n")        
        
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
    
                
        
class MockRuleset:
    def __init__(self):
        self.rules = [[0], [0,1], [2,3]]
        self.weights = [[-1, 2], [-3, 4], [7, -8]]
        self.n_literals = 2
        self.n_classes = 2
         
if __name__ == "__main__":
    

    rs = MockRuleset()
    w = Writer(rs)
    w.to_file("./generator/out.h")
    
    print("<done>")
        
        
        
        
        
        
        
