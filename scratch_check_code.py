import ast
import sys

def check_file(filepath):
    print(f"Parsing AST for {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        print("AST parsed successfully!")
    except SyntaxError as e:
        print(f"SyntaxError in {filepath}: {e}")
        return

    # Let's collect all defined names
    defined_names = set()
    used_names = set()
    imported_names = set()

    # Builtins
    import builtins
    builtin_names = set(dir(builtins))

    class Analyzer(ast.NodeVisitor):
        def __init__(self):
            self.current_scope = [{}] # Stack of scopes (dicts of name -> line)
            self.globals = set()
            self.errors = []

        def visit_Import(self, node):
            for alias in node.names:
                name = alias.asname or alias.name
                self.current_scope[-1][name] = node.lineno
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                name = alias.asname or alias.name
                self.current_scope[-1][name] = node.lineno
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            self.current_scope[-1][node.name] = node.lineno
            # New scope for function arguments and body
            func_scope = {}
            for arg in node.args.args:
                func_scope[arg.arg] = node.lineno
            if node.args.vararg:
                func_scope[node.args.vararg.arg] = node.lineno
            if node.args.kwarg:
                func_scope[node.args.kwarg.arg] = node.lineno
            for arg in node.args.kwonlyargs:
                func_scope[arg.arg] = node.lineno
            
            self.current_scope.append(func_scope)
            
            # Visit decorators before entering function scope
            for dec in node.decorator_list:
                self.visit(dec)
            
            # Visit body
            for stmt in node.body:
                self.visit(stmt)
                
            self.current_scope.pop()

        def visit_ClassDef(self, node):
            self.current_scope[-1][node.name] = node.lineno
            class_scope = {}
            self.current_scope.append(class_scope)
            for dec in node.decorator_list:
                self.visit(dec)
            for base in node.bases:
                self.visit(base)
            for stmt in node.body:
                self.visit(stmt)
            self.current_scope.pop()

        def visit_Global(self, node):
            for name in node.names:
                self.globals.add(name)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                # Write to variable
                if len(self.current_scope) == 1 or node.id in self.globals:
                    self.current_scope[0][node.id] = node.lineno
                else:
                    self.current_scope[-1][node.id] = node.lineno
            elif isinstance(node.ctx, ast.Load):
                # Check if name is defined in any active scope
                found = False
                for scope in reversed(self.current_scope):
                    if node.id in scope:
                        found = True
                        break
                if not found and node.id not in builtin_names:
                    self.errors.append((node.lineno, f"Undefined name '{node.id}'"))
            self.generic_visit(node)

    analyzer = Analyzer()
    analyzer.visit(tree)

    if analyzer.errors:
        print(f"Found {len(analyzer.errors)} potential undefined name errors:")
        from collections import defaultdict
        grouped = defaultdict(list)
        for line, err in analyzer.errors:
            grouped[err].append(line)
        
        for err, lines in sorted(grouped.items()):
            print(f"- {err} at lines: {lines[:10]} ... total {len(lines)}")
    else:
        print("No undefined name errors found via simple AST traversal!")

if __name__ == "__main__":
    check_file("app.py")
