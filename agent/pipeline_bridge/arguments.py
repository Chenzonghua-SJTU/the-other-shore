"""Lossless JSON-container decoding and explicit property-access DSL aliases."""
import ast,copy,json,re

def normalize(args,schema,allowed_properties=()):
    changes=[]
    def visit(value,node,path):
        expected=node.get('type')
        if expected in ('array','object') and isinstance(value,str):
            try:decoded=json.loads(value)
            except ValueError:decoded=None
            if (expected=='array' and isinstance(decoded,list)) or (expected=='object' and isinstance(decoded,dict)):
                changes.append({'path':path,'kind':'decode_json_container'});value=decoded
        if isinstance(value,dict):
            return {k:visit(v,node.get('properties',{}).get(k,{}),path+[k]) for k,v in value.items()}
        if isinstance(value,list):return [visit(v,node.get('items',{}),path+[i]) for i,v in enumerate(value)]
        return value
    result=visit(copy.deepcopy(args),schema,[])
    parameters={p.get('name') for p in result.get('parameter_spec',[]) if isinstance(p,dict)}
    properties=set(allowed_properties)-parameters
    class PropertyAlias(ast.NodeTransformer):
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if isinstance(node.op,ast.Pow):
                exponent=node.right
                sign=-1 if isinstance(exponent,ast.UnaryOp) and isinstance(exponent.op,ast.USub) else 1
                value=exponent.operand if sign==-1 else exponent
                if isinstance(value,ast.Constant) and type(value.value) in (int,float) and sign*value.value in (-1.5,-.5,.5,1.5):
                    power=int(2*sign*value.value)
                    root=ast.Call(func=ast.Name(id='sqrt',ctx=ast.Load()),args=[node.left],keywords=[])
                    return ast.copy_location(root if power==1 else ast.BinOp(left=root,op=ast.Pow(),right=ast.Constant(power)),node)
            return node
        def visit_Name(self,node):
            if node.id in properties:
                return ast.copy_location(ast.Call(func=ast.Name(id='q',ctx=ast.Load()),args=[ast.Constant(node.id)],keywords=[]),node)
            return node
    for key in ('law_spec','background_law_spec'):
        if not isinstance(result.get(key),dict):continue
        for axis in ('ax','ay'):
            value=result[key].get(axis)
            if not isinstance(value,str):continue
            original=value
            value=re.sub(r'''q\(\s*\\+(['"])([A-Za-z][A-Za-z0-9_]*)\\+\1\s*\)''',lambda m:'q('+repr(m.group(2))+')',value)
            try:tree=ast.parse(value,mode='eval')
            except SyntaxError:continue
            names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)} & properties
            transformed=PropertyAlias().visit(copy.deepcopy(tree))
            if names or value!=original or ast.dump(tree)!=ast.dump(transformed):
                converted=ast.unparse(ast.fix_missing_locations(transformed)).strip()
                result[key][axis]=converted
                changes.append({'path':[key,axis],'kind':'equivalent_dsl_serialization','properties':sorted(names),'original':original,'normalized':converted})
    return result,changes
