"""Bounded algebraic normalization; no fitted law templates or observation matching."""
from __future__ import annotations
import ast
import math

_NAMES = {'t', 'x', 'y', 'vx', 'vy', 'mass', 'charge'}
_LIMIT = 64


def _poly(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if not math.isfinite(node.value): raise ValueError('nonfinite')
        return {(): float(node.value)}
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return {(node.id,): 1.0}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        return {k: sign*v for k,v in _poly(node.operand).items()}
    if not isinstance(node, ast.BinOp): raise ValueError('not_polynomial')
    if isinstance(node.op, (ast.Add, ast.Sub)):
        result = _poly(node.left)
        for k,v in _poly(node.right).items():
            result[k] = result.get(k, 0.) + (-v if isinstance(node.op, ast.Sub) else v)
    elif isinstance(node.op, ast.Mult):
        result = {}
        left, right = _poly(node.left), _poly(node.right)
        if len(left)*len(right)>_LIMIT*2: raise ValueError('expansion_limit')
        for a, x in left.items():
            for b, y in right.items():
                k = tuple(sorted(a+b))
                if len(k)>6: raise ValueError('degree_limit')
                result[k] = result.get(k, 0.) + x*y
    elif isinstance(node.op, ast.Div):
        denominator = _poly(node.right)
        if set(denominator)!={()} or denominator[()]==0: raise ValueError('not_polynomial')
        result = {k:v/denominator[()] for k,v in _poly(node.left).items()}
    elif isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and type(node.right.value) is int and 0<=node.right.value<=6:
        result = {(): 1.}
        for _ in range(node.right.value):
            # AST is bounded before entry, and expansion is bounded at every step.
            left = ast.parse(_source(result), mode='eval').body
            result = _poly(ast.BinOp(left=left,op=ast.Mult(),right=node.left))
    else: raise ValueError('not_polynomial')
    if len(result)>_LIMIT or any(not math.isfinite(v) for v in result.values()): raise ValueError('expansion_limit')
    return {k:v for k,v in result.items() if v!=0}


def _source(poly):
    terms = []
    for monomial, coefficient in sorted(poly.items(),key=lambda item:(len(item[0]),item[0])):
        factors = [name if monomial.count(name)==1 else f'{name}**{monomial.count(name)}' for name in sorted(set(monomial))]
        if not factors: term = repr(coefficient)
        else:
            product = '*'.join(factors)
            term = product if coefficient==1 else f'-({product})' if coefficient==-1 else f'({coefficient!r})*({product})'
        terms.append(term)
    return ' + '.join(terms) or '0'


def normalize_polynomial(source):
    """Expand only bounded polynomials; never cancel denominators or alter domains."""
    try:
        node = ast.parse(source,mode='eval').body
        if sum(1 for _ in ast.walk(node))>256: return source
        expanded = _source(_poly(node))
        return expanded if len(expanded)<=4096 else source
    except (ValueError,TypeError,OverflowError,RecursionError,SyntaxError):
        return source


def expression_key(source):
    """Algebraic keys with 12-significant-digit coefficient roundoff, never sample equality.

    Polynomial factorization and commutative ordering are recognized. General
    transcendental identities are deliberately left alone to preserve domains.
    """
    def key(node):
        try:
            return ('poly',tuple((k,format(v,'.12g')) for k,v in sorted(_poly(node).items())))
        except (ValueError,TypeError,OverflowError,RecursionError): pass
        if isinstance(node,ast.BinOp):
            if isinstance(node.op,(ast.Add,ast.Mult)):
                children=[]
                def flatten(n):
                    if isinstance(n,ast.BinOp) and type(n.op) is type(node.op): flatten(n.left);flatten(n.right)
                    else: children.append(key(n))
                flatten(node)
                return (type(node.op).__name__,tuple(sorted(children,key=repr)))
            return (type(node.op).__name__,key(node.left),key(node.right))
        if isinstance(node,ast.UnaryOp): return (type(node.op).__name__,key(node.operand))
        if isinstance(node,ast.Call): return ('call',ast.dump(node.func),tuple(key(a) for a in node.args))
        return ast.dump(node)
    return key(ast.parse(source,mode='eval').body)


def candidate_key(candidate):
    values={p['name']:p['initial'] for p in candidate['parameter_spec']}
    class Values(ast.NodeTransformer):
        def visit_Name(self,node):
            return ast.copy_location(ast.Constant(value=values[node.id]),node) if node.id in values else node
    return tuple(expression_key(ast.unparse(Values().visit(ast.parse(candidate['law_spec'][axis],mode='eval')))) for axis in ('ax','ay'))


def candidate_structure_key(candidate):
    """Same parameterized formula/share pattern, irrespective of optimized values.

    Coefficient refits can differ in their last digits. Those are alternative
    fits of one hypothesis, not separate formula slots. Structural literals and
    sharing relationships remain part of the key.
    """
    parameters={p['name'] for p in candidate['parameter_spec']}
    mapping={}
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id in parameters:
                mapping.setdefault(node.id,'parameter_'+str(len(mapping)))
                return ast.copy_location(ast.Name(id=mapping[node.id],ctx=ast.Load()),node)
            return node
    return tuple(expression_key(ast.unparse(Rename().visit(ast.parse(candidate['law_spec'][axis],mode='eval')))) for axis in ('ax','ay'))
