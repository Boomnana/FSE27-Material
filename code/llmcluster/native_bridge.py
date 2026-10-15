"""Explicit import boundary around the pinned, unmodified official class.

Only the upstream ClusterManager AST is compiled, with its method AST unchanged.
Its imports of OpenAI and Gold-dependent evaluation are not executed. The three
injected boundaries are registered: complete report loading, query transport,
and UTF-8 output. No LLM response is compiled, evaluated, or executed.
"""
import ast
import builtins
import copy
from pathlib import Path
import re
from typing import List, Tuple

from common import HERE, require, save


def load_native_class(query, reports, run, upstream=None):
    upstream = Path(upstream or HERE / 'upstream')
    run = Path(run).resolve()
    prompt_namespace = {}
    exec(compile((upstream / 'prompt.py').read_text(encoding='utf-8'),
                 str(upstream / 'prompt.py'), 'exec'), prompt_namespace)
    source = (upstream / 'main.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ClusterManager']
    require(len(classes) == 1, 'Pinned upstream class not found')

    def load_reports(path):
        require(Path(path).resolve().parent == run, 'Unexpected native input path')
        return [{'index': str(r['native_id']), 'description': r['description'],
                 'img_url': '', 'bug_type': 'UNUSED_ONLINE'} for r in reports]

    def utf8_open(path, mode='r', *args, **kwargs):
        target = Path(path).resolve()
        require(target.is_relative_to(run), 'Native file access escapes job directory')
        if 'b' not in mode:
            kwargs.setdefault('encoding', 'utf-8')
        return builtins.open(target, mode, *args, **kwargs)

    namespace = {'List': List, 'Tuple': Tuple, 'query': query,
                 'cluster_sys_prompt': prompt_namespace['cluster_sys_prompt'],
                 'load_reports': load_reports, 'open': utf8_open}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(upstream / 'main.py'), 'exec'), namespace)
    native = namespace['ClusterManager']

    class RecordedNative(native):
        def clusters_from_response(self):
            
            
            
            text = (run / 'response' / f'app{self.app}-cluster.txt').read_text(encoding='utf-8')
            head = text.rfind('CURRENT CLUSTERS:') + len('CURRENT CLUSTERS:')
            tail = text.find('Token Usage:', head, len(text))
            selected = text[head:len(text) if tail == -1 else tail].strip()
            for chunk in re.findall(r'\{([^{}]*)\}', selected):
                for piece in chunk.split(','):
                    piece = piece.strip()
                    try:
                        value = int(piece)
                        require(1 <= value <= len(reports), 'Native response ID outside population')
                    except ValueError:
                        if '-' not in piece:
                            raise
                        first, last = map(int, piece.split('-', 1))
                        require(1 <= first <= last <= len(reports), 'Native response range outside population')
            return super().clusters_from_response()

        def make_correction(self, clusters):
            save(run / 'native_pre_correction.json', copy.deepcopy(clusters))
            corrected = super().make_correction(clusters)
            save(run / 'native_post_correction.json', copy.deepcopy(corrected))
            return corrected

    return RecordedNative


def validate_partition(partition, n):
    require(isinstance(partition, list) and all(len(group) > 0 for group in partition), 'Empty or malformed native partition')
    flat = [item for group in partition for item in group]
    require(all(type(item) is int for item in flat), 'Noninteger native ID')
    require(len(flat) == n and sorted(flat) == list(range(1, n + 1)), 'Native correction did not produce complete unique population')
    return [list(group) for group in partition]
