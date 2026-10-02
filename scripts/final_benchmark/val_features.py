pass
import ast

def infer_features(spec, record):
    code = spec.get('python_code', '')
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = ast.parse('pass')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    methods = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]
    mapping = {'plot': 'line', 'bar': 'bar', 'barh': 'bar', 'scatter': 'scatter', 'fill_between': 'area', 'fill_betweenx': 'area', 'pie': 'pie', 'hist': 'histogram', 'boxplot': 'box', 'violinplot': 'violin', 'imshow': 'heatmap_or_image', 'pcolormesh': 'heatmap', 'contour': 'contour', 'contourf': 'contour', 'errorbar': 'error_bar', 'hexbin': 'hexbin'}
    types = sorted({mapping[m] for m in methods if m in mapping}) if spec['kind'] == 'chart' else []
    tags = record.get('task_tags', [])
    hard = {'multi_step', 'approximate_reading'} & set(tags)
    medium = {'comparison', 'counting', 'extrema_ranking', 'difference', 'sum_aggregation', 'ratio_percentage', 'mean_median', 'conditional_filter', 'intersection_threshold'} & set(tags)
    difficulty = {'label': 'hard' if hard else 'medium' if medium else 'easy', 'reasoning_steps': None, 'rationale': 'Provisional rubric applied to existing task tags; visual/API annotation pending.'}
    return {'chart_types': types, 'difficulty': difficulty, 'features': {'annotation_source': 'renderer AST + existing task tags; provisional and may conflate decorative marks', 'recovery_description': spec.get('recovery', {}), 'drawing_methods': sorted(set(methods) & set(mapping)), 'reasoning_operations': tags, 'axis_scales': ['log'] if 'log' in code else [], 'has_error_bands': None, 'panel_count': spec.get('recovery', {}).get('panel_count'), 'label_count': len(spec.get('labels', {}))}}
