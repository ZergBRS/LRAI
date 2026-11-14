"""ЛР3. Нечёткий вывод; база знаний читается из отдельного текстового файла."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from html import escape
from pathlib import Path

BASE = Path(__file__).resolve().parent / 'база_знаний.txt'
IMPLICATIONS = {
    'mamdani': lambda a, b: min(a, b),
    'larsen': lambda a, b: a * b,
    'kleene_dienes': lambda a, b: max(1 - a, b),
    'lukasiewicz': lambda a, b: min(1.0, 1 - a + b),
}
AGGREGATIONS = ('max', 'prob_sum', 'bounded_sum', 'min')
METHODS = ('max_min', 'max_product', 'levels')


def number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('Нужно конечное число')
    return float(x)


def membership(x, term):
    points = term['points']
    a, d = points[0], points[-1]
    b, c = (points[1], points[1]) if term['type'] == 'triangle' else points[1:3]
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


def aggregate(values, method):
    values = list(values)
    if not values:
        return 0.0
    if method == 'max':
        return max(values)
    if method == 'min':
        return min(values)
    if method == 'prob_sum':
        return 1 - math.prod(1 - v for v in values)
    if method == 'bounded_sum':
        return min(1.0, sum(values))
    raise ValueError('Неизвестная агрегация')


def conjunction(values, method):
    if method == 'min':
        return min(values)
    if method == 'product':
        return math.prod(values)
    raise ValueError('Неизвестный оператор И')


def universe(domain, step):
    lo, hi = domain
    if not math.isfinite(step) or step <= 0 or step > hi - lo:
        raise ValueError('Шаг должен быть положительным и не больше диапазона')
    n = math.ceil((hi - lo) / step)
    if n > 2000:
        raise ValueError('Слишком мелкий шаг: допускается не более 2001 узла')
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def observed_membership(x, value, width):
    return max(0.0,1-abs(x-value)/width) if width else float(x == value)


def observation_grid(variable, value, width, step):
    lo,hi=variable['range']
    nodes=universe(variable['range'],step)+[value,max(lo,value-width),min(hi,value+width)]
    nodes+= [x for term in variable['terms'].values() for x in term['points']]
    grid=sorted(set(nodes))
    # На каждом линейном участке максимум min(A', A_i) находится
    # на границе или в точке пересечения A' с A_i.
    if width:
        for term in variable['terms'].values():
            for x0,x1 in zip(grid,grid[1:]):
                d0=observed_membership(x0,value,width)-membership(x0,term)
                d1=observed_membership(x1,value,width)-membership(x1,term)
                if d0*d1 < 0:
                    nodes.append(x0+(x1-x0)*d0/(d0-d1))
    return sorted(set(nodes))


def centroid(xs, ys):
    """Точные интегралы для ломаной, соединяющей рассчитанные узлы."""
    area = moment = 0.0
    for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:]):
        h = x1 - x0
        area += h * (y0 + y1) / 2
        moment += h * ((2*x0 + x1)*y0 + (x0 + 2*x1)*y1) / 6
    return (moment / area if area > 1e-12 else None), area


def compose(observed, relation, method):
    """max-min или max-product композиция с последовательностью строк R."""
    if method not in ('max_min','max_product'):
        raise ValueError('Неизвестная композиция')
    curve = None
    for a, row in zip(observed, relation):
        if curve is None:
            curve = [0.0] * len(row)
        if len(row) != len(curve):
            raise ValueError('Строки отношения должны иметь одинаковую длину')
        for j, r in enumerate(row):
            value = min(a, r) if method == 'max_min' else a*r
            curve[j] = max(curve[j], value)
    return curve or []


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Повторяющееся имя в JSON: {key}')
        result[key] = value
    return result


def parse_text_base(text):
    """Читаемые объявления MODEL, INPUT, OUTPUT, TERM и правила ЕСЛИ ... ТО."""
    data = {'models': {}}
    model = variable = None
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        try:
            parts = shlex.split(line)
            kind = parts[0]
            if kind == 'MODEL':
                _, key, title = parts
                if key in data['models']:
                    raise ValueError('Повтор модели')
                model = {'name': title, 'inputs': {}, 'rules': []}
                data['models'][key] = model
                variable = None
            elif model is None:
                raise ValueError('Сначала объявите MODEL')
            elif kind == 'INPUT':
                _, name, low, high, default, unit, label = parts
                if name in model['inputs']:
                    raise ValueError('Повтор входной переменной')
                variable = {'range': [float(low), float(high)], 'default': float(default),
                            'unit': unit, 'label': label, 'terms': {}}
                model['inputs'][name] = variable
            elif kind == 'OUTPUT':
                _, name, low, high, unit, label = parts
                if 'output' in model:
                    raise ValueError('Повтор выходной переменной')
                model['output_name'] = name
                variable = {'range': [float(low), float(high)], 'unit': unit,
                            'label': label, 'terms': {}}
                model['output'] = variable
            elif kind == 'TERM':
                if variable is None:
                    raise ValueError('Сначала объявите переменную')
                _, name, shape, *points = parts
                if name in variable['terms']:
                    raise ValueError('Повтор терма')
                variable['terms'][name] = {'type': shape, 'points': list(map(float, points))}
            elif kind == 'ЕСЛИ':
                conditions, conclusion = line[5:].split(' ТО ')
                pairs = [condition.strip().split('=') for condition in conditions.split(' И ')]
                names = [name.strip() for name, term in pairs]
                if len(set(names)) != len(names):
                    raise ValueError('Повтор переменной в условии')
                name, term = map(str.strip, conclusion.split('='))
                if name != model.get('output_name'):
                    raise ValueError('Неизвестная выходная переменная')
                model['rules'].append({'if': {n.strip(): t.strip() for n,t in pairs}, 'then': term})
            else:
                raise ValueError('Неизвестная команда базы')
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'Строка {line_no}: {error}') from error
    return data


def load_base(path=BASE):
    text = Path(path).read_text(encoding='utf-8-sig')
    data = json.loads(text, object_pairs_hook=unique_object) if text.lstrip().startswith('{') else parse_text_base(text)
    if not isinstance(data, dict) or not isinstance(data.get('models'), dict) or not data['models']:
        raise ValueError('В базе нужен непустой объект models')
    for name, model in data['models'].items():
        if not isinstance(model.get('inputs'), dict) or not model['inputs']:
            raise ValueError(f'{name}: нужны входные переменные')
        for variable in [*model['inputs'].values(), model['output']]:
            lo, hi = map(number, variable['range'])
            if lo >= hi or not variable['terms']:
                raise ValueError('Неверный диапазон или пустые термы')
            for term in variable['terms'].values():
                size = {'triangle': 3, 'trapezoid': 4}.get(term['type'])
                points = list(map(number, term['points']))
                if size is None or len(points) != size or points != sorted(points) or points[0] == points[-1]:
                    raise ValueError('Неверные параметры функции принадлежности')
                if not lo <= points[0] <= points[-1] <= hi:
                    raise ValueError('Опорные точки терма должны лежать в диапазоне')
        if not isinstance(model['rules'], list) or not model['rules']:
            raise ValueError('База правил пуста')
        for rule in model['rules']:
            if not isinstance(rule['if'], dict) or not rule['if']:
                raise ValueError('У правила нужны условия')
            for var, term in rule['if'].items():
                if var not in model['inputs'] or term not in model['inputs'][var]['terms']:
                    raise ValueError(f'Неизвестный вход или терм: {var}={term}')
            if rule['then'] not in model['output']['terms']:
                raise ValueError('Неизвестный выходной терм')
    return data


def infer(model, inputs, method='levels', implication='mamdani', aggregation='max',
          and_method='min', width=0.0, step=0.5):
    if method not in METHODS or implication not in IMPLICATIONS or aggregation not in AGGREGATIONS:
        raise ValueError('Неизвестный метод вывода, импликация или агрегация')
    if and_method not in ('min', 'product'):
        raise ValueError('Оператор И: min или product')
    width, step = number(width), number(step)
    if width < 0:
        raise ValueError('Ширина нечёткого наблюдения не может быть отрицательной')
    if set(inputs) != set(model['inputs']):
        raise ValueError('Нужно задать все входы выбранной модели')
    for name, variable in model['inputs'].items():
        value = number(inputs[name])
        if not variable['range'][0] <= value <= variable['range'][1]:
            raise ValueError(f'{name}: значение вне диапазона {variable["range"]}')
    if len(inputs) != 1 and (method != 'levels' or width):
        raise ValueError('Для нескольких входов используется levels с чёткими наблюдениями')
    output = model['output']
    ys = universe(output['range'], step)
    terms = {name: [membership(y, term) for y in ys] for name, term in output['terms'].items()}
    I = IMPLICATIONS[implication]
    rules = model['rules']
    input_grid = observed = None
    if len(inputs) == 1:
        name = next(iter(inputs))
        value = inputs[name]
        input_grid = observation_grid(model['inputs'][name],value,width,step)
        observed = [observed_membership(x,value,width) for x in input_grid]

    alphas = []
    for rule in rules:
        if len(inputs) == 1 and width:
            name, term = next(iter(rule['if'].items()))
            mf = model['inputs'][name]['terms'][term]
            alpha = max(min(a, membership(x, mf)) for x, a in zip(input_grid, observed))
        else:
            alpha = conjunction([membership(inputs[n], model['inputs'][n]['terms'][t])
                                 for n, t in rule['if'].items()], and_method)
        alphas.append(alpha)

    if method == 'levels':
        # Нулевые уровни тоже участвуют: для логических импликаций I(0,b)=1.
        curve = [aggregate((I(a, terms[r['then']][j]) for a, r in zip(alphas, rules)), aggregation)
                 for j in range(len(ys))]
    else:
        name = next(iter(inputs))
        support = [(x,a) for x,a in zip(input_grid, observed) if a > 0]
        def relation_rows():
            for x,_ in support:
                antecedents = [membership(x, model['inputs'][name]['terms'][r['if'][name]]) for r in rules]
                yield [aggregate((I(a, terms[r['then']][j]) for a,r in zip(antecedents,rules)), aggregation)
                       for j in range(len(ys))]
        curve = compose([a for _,a in support],relation_rows(),method)
    value, area = centroid(ys, curve)
    return {'model': model['name'], 'inputs': inputs, 'method': method, 'implication': implication,
            'aggregation': aggregation, 'and': and_method, 'width': width, 'step': step,
            'value': value, 'area': area, 'alpha': alphas, 'grid': ys, 'membership': curve}


def value_text(value):
    return 'не определён (пустое множество)' if value is None else f'{value:.4f}'


def describe_rule(rule):
    return 'ЕСЛИ ' + ' И '.join(f'{k}={v}' for k, v in rule['if'].items()) + ' ТО Приоритет=' + rule['then']


def show_result(model, result):
    print('\nМодель:', result['model'])
    print('Входы:', ', '.join(f'{k}={v:g}' for k, v in result['inputs'].items()))
    print(f"Метод: {result['method']}; импликация: {result['implication']}; агрегация: {result['aggregation']}; И: {result['and']}")
    print(f"Полуширина наблюдения: {result['width']:g}; шаг сетки: {result['step']:g}")
    print('Ненулевые уровни истинности предпосылок:')
    for i, (a, rule) in enumerate(zip(result['alpha'], model['rules']), 1):
        if a > 1e-12:
            print(f'  {i}: α={a:.6f}; {describe_rule(rule)}')
    if result['implication'] in ('kleene_dienes','lukasiewicz'):
        print('Учтены также нулевые уровни: для этой импликации I(0,b)=1.')
    print('Площадь выходного множества:', f"{result['area']:.6f}")
    print('Приоритет диагностики:', value_text(result['value']), 'баллов из 100')
    if result['value'] is not None and max(result['membership'])-min(result['membership']) < 1e-10:
        print('Выходное множество постоянно: результат не различает уровни приоритета.')
    print('Баллы приоритета не являются вероятностью отказа.')


def compare(model, inputs, width=0, aggregation='max', and_method='min', step=0.5):
    methods = METHODS if len(inputs) == 1 else ('levels',)
    return [infer(model, inputs, method, implication, aggregation, and_method, width, step)
            for implication in IMPLICATIONS for method in methods]


def show_comparison(results):
    print('\nИмпликация        Метод          Приоритет')
    for r in results:
        print(f"{r['implication']:<18}{r['method']:<15}{value_text(r['value'])}")


def ask_float(prompt, default):
    text = input(f'{prompt} [{default}]: ').strip().replace(',', '.')
    value = float(text) if text else float(default)
    if not math.isfinite(value):
        raise ValueError('Введите конечное число')
    return value


def choose(label, options, default=0):
    print(label, ', '.join(f'{i+1} — {v}' for i,v in enumerate(options)))
    text = input(f'Выбор [{default+1}]: ').strip()
    index = int(text)-1 if text else default
    if not 0 <= index < len(options):
        raise ValueError('Нет такого варианта')
    return options[index]


def ask_inputs(model):
    return {name: ask_float(f"{v['label']}, {v['unit']}, диапазон {v['range']}", v['default'])
            for name,v in model['inputs'].items()}


def save_graph(results, path):
    """График SVG открывается в браузере; сторонние пакеты не нужны."""
    colors = ['#165aa7', '#c13b31', '#248145', '#8d48aa', '#be730c', '#00858e']
    left, top, width, height = 70, 60, 760, 300
    low, high = results[0]['grid'][0], results[0]['grid'][-1]
    def point(x, y):
        return f'{left + width*(x-low)/(high-low):.2f},{top + height*(1-y):.2f}'
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="620" viewBox="0 0 920 620">',
           '<rect width="920" height="620" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#222">',
           '<text x="70" y="28" font-size="20">Приоритет диагностики приложения</text>']
    for i in range(6):
        y = top+height*i/5
        svg.append(f'<path d="M70 {y} H830" stroke="#ddd"/><text x="34" y="{y+5}" font-size="14">{1-i/5:.1f}</text>')
    for i in range(6):
        x = left+width*i/5
        svg.append(f'<text x="{x}" y="385" text-anchor="middle" font-size="14">{low+(high-low)*i/5:g}</text>')
    svg.append('<text x="70" y="414" font-size="16">Приоритет, баллы от 0 до 100; по вертикали — принадлежность</text>')
    for i,r in enumerate(results):
        color=colors[i % len(colors)]
        points=' '.join(point(x,y) for x,y in zip(r['grid'],r['membership']))
        dash=' stroke-dasharray="7 4"' if i % 2 else ''
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        label=escape(f"{r['method']} / {r['implication']} / {r['aggregation']}: {value_text(r['value'])}")
        svg.append(f'<text x="70" y="{450+25*i}" fill="{color}" font-size="15">{label}</text>')
    svg.append('</g></svg>')
    Path(path).write_text(''.join(svg), encoding='utf-8')


def run_demo(model, inputs, width=0):
    # Сравниваем механизмы на одних входах и с одними параметрами базы.
    settings = [('max_min','mamdani','max'), ('max_product','mamdani','max')] if len(inputs)==1 else []
    settings += [('levels','mamdani','max'), ('levels','larsen','max'),
                 ('levels','mamdani','prob_sum'), ('levels','mamdani','bounded_sum')]
    results = [infer(model, inputs, method, implication, aggregation, width=width)
               for method, implication, aggregation in settings]
    primary = next(r for r in results if r['method']=='levels' and r['implication']=='mamdani' and r['aggregation']=='max')
    show_result(model, primary)
    print('\nСравнение на тех же входах:')
    print('Механизм       Импликация   Агрегация       Приоритет')
    for r in results:
        print(f"{r['method']:<15}{r['implication']:<13}{r['aggregation']:<16}{value_text(r['value'])}")
    path=Path.cwd()/('график_один_вход.svg' if len(inputs)==1 else 'график_три_входа.svg')
    save_graph(results,path)
    print('График сохранён:', path)
    print('Откройте файл SVG в браузере, чтобы посмотреть кривые.')


def menu(path):
    while True:
        print('\nЛР № 3 — нечёткие системы логического вывода')
        print('1 — один вход: ввести ошибки и сравнить результаты')
        print('2 — три входа: ввести ошибки, время и память')
        print('3 — расширенное сравнение операторов')
        print('4 — показать переменные и правила; 0 — выход')
        command = input('Выбор: ').strip()
        if command == '0':
            return
        try:
            base = load_base(path)
            if command == '4':
                for key,m in base['models'].items():
                    print(f"\n{key}: {m['name']}")
                    for name,v in [*m['inputs'].items(),('Приоритет',m['output'])]:
                        print(name, v['range'], {t:p['points'] for t,p in v['terms'].items()})
                    for i,r in enumerate(m['rules'],1):print(i,describe_rule(r))
                continue
            if command not in ('1','2','3'):
                print('Неизвестная команда')
                continue
            key = choose('Модель:',list(base['models'])) if command=='3' else ('single' if command=='1' else 'multi')
            model = base['models'][key]
            inputs = ask_inputs(model)
            width = ask_float('Неточность наблюдения: полуширина в п.п. (0 — точное значение)',12) if len(inputs)==1 else 0
            if command in ('1','2'):
                run_demo(model,inputs,width)
            else:
                agg = choose('Агрегация:',AGGREGATIONS)
                and_method = choose('Оператор И:',('min','product')) if len(inputs)>1 else 'min'
                show_comparison(compare(model,inputs,width,agg,and_method))
                print('50 при постоянном выходном множестве не означает средний приоритет.')
                print('Для логических импликаций с max нулевые предпосылки дают I(0,b)=1.')
        except (ValueError,KeyError,TypeError,OSError) as e:
            print('Ошибка:',e)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--model',choices=['single','multi'])
    parser.add_argument('--values',type=float,nargs='+')
    parser.add_argument('--width',type=float,default=0)
    parser.add_argument('--step',type=float,default=0.5)
    parser.add_argument('--method',choices=METHODS,default='levels')
    parser.add_argument('--implication',choices=list(IMPLICATIONS),default='mamdani')
    parser.add_argument('--aggregation',choices=AGGREGATIONS,default='max')
    parser.add_argument('--and-method',choices=['min','product'],default='min')
    parser.add_argument('--compare',action='store_true')
    parser.add_argument('--output',type=Path,help='JSON результата или CSV сравнения')
    args=parser.parse_args()
    if args.model is None:
        menu(args.base)
        return
    model=load_base(args.base)['models'][args.model]
    values=args.values or [v['default'] for v in model['inputs'].values()]
    if len(values)!=len(model['inputs']):
        raise ValueError('Число значений должно совпадать с числом входов')
    inputs=dict(zip(model['inputs'],values))
    if args.compare:
        results=compare(model,inputs,args.width,args.aggregation,args.and_method,args.step)
        show_comparison(results)
        if args.output:
            with args.output.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['method','implication','aggregation','value','area'])
                writer.writeheader()
                writer.writerows({k:r[k] for k in writer.fieldnames} for r in results)
    else:
        result=infer(model,inputs,args.method,args.implication,args.aggregation,args.and_method,args.width,args.step)
        show_result(model,result)
        if args.output:
            args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    try:
        main()
    except (EOFError,KeyboardInterrupt):
        print('\nРабота завершена.')
    except (ValueError,KeyError,TypeError,OSError) as error:
        raise SystemExit('Ошибка: '+str(error))
