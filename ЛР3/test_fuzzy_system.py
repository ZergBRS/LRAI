import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import нечёткая_система as app


class FuzzyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=app.load_base()
        cls.single=cls.base['models']['single']
        cls.multi=cls.base['models']['multi']

    def test_memberships_and_shoulders(self):
        t={'type':'triangle','points':[0,5,10]}
        self.assertEqual([app.membership(x,t) for x in [-1,0,2.5,5,7.5,10,11]],[0,0,.5,1,.5,0,0])
        self.assertEqual(app.membership(0,{'type':'triangle','points':[0,0,5]}),1)
        self.assertEqual(app.membership(10,{'type':'triangle','points':[5,10,10]}),1)

    def test_trapezoid(self):
        t={'type':'trapezoid','points':[0,2,6,10]}
        self.assertEqual([app.membership(x,t) for x in [0,1,2,4,6,8,10]],[0,.5,1,1,1,.5,0])

    def test_implications_by_hand(self):
        expected={'mamdani':.3,'larsen':.12,'kleene_dienes':.6,'lukasiewicz':.9}
        for key,val in expected.items():
            with self.subTest(key=key):self.assertAlmostEqual(app.IMPLICATIONS[key](.4,.3),val)

    def test_zero_antecedent_is_not_silently_discarded(self):
        self.assertEqual(app.IMPLICATIONS['kleene_dienes'](0,.2),1)
        self.assertEqual(app.IMPLICATIONS['lukasiewicz'](0,.2),1)
        r=app.infer(self.single,{'Ошибки':52},implication='kleene_dienes')
        self.assertTrue(all(y==1 for y in r['membership']))
        self.assertAlmostEqual(r['value'],50)

    def test_aggregation_by_hand(self):
        for name,expected in [('max',.7),('min',.2),('prob_sum',.76),('bounded_sum',.9)]:
            with self.subTest(name=name):self.assertAlmostEqual(app.aggregate([.2,.7],name),expected)

    def test_compositions_on_manual_matrix(self):
        relation=[[.2,.8],[.6,.4]]
        for method,expected in [('max_min',[.6,.5]),('max_product',[.54,.4])]:
            with self.subTest(method=method):
                for a,b in zip(app.compose([.5,.9],relation,method),expected):self.assertAlmostEqual(a,b)

    def test_centroid_asymmetric_triangle_analytic(self):
        val,area=app.centroid([0,1,4],[0,1,0])
        self.assertAlmostEqual(area,2)
        self.assertAlmostEqual(val,5/3)

    def test_empty_output_has_no_invented_centroid(self):
        self.assertEqual(app.centroid([0,1,2],[0,0,0]),(None,0))
        self.assertIsNone(app.infer(self.single,{'Ошибки':52},aggregation='min')['value'])

    def test_rule_counts_and_coverage(self):
        self.assertEqual(len(self.single['rules']),7)
        self.assertEqual(len(self.multi['rules']),27)
        self.assertEqual(len(self.multi['inputs']),3)
        for m in [self.single,self.multi]:
            for v in [*m['inputs'].values(),m['output']]:
                for x in app.universe(v['range'],.5):
                    self.assertGreater(max(app.membership(x,t) for t in v['terms'].values()),0)

    def test_crisp_compositions_coincide_including_off_grid(self):
        for x in [0,17.31,52,100]:
            a=app.infer(self.single,{'Ошибки':x},method='max_min')
            b=app.infer(self.single,{'Ошибки':x},method='max_product')
            self.assertEqual(a['membership'],b['membership'])

    def test_mamdani_composition_equals_truth_levels_for_fuzzy_input(self):
        a=app.infer(self.single,{'Ошибки':52},method='max_min',width=12)
        b=app.infer(self.single,{'Ошибки':52},method='levels',width=12)
        for x,y in zip(a['membership'],b['membership']):self.assertAlmostEqual(x,y)
        # Пересечение двух линейных склонов: α=1−расстояние/(w+полуширина терма).
        half_width=self.single['inputs']['Ошибки']['terms']['умеренные']['points'][2]-50
        self.assertAlmostEqual(b['alpha'][3],1-2/(12+half_width))

    def test_two_compositions_are_distinct_for_fuzzy_observation(self):
        a=app.infer(self.single,{'Ошибки':52},method='max_min',width=12)
        b=app.infer(self.single,{'Ошибки':52},method='max_product',width=12)
        self.assertGreater(abs(a['value']-b['value']),.1)

    def test_multi_rule_firing_by_hand(self):
        r=app.infer(self.multi,{'Ошибки':52,'Время':55,'Память':70})
        # Условия «средний, средний, большой»: min(.96,55/60,.4)=.4.
        index=next(i for i,rule in enumerate(self.multi['rules'])
                   if rule['if']=={'Ошибки':'средний','Время':'средний','Память':'большой'})
        self.assertAlmostEqual(r['alpha'][index],.4)
        p=app.infer(self.multi,{'Ошибки':52,'Время':55,'Память':70},and_method='product')
        self.assertAlmostEqual(p['alpha'][index],.96*(55/60)*.4)

    def test_each_input_changes_priority(self):
        start={'Ошибки':52,'Время':55,'Память':70}
        normal=app.infer(self.multi,start)['value']
        for name,value in [('Ошибки',80),('Время',100),('Память',95)]:
            with self.subTest(name=name):
                changed=app.infer(self.multi,dict(start,**{name:value}))['value']
                self.assertGreater(changed,normal)

    def test_severity_increases_on_reference_scenarios(self):
        values=[app.infer(self.multi,dict(zip(self.multi['inputs'],row)))['value']
                for row in [(10,10,10),(52,55,70),(90,100,90)]]
        self.assertEqual(values,sorted(values))
        self.assertGreater(values[-1]-values[0],50)

    def test_all_operator_combinations_are_bounded(self):
        for implication in app.IMPLICATIONS:
            for aggregation in app.AGGREGATIONS:
                for method in app.METHODS:
                    r=app.infer(self.single,{'Ошибки':52},method,implication,aggregation,width=12,step=2)
                    self.assertTrue(all(-1e-12 <= v <= 1+1e-12 for v in r['membership']))
                    if r['value'] is not None:self.assertTrue(0<=r['value']<=100)

    def test_finer_grid_stabilizes_result(self):
        a=app.infer(self.single,{'Ошибки':52},width=12,step=.5)['value']
        b=app.infer(self.single,{'Ошибки':52},width=12,step=.25)['value']
        self.assertLess(abs(a-b),.03)

    def test_wrong_inputs_and_options_are_rejected(self):
        cases=[({'Ошибки':-1},{}),({'Ошибки':101},{}),({'Ошибки':float('nan')},{}),
               ({},{}),({'Ошибки':52},{'width':-1}),({'Ошибки':52},{'step':0}),
               ({'Ошибки':52},{'implication':'unknown'}),({'Ошибки':52},{'and_method':'unknown'})]
        for inputs,kw in cases:
            with self.subTest(inputs=inputs,kw=kw):
                with self.assertRaises(ValueError):app.infer(self.single,inputs,**kw)

    def test_relational_multi_input_is_explicitly_rejected(self):
        with self.assertRaises(ValueError):
            app.infer(self.multi,{'Ошибки':52,'Время':55,'Память':70},method='max_min')

    def test_external_rule_change_changes_actual_result(self):
        data=copy.deepcopy(self.base)
        for rule in data['models']['single']['rules']:rule['then']='критический'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'kb.txt';p.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
            m=app.load_base(p)['models']['single']
            self.assertGreater(app.infer(m,{'Ошибки':52})['value'],90)

    def test_external_base_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'kb.txt'
            for raw in ['{"models":{},"models":{}}','{}','not json']:
                p.write_text(raw,encoding='utf-8')
                with self.assertRaises(ValueError):app.load_base(p)
            bad=copy.deepcopy(self.base)
            bad['models']['single']['rules'][0]['then']='опечатка'
            p.write_text(json.dumps(bad),encoding='utf-8')
            with self.assertRaises(ValueError):app.load_base(p)

    def test_cli_single_multi_and_exports(self):
        script=Path(app.__file__)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result.json'
            r=subprocess.run([sys.executable,'-B',str(script),'--model','single','--values','52','--width','12','--output',str(output)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('Приоритет диагностики:',r.stdout)
            self.assertAlmostEqual(json.loads(output.read_text())['value'],51.3708,places=3)
            r=subprocess.run([sys.executable,'-B',str(script),'--model','multi','--values','52','55','70'],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('56.0788',r.stdout)
            csv=Path(tmp)/'compare.csv'
            r=subprocess.run([sys.executable,'-B',str(script),'--model','single','--values','52','--width','12','--compare','--output',str(csv)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(len(csv.read_text(encoding='utf-8-sig').splitlines()),13)

    def test_readable_text_base_and_rule_edit(self):
        text=app.BASE.read_text(encoding='utf-8')
        self.assertIn('ЕСЛИ Ошибки=умеренные ТО Приоритет=средний',text)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'base.txt'
            path.write_text(text.replace('ТО Приоритет=средний','ТО Приоритет=критический'),encoding='utf-8')
            changed=app.load_base(path)['models']['single']
            self.assertGreater(app.infer(changed,{'Ошибки':50})['value'],90)
            for old,new in [('TERM единичные','TERM умеренные'),('Ошибки=единичные','Ошибки=опечатка'),('\nINPUT Ошибки','\nUNKNOWN Ошибки')]:
                path.write_text(text.replace(old,new,1),encoding='utf-8')
                with self.assertRaises(ValueError):app.load_base(path)

    def test_interactive_menu(self):
        from xml.etree import ElementTree
        for commands,filename,result in [('1\n52\n12\n0\n','график_один_вход.svg','51.9548'),
                                         ('2\n52\n55\n70\n0\n','график_три_входа.svg','56.0788')]:
            with tempfile.TemporaryDirectory() as tmp:
                r=subprocess.run([sys.executable,'-B',app.__file__],input=commands,cwd=tmp,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                self.assertIn(result,r.stdout)
                self.assertNotIn('Ошибка:',r.stdout)
                graph=ElementTree.parse(Path(tmp)/filename)
                self.assertGreaterEqual(len(graph.findall('.//{http://www.w3.org/2000/svg}polyline')),4)


if __name__=='__main__':unittest.main(verbosity=2)
