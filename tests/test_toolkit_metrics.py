from __future__ import annotations

import ast
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tortuosite_score.vessels_detection import toolkit_metrics as adapter
from tortuosite_score.vessels_detection.scoring import (
    available_scoring_methods, scoring_config, score_run, summarize_eye_score,
)
from tortuosite_score.visualization.toolkit_figures import numeric_substitution, atlas_population
from tortuosite_score.vessels_detection.toolkit_worker import compute

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workspace/tortuosity_metrics_python"
VENDOR = ROOT / "tortuosite_score/external_models/curve_toolkit"

# Independent reference: import the unmodified local repository, use its public
# Curve API and exactly the standalone batch recipe. Never import our vendor.
REFERENCE = '''
import json,sys,math,warnings
import numpy as np
from Curve import Curve
payload=json.load(sys.stdin)
p=np.asarray(payload['points'],dtype=float)
try:
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  c=Curve(p[:,0],p[:,1]); n=math.ceil(len(c)*.25)
  c=c.smooth('cubic_spline',smooth=8/c.arclength(),num_points=n).parametrize('scc',num_points=n)
  result={'points':c.points.tolist(),'metrics':{}}
  for key,registry in payload['metrics'].items():
   try:
    kw={'smooth':1.0} if key in ['mean_curvature','total_curvature','tau_3','tau_5','asdc','icm','td'] else {}
    value=float(c.tortuosity(registry,**kw))
    result['metrics'][key]={'value':value,'status':'ok' if np.isfinite(value) else 'non_finite'}
   except Exception as e:
    result['metrics'][key]={'status':'error','reason':type(e).__name__+': '+str(e)}
except Exception as e:
 result={'status':'error','reason':type(e).__name__+': '+str(e)}
print(json.dumps(result))
'''


class ToolkitParityTests(unittest.TestCase):
    @unittest.skipUnless(SOURCE.exists(), "local reference toolkit not present")
    def test_vendored_algorithms_differ_only_in_imports(self):
        for relative in ['Curve.py','Tortuosity.py','utils/Derivatives.py','utils/Smoothers.py','utils/Parametrization_Strategy.py','utils/geometry.py']:
            def without_imports(path):
                tree=ast.parse(path.read_text())
                tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
                return ast.dump(tree)
            self.assertEqual(without_imports(SOURCE/relative),without_imports(VENDOR/relative),relative)

    @unittest.skipUnless(SOURCE.exists(), "local reference toolkit not present")
    def test_all_metrics_match_original_on_synthetic_bundled_and_saved_curves(self):
        x=np.arange(240.)
        theta=np.linspace(0,1.5,240)
        saved=score_run(ROOT/'demo/streamlit_runs/1_OD',scoring_config('arc_chord'))[1]
        row=saved.loc[saved.vessel_name=='3°A-inf1'].iloc[0]
        cases=[np.column_stack([x,np.zeros_like(x)]),
               np.column_stack([100*np.cos(theta),100*np.sin(theta)]),
               np.column_stack([x,12*np.sin(x/15)]),
               np.column_stack([-x,2*np.sin(x/20)]), # direction crosses +/-180
               np.loadtxt(SOURCE/'data/RET_TORT/arteries/Arteria-010.txt'),
               np.asarray(row.path_points)*row.coordinate_scale,
               np.array([[0.,0.],[100.,0.]]),
               np.column_stack([np.linspace(0,700,8),np.zeros(8)]),
               np.zeros((20,2)),
               np.repeat(np.column_stack([x,np.sin(x/10)]),2,axis=0)]
        for i,points in enumerate(cases):
            with self.subTest(curve=i):
                run=subprocess.run([sys.executable,'-c',REFERENCE],cwd=SOURCE,
                    input=json.dumps({'points':points.tolist(),'metrics':adapter.METRICS}),
                    text=True,capture_output=True,timeout=30,env={**os.environ,'MPLBACKEND':'Agg'})
                self.assertEqual(run.returncode,0,run.stderr)
                expected=json.loads(run.stdout)
                for metric in adapter.METRICS:
                    actual=adapter.toolkit_metrics(points,metric)
                    if expected.get('status')=='error':
                        self.assertEqual(actual['status'],'error')
                        self.assertEqual(actual['reason'],expected['reason'])
                        continue
                    want=expected['metrics'][metric]
                    self.assertEqual(actual['status'],want['status'],(i,metric,actual['reason']))
                    self.assertTrue(np.array_equal(actual['preprocessing']['points'],expected['points']))
                    if want['status']=='ok':
                        self.assertAlmostEqual(actual['value'],want['value'],delta=max(1e-12,abs(want['value'])*1e-12))
                        # Every successful metric has source-captured formula terms.
                        text=numeric_substitution(metric,actual)
                        self.assertIn(f"{actual['value']:.7g}",text)

    def test_wrapping_and_asdc_quirks_are_not_corrected(self):
        points=np.array([[0.,0.],[-10.,.01],[-20.,0.],[-30.,.01]])
        soam=compute({'operation':'metric','metric':'soam','points':points.tolist()})
        self.assertGreater(max(abs(a) for a in soam['diagnostics']['compute']['alpha']),350)
        x=np.arange(200.)
        result=adapter.toolkit_metrics(np.column_stack([x,10*np.sin(x/12)]),'asdc')
        c=result['diagnostics']['compute']
        self.assertAlmostEqual(result['value'],np.trapezoid(np.gradient(c['kappa'])**2)/c['L_x'])

    def test_metric_failure_does_not_discard_other_results(self):
        # Equal endpoints: source DM is inf; SCC still produces a finite value.
        points=[[0.,0.],[10.,0.],[0.,0.]]
        dm=compute({'operation':'metric','metric':'dm','points':points})
        scc=compute({'operation':'metric','metric':'scc','points':points})
        self.assertEqual(dm['status'],'non_finite');self.assertTrue(math.isnan(dm['value']))
        self.assertEqual(scc['status'],'ok')


class ToolkitIntegrationTests(unittest.TestCase):
    def test_all_methods_share_prefix_and_existing_default(self):
        methods={s.method_id:s for s in available_scoring_methods()}
        for key in adapter.METRICS:
            self.assertTrue(methods[f'toolkit_{key}'].label.startswith('Toolkit — '))
        self.assertEqual(scoring_config().method_id,'local_bump')

    def test_timeout_kills_worker_and_recovers(self):
        worker=adapter._Worker()
        try:
            start=time.monotonic()
            failed=worker.call({'operation':'prepare','points':[[0,0],[100,0]]},.001)
            self.assertEqual(failed['status'],'timeout')
            self.assertLess(time.monotonic()-start,3)
            recovered=worker.call({'operation':'metric','metric':'dm','points':[[0,0],[100,0]]},30)
            self.assertEqual(recovered['value'],1)
        finally:
            worker.close()

    def test_preprocessing_is_shared_and_cached_values_are_not_mutable(self):
        x=np.arange(150.)+10000 # unique geometry for this test
        points=np.column_stack([x,5*np.sin(x/15)])
        adapter._prepare_cached.cache_clear();adapter._metric_cached.cache_clear()
        with tempfile.TemporaryDirectory() as cache,patch.dict(os.environ,{'TORTUOSITY_TOOLKIT_CACHE':cache}):
            with patch.object(adapter._WORKER,'call',wraps=adapter._WORKER.call) as call:
                first=adapter.toolkit_metrics(points,'dm');adapter.toolkit_metrics(points,'scc')
                self.assertEqual(sum(c.args[0]['operation']=='prepare' for c in call.call_args_list),1)
                first['preprocessing']['points'][0][0]=-999
                self.assertNotEqual(adapter.toolkit_metrics(points,'dm')['preprocessing']['points'][0][0],-999)

    def test_invalid_scores_excluded_even_when_length_filter_disabled(self):
        table=pd.DataFrame({'primary_score':[1.,3.,np.nan,np.inf],
            'vessel_length':[100.,300.,900.,900.], 'score_geometry_valid':[True,True,False,False],
            'eligible':[True,True,False,False],'category':['artere']*4,
            'arc_chord_diagnostic':[1.]*4})
        from tortuosite_score.vessels_detection.local_bump_score import LocalBumpSettings
        config=scoring_config('toolkit_dm',LocalBumpSettings(filter_short_vessels=False))
        result=summarize_eye_score(table,config)
        self.assertEqual(result['eligible_vessel_count'],2)
        self.assertEqual(result['eligible_total_length'],400)
        self.assertEqual(result['eye_tortuosity_score'],2.5)
        self.assertTrue(math.isnan(result['all_vessels_tail_score']))

    def test_empty_atlas_and_index_tables_fit_on_page(self):
        from matplotlib.backends.backend_pdf import PdfPages
        from tortuosite_score.visualization.toolkit_figures import write_toolkit_atlas, _table_page
        buffer=io.BytesIO()
        with PdfPages(buffer) as pdf:
            stats=write_toolkit_atlas(pdf,[],scoring_config('toolkit_dm'))
        self.assertEqual(stats['artery_count'],0)
        self.assertTrue(buffer.getvalue().startswith(b'%PDF'))
        # Full index tables must fit even at their maximum page population.
        class Capture:
            def savefig(self,figure,**kwargs):self.figure=figure
        capture=Capture()
        _table_page(capture,'Index',pd.DataFrame({'Artère':['1°A-inf']*28,'Score':[1]*28}))
        table=next(iter(capture.figure.axes[0].tables))
        self.assertLessEqual(sum(cell.get_height() for (row,col),cell in table.get_celld().items() if col==0),1)

    def test_vein_atlas_uses_vein_population_and_labels(self):
        from tortuosite_score.visualization.toolkit_figures import _cover, _detail_page
        run=ROOT/'demo/streamlit_runs/1_OD';config=scoring_config('toolkit_dm')
        summary,table=score_run(run,config)
        scored=[(run,summary,table)]
        records,excluded=atlas_population(scored,vessel_type='veine')
        self.assertGreater(len(records),0)
        self.assertTrue(all(r.name_info.vessel_type=='veine' for r in records))
        self.assertEqual([r.row['primary_score'] for r in records],sorted(r.row['primary_score'] for r in records))
        self.assertFalse(any(r.name_info.vessel_type=='artere' for r in records))
        figure=_detail_page(records[0],config,1,len(records),3)
        text=' '.join(t.get_text() for t in figure.texts)
        self.assertIn(f"{records[0].name_info.rank}°V",text)
        self.assertIn('this vein',text)
        self.assertNotIn('this artery',text)
        class Capture:
            def savefig(self,figure,**kwargs):self.figure=figure
        capture=Capture()
        _cover(capture,config,records,excluded,vessel_type='veine')
        cover=' '.join(t.get_text() for t in capture.figure.texts)
        self.assertIn('veins',cover)
        self.assertIn('Atlas: named rank 1–3 veins',cover)
        with self.assertRaises(ValueError):atlas_population(scored,vessel_type='invalid')

    def test_three_pdf_archive_contains_report_and_both_real_atlases(self):
        import zipfile
        from tortuosite_score.app.results_page import generate_toolkit_pdf_archive, _build_local_bump_summary_table
        run=ROOT/'demo/streamlit_runs/1_OD';config=scoring_config('toolkit_dm')
        summary,table=score_run(run,config);scored=[(run,summary,table)]
        # Inspect text artists from every rendered page, including charts and tables.
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.text import Text
        page_text = []
        savefig = PdfPages.savefig
        def capture(pdf, figure, *args, **kwargs):
            texts = [item.get_text() for item in figure.findobj(Text)]
            for ax in figure.axes:
                for table in ax.tables:
                    texts.extend(cell.get_text().get_text() for cell in table.get_celld().values())
            page_text.append(" ".join(" ".join(text.split()) for text in texts))
            return savefig(pdf, figure, *args, **kwargs)
        with patch.object(PdfPages, "savefig", capture):
            archive=generate_toolkit_pdf_archive(scored,config,_build_local_bump_summary_table(scored))
        text = " ".join(page_text)
        for expected in ("Retinal tortuosity report", "Model and preprocessing",
                         "Weighted mean", "Raw p-values", "BH-adjusted p-values",
                         "Vessel", "Category", "artery", "vein",
                         "Atlas: named rank 1–3 arteries", "Atlas: named rank 1–3 veins",
                         "Numerical substitution", "Normalized x (px)"):
            self.assertIn(expected, text)
        for french in ("Rapport", "Vaisseau", "Categorie", "Methode", "moyenne",
                       "courbure", "Tracé", "Fiche", "veines", "artères", "Motif"):
            self.assertNotIn(french, text)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            self.assertIsNone(bundle.testzip())
            self.assertEqual(set(bundle.namelist()),{
                'rapport_tortuosite_toolkit_dm.pdf',
                'atlas_toolkit_dm_arteres.pdf','atlas_toolkit_dm_veines.pdf'})
            for name in bundle.namelist():self.assertTrue(bundle.read(name).startswith(b'%PDF'))
        from tortuosite_score.visualization.toolkit_figures import write_toolkit_atlas
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(io.BytesIO()) as pdf:
            stats=write_toolkit_atlas(pdf,[],config,vessel_type='veine')
        self.assertEqual(stats['vein_count'],0)
        self.assertEqual(stats['excluded_vein_count'],0)

    def test_saved_eye_export_and_atlas_agree(self):
        from tortuosite_score.vessels_detection.clinical_excel import generate_classified_vessels_excel
        run=ROOT/'demo/streamlit_runs/1_OD';config=scoring_config('toolkit_dm')
        summary,table=score_run(run,config)
        for _,row in table.iterrows():
            raw=np.asarray(row.path_points)
            normalized=adapter.toolkit_metrics(raw*row.coordinate_scale,'dm')
            native=adapter.toolkit_metrics(raw,'dm')
            self.assertEqual(row.toolkit_status,normalized['status'])
            if normalized['status']=='ok':self.assertEqual(row.primary_score,normalized['value'])
            if native['status']=='ok':self.assertEqual(row.raw_primary_score,native['value'])
        book=pd.ExcelFile(io.BytesIO(generate_classified_vessels_excel([run],config)))
        clean=pd.read_excel(book,'Clean_vessels');rejected=pd.read_excel(book,'Rejected_vessels')
        self.assertIn('toolkit_failure_reason',rejected.columns)
        # Preserve the workbook's classified-artery membership rule: invalid
        # arteries can remain in Clean_vessels with NA, but never in statistics.
        finite=clean.score_normalized.notna()
        self.assertTrue(clean.loc[finite,'toolkit_status'].eq('ok').all())
        self.assertEqual(len(clean)+len(rejected),len(table))
        combined=pd.concat([clean,rejected],ignore_index=True)
        self.assertTrue(combined.toolkit_failure_reason.notna().any())
        invalid=combined.toolkit_status.ne('ok')
        self.assertTrue(combined.loc[invalid,'score_normalized'].isna().all())
        records,excluded=atlas_population([(run,summary,table)])
        self.assertEqual([r.row['primary_score'] for r in records],sorted(r.row['primary_score'] for r in records))
        self.assertTrue(all(r.name_info.vessel_type=='artere' for r in records))
        self.assertTrue(any('A curve requires' in r['Reason'] for r in excluded))


if __name__=='__main__':unittest.main()
