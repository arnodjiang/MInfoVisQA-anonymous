pass
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .api import now, read, save

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data/visual_benchmark/baseline')
    p.add_argument('--source', default='data/processed/normal_qa_v4')
    p.add_argument('--stage-timeout', type=int, default=900)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--audit', action='store_true')
    p.add_argument('--audit-language-batch-size', type=int, default=3)
    p.add_argument('--repair-plan')
    p.add_argument('--package-output')
    a = p.parse_args()
    out = Path(a.output).resolve()
    env = dict(os.environ, MVISQA_SOURCE=str(Path(a.source).resolve()), MVISQA_OUTPUT=str(out))
    env.pop('MVISQA_DISABLE_SANDBOX', None)
    logs = out / 'serial_runs'
    logs.mkdir(exist_ok=True)
    run_id = str(time.time_ns())
    ids = [r['id'] for r in map(json.loads, (out / 'source/candidates.jsonl').read_text().splitlines())]
    repairs = read(a.repair_plan).get('repairs', {}) if a.repair_plan else {}
    if repairs:
        order = {'translation': 0, 'layout': 1, 'source': 2}
        ids.sort(key=lambda cid: order.get(repairs.get(cid), 3))

    def run(module, args, label, cid=None):
        started = now()
        path = logs / (run_id + '_' + label + '.log')
        with path.open('w') as stream:
            proc = subprocess.Popen([sys.executable, '-m', 'scripts.final_benchmark.' + module] + args, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save(logs / 'active_stage.json', {'label': label, 'pid': proc.pid, 'started_at': started, 'deadline_unix': time.time() + a.stage_timeout, 'status': 'running', 'log': str(path)})
            timed_out = False
            try:
                deadline = time.time() + a.stage_timeout
                while proc.poll() is None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(proc.args, a.stage_timeout)
                    try:
                        proc.wait(timeout=min(5, remaining))
                    except subprocess.TimeoutExpired:
                        pass
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                if cid:
                    for attempt in (out / 'api').glob('*/' + cid + '/*/attempt_*.json'):
                        meta = read(attempt)
                        if meta.get('status') == 'started' and meta.get('started_at', '') >= started:
                            meta.update(status='interrupted', error_type='StageWallClockTimeout', finished_at=now())
                            save(attempt, meta)
        result = dict(label=label, started_at=started, finished_at=now(), exit_code=proc.returncode, timed_out=timed_out, stage_wall_clock_limit_seconds=a.stage_timeout, log=str(path))
        if cid and module == 'pipeline':
            stage = args[args.index('--stage') + 1]
            markers = {'recovery': 'recovery_complete.json', 'qa': 'qa.json', 'render': 'render_complete.json'}
            result['artifact_complete'] = len(list((out / 'cases' / cid / 'locales').glob('*.json'))) == 11 if stage == 'translate' else (out / 'cases' / cid / markers[stage]).exists()
        elif cid and module == 'val_audit':
            result['artifact_complete'] = (out / 'validation_release/reviews' / f'{cid}.json').exists()
        save(logs / 'active_stage.json', dict(result, status='finished'))
        with (logs / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
    for round_id in range(a.rounds):
        for cid in ids:
            f = out / 'cases' / cid
            if cid in repairs and (not (f / 'quality_repair_complete.json').exists()):
                run('quality_repair', ['--output', str(out), '--id', cid, '--mode', repairs[cid]], f'{round_id}_{cid}_quality_repair', cid)
                if not (f / 'quality_repair_complete.json').exists():
                    continue
            requirements = [('recovery', 'recovery_complete.json'), ('qa', 'qa.json'), ('translate', None), ('render', 'render_complete.json')]
            for (stage, marker) in requirements:
                if marker and (f / marker).exists():
                    continue
                if stage == 'translate' and len(list((f / 'locales').glob('*.json'))) == 11:
                    continue
                if stage == 'qa' and (not (f / 'recovery_complete.json').exists()):
                    break
                if stage == 'translate' and (not (f / 'qa.json').exists()):
                    break
                if stage == 'render' and len(list((f / 'locales').glob('*.json'))) != 11:
                    break
                run('pipeline', ['--output', str(out), '--stage', stage, '--ids', cid, '--workers', '1', '--retry-failed'], f'{round_id}_{cid}_{stage}', cid)
                if marker and (not (f / marker).exists()):
                    break
            if cid in repairs and a.audit and (f / 'quality_repair_complete.json').exists() and (f / 'render_complete.json').exists() and (not (out / 'validation_release/reviews' / f'{cid}.json').exists()):
                run('export', ['--output', str(out), '--partial'], f'{round_id}_{cid}_export')
                run('val_verify', [], f'{round_id}_{cid}_verify')
                run('val_audit', ['--ids', cid, '--workers', '1', '--retry-failed', '--language-batch-size', str(a.audit_language_batch_size)], f'{round_id}_{cid}_reaudit', cid)
        run('export', ['--output', str(out), '--partial'], f'{round_id}_export')
        run('val_verify', [], f'{round_id}_verify')
        run('val_audit', ['--export-only'], f'{round_id}_admission')
        run('val_report', [], f'{round_id}_report')
        if all(((out / 'cases' / cid / 'render_complete.json').exists() for cid in ids)) and all(((out / 'cases' / cid / 'quality_repair_complete.json').exists() for cid in repairs)):
            break
    if a.audit:
        for cid in ids:
            if not (out / 'cases' / cid / 'render_complete.json').exists():
                continue
            if (out / 'validation_release/reviews' / f'{cid}.json').exists():
                continue
            run('val_audit', ['--ids', cid, '--workers', '1', '--retry-failed', '--language-batch-size', str(a.audit_language_batch_size)], f'audit_{cid}', cid)
        run('val_audit', ['--export-only'], 'final_admission')
        run('val_report', [], 'final_report')
    if a.package_output:
        run('package_validation', ['--input', str(out), '--output', a.package_output], 'package')
    save(logs / 'last_completed_run.json', {'run_id': run_id, 'finished_at': now(), 'generated_cases': sum(((out / 'cases' / cid / 'render_complete.json').exists() for cid in ids)), 'expected_cases': len(ids), 'note': 'Completion of this bounded run is not certification of the benchmark.'})
if __name__ == '__main__':
    main()
