"""Flat, human-facing reports over saved runs; never moves checkpoints or calls models."""
import json
from pathlib import Path
import os
import adaptive_pipeline as core


def report_dir(config,mode):
    base=core.ROOT/('experimenting/outputs' if config['profile']=='experimenting' else 'outputs')
    return base/('hasil_pilot10' if mode=='pilot' else 'hasil_benchmark')


def json_file(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def csv_file(path):
    return core.read_csv(path) if path.exists() else []


def write_rows(path,rows,initial):
    fields=list(initial)
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    core.atomic_csv(path,rows,fields)


def link(path,relative_to):
    return Path(os.path.relpath(path,relative_to)).as_posix().replace(' ','%20')


def refresh(config,mode):
    # Import lazily to avoid circular imports in the menu.
    from experiment_cli import destination
    folder=report_dir(config,mode); folder.mkdir(parents=True,exist_ok=True)
    sources=[];summary=[];baselines=[];samples=[];validity=[];errors=[]
    rewrites={(branch,step):[] for branch in ('llm','gan') for step in (1,2)}
    names=list(config['defenders'])
    for name in names:
        for branch in ('baseline','llm','gan'):
            run=destination(config,mode,branch)/name
            saved=json_file(run/'run.json');metrics=json_file(run/'metrics.json');checkpoint=json_file(run/'checkpoint.json')
            if not saved:
                summary.append((name,branch,'BELUM DIJALANKAN','','','','',''))
                continue
            if metrics.get('identities') and metrics['identities']!=saved.get('identities'):
                raise ValueError('Report/run identity mismatch: '+str(run))
            state=checkpoint.get('state','unknown').upper()
            if branch!='baseline' and saved.get('baseline_only'):state='BASELINE SAJA; REWRITE BELUM DIJALANKAN'
            eligible=metrics.get('eligible','');flips=metrics.get('observed_valid_flips','')
            invalid=metrics.get('invalid','');still=metrics.get('still_spam','')
            if branch=='baseline':flips=invalid=still=''
            summary.append((name,branch,state,metrics.get('selected_rows',''),eligible,flips,invalid,still))
            sources.append(dict(defender=name,branch=branch,run=str(run),identities=saved.get('identities',{}),
                checkpoint_state=state,metrics_updated_utc=core.api.dt.datetime.fromtimestamp((run/'metrics.json').stat().st_mtime,core.api.dt.timezone.utc).isoformat() if (run/'metrics.json').exists() else None))
            rows=csv_file(run/'sample_results.csv')
            if branch=='baseline':baselines.extend(dict(defender=name,**row) for row in rows)
            else:
                samples.extend(dict(branch='gemma' if branch=='llm' else 'gan',defender=name,**row) for row in rows)
                for step in (1,2):
                    rewrites[branch,step].extend(csv_file(run/('step'+str(step)+'_rewrites.csv')))
                validity.extend(dict(branch='gemma' if branch=='llm' else 'gan',defender=name,**row) for row in csv_file(run/'validity_summary.csv'))
            latest={}
            for event in csv_file(run/'events.csv'):latest[event['sample_id'],event['stage']]=event
            errors.extend(dict(branch=branch,defender=name,**event) for event in latest.values() if event['status']=='ERROR')
    write_rows(folder/'baseline.csv',baselines,['defender','sample_id','true_label','baseline_label','baseline_time_sec'])
    for (branch,step),rows in rewrites.items():
        write_rows(folder/(('gemma' if branch=='llm' else 'gan')+'_step'+str(step)+'.csv'),rows,
            ['defender','sample_id','step','validation_status','validation_reason','predicted_label','classification_status',
             'original_text','rewrite_text','generation_time_sec','validation_time_sec','classification_time_sec'])
    write_rows(folder/'rekap_per_email.csv',samples,['branch','defender','sample_id','true_label','baseline_label','state','step1_validation','step1_label','step2_validation','step2_label'])
    write_rows(folder/'validitas_per_step.csv',validity,['branch','defender','step','VALID','INVALID','UNCERTAIN','NO_CHANGE','ERROR'])
    write_rows(folder/'errors.csv',errors,['branch','defender','sample_id','stage','status','error'])
    core.atomic_json(folder/'sources.json',dict(updated_utc=core.api.utc_now(),profile=config['profile'],mode=mode,sources=sources))
    lines=['# Ringkasan hasil '+('pilot 10 data' if mode=='pilot' else 'benchmark'),'',
        'Profil: **'+config['profile']+'**. Diperbarui: '+core.api.utc_now()+'.',
        '', '## Mulai membaca di sini','',
        '| File | Isi |','|---|---|',
        '| [baseline.csv](baseline.csv) | Hasil klasifikasi original |',
        '| [rekap_per_email.csv](rekap_per_email.csv) | Status akhir tiap email, Gemma dan GAN dalam satu tabel |',
        '| [gemma_step1.csv](gemma_step1.csv), [gemma_step2.csv](gemma_step2.csv) | Teks rewrite Gemma dan hasil validasi/klasifikasi |',
        '| [gan_step1.csv](gan_step1.csv), [gan_step2.csv](gan_step2.csv) | Teks rewrite GAN dan hasil validasi/klasifikasi |',
        '| [validitas_per_step.csv](validitas_per_step.csv) | Jumlah diterima/ditolak serta alasan per step |',
        '| [errors.csv](errors.csv) | Error tahap yang belum terselesaikan; error lama yang sudah pulih tidak ditampilkan |',
        '', '## Status tugas','',
        '| Defender | Tugas | Status | Original | Eligible | Flip valid | Invalid | Tetap SPAM |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:lines.append('| '+' | '.join(str(v) for v in row)+' |')
    lines+=['','**BELUM DIJALANKAN bukan hasil nol atau gagal.** CSV kosong/header-only berarti belum ada baris tersimpan untuk tahap tersebut. '
        'Status RUNNING/INCOMPLETE berarti ringkasan bisa belum memuat semua progres. Kolom Flip/Invalid/Tetap SPAM pada baseline tidak digunakan untuk menilai serangan.',
        '', 'INVALID adalah hasil validasi, bukan error program. Hanya VALID boleh diklasifikasikan. '
        'Tidak ada Step 2 jika Step 1 ditolak atau sudah berhasil mengubah label. '
        'Bila classifier dan evaluator sama-sama PPLX, hasil ini belum mengukur ketahanan Clef/d3.',
        '', '## Cara melanjutkan','',
        'Gunakan menu utama **1 → 4 → 1** untuk melanjutkan kedua cabang; tahap OK dilewati. '
        'Gunakan menu 3 / `--retry-errors` hanya jika ingin mencoba ulang tahap ERROR. '
        'CLI memeriksa bobot dan runtime GAN sebelum memulai pekerjaan yang mencakup GAN.',
        '', 'Laporan ini adalah salinan baca dari hasil tersimpan, bukan checkpoint. Jangan mengedit CSV laporan untuk mengubah status run. '
        'Menu **4 → Perbarui ringkasan** membangunnya kembali tanpa API. '
        'Evaluasi kemiripan lengkap tersedia melalui menu 4 → Evaluasi/arsip teknis.',
        '', '## Arsip teknis (hanya bila diperlukan)','',
        'Event log, respons mentah, CSV individual dan cache tetap berada di folder run asal agar resume dan audit tetap cocok. '
        'Lokasi hasil asli:']
    for source in sources:
        lines.append('- '+source['defender']+' / '+source['branch']+': [folder run]('+link(Path(source['run']),folder)+').')
        evaluation=Path(source['run'])/'evaluation/evaluation.json'
        if evaluation.exists():lines.append('  Evaluasi kemiripan: [evaluation.json]('+link(evaluation,folder)+').')
    (folder/'RINGKASAN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    root=folder.parent
    (root/'BUKA_HASIL.md').write_text('# Buka hasil eksperimen\n\nMulai dari laporan berikut; tidak perlu menelusuri folder respons/cache.\n\n'
        '- [Pilot 10 data](hasil_pilot10/RINGKASAN.md)\n- [Benchmark penuh](hasil_benchmark/RINGKASAN.md)\n\n'
        'Laporan yang belum dibuat tersedia setelah run atau melalui menu 4 → Perbarui ringkasan. '
        'Folder validated_* / paper3000_* menyimpan checkpoint dan arsip teknis; jangan dihapus saat masih diperlukan untuk resume.\n',encoding='utf-8')
    return folder/'RINGKASAN.md'
