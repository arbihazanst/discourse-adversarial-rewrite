"""Menu terminal main/experimenting dan perintah eksperimen tanpa launcher per model."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import adaptive_pipeline as pipeline

HERE=Path(__file__).resolve().parent
MAIN=pipeline.ROOT/'decision_models/adaptive_main.json'
LOCAL=pipeline.ROOT/'decision_models/adaptive_local.json'


def ask(label, default=''):
    value=input(f'{label} [{default or "empty"}]: ').strip()
    return '' if value=='-' else value or default


def load_profile(path):
    value=json.loads(path.read_text(encoding='utf-8-sig'))
    if value.get('profile') not in {'main','experimenting'}:raise ValueError('Profil harus main atau experimenting')
    if not value.get('defenders'):raise ValueError('Configure at least one defender')
    if value.get('protocol')!='validity_gated_v1' or not value.get('rewrite_evaluator'):
        raise ValueError('Profil lama tanpa validity gate. Gunakan --config decision_models/adaptive_main.json atau --config experimenting/adaptive_config.json; pindahkan pengaturan lokal secara eksplisit.')
    from validity_pipeline import configuration
    configuration(value)
    for name in value['defenders']:pipeline.validate_model_key(name)
    return value


def edit_endpoint(current, attacker=False):
    config=copy.deepcopy(current)
    allowed='openai, ollama_chat' if attacker else 'direct, openwebui, ollama, tool'
    config['transport']=ask('Transport ('+allowed+')',config.get('transport','openai' if attacker else 'direct'))
    config['url']=ask('URL endpoint lengkap',config.get('url',''))
    config['model']=ask('Model / Pipe ID di server',config.get('model',''))
    config['revision']=ask('Revision + quantization sebenarnya (- untuk kosong)',config.get('revision',''))
    config['key_env']=ask('NAMA variabel API key (- jika tanpa autentikasi)',config.get('key_env',''))
    if config['transport']=='tool':config['tool_id']=ask('Tool ID terverifikasi',config.get('tool_id',''))
    else:config.pop('tool_id',None)
    if not attacker and config['transport']=='direct':
        path=ask('File template JSON native (wajib untuk PPLX direct)')
        if path:config['template']=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    else:config.pop('template',None)
    if attacker:
        config['temperature']=float(ask('Temperature',str(config.get('temperature',0))))
        config['max_tokens']=int(ask('Maksimal token output',str(config.get('max_tokens',1024))))
        if config['temperature']<0 or config['max_tokens']<1:raise ValueError('Invalid attacker settings')
    pipeline.validate_endpoint(config,attacker=attacker,live=False)
    if not attacker:pipeline.defender_payload(config,'__PREFLIGHT__')
    return config


def build_command(config_path, model, action, reuse=False, attack_kind='llm'):
    command=[sys.executable,str(HERE/('gan_pipeline.py' if attack_kind=='gan' else 'adaptive_pipeline.py')),'--config',str(config_path),'--model',model]
    flags={'dry':['--dry-run','--pilot'], 'baseline':['--baseline-only','--pilot'],
           'pilot':['--pilot','--resume'], 'full':['--resume'], 'retry':['--resume','--retry-errors'],
           'check':['--check-server']}
    command+=flags[action]
    if reuse and attack_kind=='llm':command.append('--reuse-step1')
    return command


def child_environment(config):
    env=os.environ.copy()
    # Child runners can load .env credentials without restoring route overrides
    # that the menu intentionally removed in favor of the displayed profile.
    env['DECISION_ENV_KEYS_ONLY']=','.join(sorted({item.get('key_env','') for item in
        [*config['defenders'].values(),config.get('attacker',{}),config.get('rewrite_evaluator',{})] if item.get('key_env')}))
    # Values displayed in the menu are authoritative; retain API-key variables,
    # but remove route overrides that would silently supersede this profile.
    for key in ('ATTACKER_URL','ATTACKER_MODEL','ATTACKER_REVISION','ATTACKER_API_KEY_ENV'):
        env.pop(key,None)
    for name in set(config['defenders'])|{'pplx'}:
        for suffix in ('URL','MODEL_ID','CHAT_MODEL','CHECKPOINT_ID','TOOL_ID'):env.pop(name.upper()+'_'+suffix,None)
    return env


def run_command(command, config):
    print('\nMenjalankan tahap. Ctrl+C menyimpan progres dan kembali ke menu.',flush=True)
    process=subprocess.Popen(command,cwd=pipeline.ROOT,env=child_environment(config))
    try:code=process.wait()
    except KeyboardInterrupt:
        # Foreground Ctrl+C reaches the child as well. Let it commit its snapshot.
        print('\nWaiting for the runner to save progress...',flush=True)
        try:code=process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate();code=process.wait()
    print('\nSelesai.' if code==0 else f'\nTahap belum selesai (exit {code}). Periksa pesan error; hasil tersimpan dapat dilanjutkan.')
    return code


def output_for(config,attack_kind):
    return config.get('gan_output_dir','outputs/paper3000_gan') if attack_kind=='gan' else config['output_dir']


EXPERIMENT=pipeline.ROOT/'experimenting'
TASKS=[('baseline','Baseline saja - klasifikasi original, tanpa rewrite'),
       ('llm','Gemma - maksimal 2 rewrite + validasi PPLX + klasifikasi'),
       ('gan','GAN SST - maksimal 2 rewrite + validasi PPLX + klasifikasi'),
       ('all','Gemma + GAN - dua cabang, baseline dipakai ulang')]
OUTPUT_KEYS={
    ('pilot','baseline'):'tool_pilot_output_dir',
    ('pilot','llm'):'tool_llm_pilot_output_dir',
    ('pilot','gan'):'tool_gan_pilot_output_dir',
    ('full','baseline'):'output_dir',
    ('full','llm'):'output_dir',
    ('full','gan'):'gan_output_dir',
}


def destination(config,mode,task):
    prefix='experimenting/' if config['profile']=='experimenting' else ''
    defaults={'baseline':'pplx_gemma_fixed_pilot10','llm':'pplx_llm_pilot10','gan':'pplx_gan_pilot10'}
    fallback=prefix+'outputs/'+(defaults[task] if mode=='pilot' else
                               'paper3000_gan' if task=='gan' else 'paper3000_adaptive')
    return pipeline.rooted(config.get(OUTPUT_KEYS[mode,task],fallback))


def validate_paths(config):
    paths={key:destination(config,*key) for key in OUTPUT_KEYS}
    for path in paths.values():
        if config['profile']=='experimenting' and not path.is_relative_to(EXPERIMENT.resolve()):
            raise ValueError('Output experimenting harus berada di dalam folder experimenting.')
        if config['profile']=='main' and path.is_relative_to(EXPERIMENT.resolve()):
            raise ValueError('Output main tidak boleh masuk ke folder experimenting.')
    distinct=[paths['pilot',kind] for kind in ('baseline','llm','gan')]
    distinct.extend([paths['full','llm'],paths['full','gan']])
    if len(set(distinct))!=5:raise ValueError('Pisahkan folder pilot baseline, Gemma, GAN, dan kedua benchmark penuh.')


def banner(title):
    print('\n'+'='*64+'\n'+title+'\n'+'='*64)


def field(name,value):
    import textwrap,shutil
    width=max(48,min(100,shutil.get_terminal_size((80,24)).columns))
    value=str(value).replace(str(pipeline.ROOT)+os.sep,'')
    print(textwrap.fill(f'{name}: {value}',width=width,subsequent_indent='  ',break_long_words=False,break_on_hyphens=False))


def choose(title,options):
    while True:
        print('\n'+title)
        for index,(_,label) in enumerate(options,1):print(f'  {index}. {label}')
        print('  0. Kembali')
        answer=input('Pilih nomor: ').strip()
        if answer=='0':return None
        if answer.isdigit() and 1<=int(answer)<=len(options):return options[int(answer)-1][0]
        print('Nomor tidak tersedia. Pilih dari daftar di atas.')


def show_profile(config,path,selected):
    banner('DECISION RESEARCH | '+config['profile'].upper())
    field('Profil',path)
    field('Dataset',config['dataset'])
    names=pipeline.selected_models(config,selected)
    for name in names:
        d=config['defenders'][name]
        field('Defender '+name,d.get('model') or '(model belum diisi)')
        field('Endpoint',d['url'])
    field('Attacker Gemma',config['attacker']['model']+' | '+config['attacker']['url'])
    if config.get('rewrite_evaluator'):
        evaluator=config['rewrite_evaluator']
        field('Evaluator validitas','PPLX | '+evaluator['url'])
        print('Invalid: simpan alasan, hentikan data; hanya VALID dikirim ke defender.')
    print('GAN: model SST lokal di CPU. Label spam/ham tetap berasal dari defender.')
    print('Pilot: 10 original (5 spam + 5 ham). Benchmark: seluruh dataset.')


def build_plan(path,config,selected,mode,task,retry=False,dry=False,check=False):
    validate_paths(config)
    pipeline.selected_models(config,selected)
    if mode=='pilot' and config['profile']=='experimenting' and selected!='pplx':
        raise ValueError('Pilot koneksi 10 data saat ini khusus PPLX. Pilih PPLX di Pengaturan; model lain tersedia di benchmark.')
    steps=['baseline']+(['llm','gan'] if task=='all' else [task] if task!='baseline' else [])
    plan=[]
    for step in steps:
        if mode=='pilot' and config['profile']=='experimenting':
            command=[sys.executable,str(HERE/'pilot_openwebui.py'),'--config',str(path),
                     '--profile-routes','--attack','none' if step=='baseline' else step]
        else:
            command=build_command(path,selected,'full',config.get('reuse_historical_step1',False),
                                  'gan' if step=='gan' else 'llm')
            if step=='baseline':
                command=[item for item in command if item!='--reuse-step1']
                command.append('--baseline-only')
        if config.get('rewrite_evaluator') and not (mode=='pilot' and config['profile']=='experimenting'):
            command+=['--output-dir',str(destination(config,mode,step))]
            if mode=='pilot':command.append('--pilot')
            if step!='baseline' and destination(config,mode,'baseline')!=destination(config,mode,step):
                command+=['--baseline-from',str(destination(config,mode,'baseline'))]
        if retry:command.append('--retry-errors')
        if dry:command.append('--dry-run')
        if check:command.append('--check-server')
        plan.append((step,command,destination(config,mode,step)))
    return plan


def execute_plan(path,config,selected,mode,task,retry=False,dry=False,check=False):
    plan=build_plan(path,config,selected,mode,task,retry,dry,check)
    banner('RENCANA EKSPERIMEN')
    field('Ukuran','Pilot 10 data' if mode=='pilot' else 'Benchmark seluruh dataset')
    field('Defender',selected)
    field('Tindakan','Periksa offline (tanpa API)' if dry else 'Cek layanan (tanpa klasifikasi)' if check else
          'Resume + ulangi tahap ERROR' if retry else 'Jalankan / lanjutkan; hasil OK dilewati')
    for index,(step,_,output) in enumerate(plan,1):
        field(f'{index}. '+dict(TASKS)[step],output)
    if task=='all':print('Gemma dan GAN menggunakan original yang sama; GAN tidak menyerang hasil Gemma.')
    if mode=='full':print('Benchmark memerlukan revision/checkpoint model yang sebenarnya.')
    for index,(step,command,output) in enumerate(plan,1):
        print(f'\n--- Tahap {index}/{len(plan)}: {step.upper()} ---')
        code=run_command(command,config)
        if code:
            print('Tahap berikutnya tidak dijalankan. Gunakan menu 3 untuk melanjutkan.')
            return code
        field('Lokasi hasil',output)
    return 0


def workflow(path,config,selected,mode,retry=False):
    task=choose('Pilih pekerjaan',TASKS)
    if task is None:return
    print('\nBaseline diklasifikasikan lebih dahulu; hasil yang cocok dipakai ulang.')
    for step in (['baseline','llm','gan'] if task=='all' else ['baseline',task] if task!='baseline' else ['baseline']):
        field(step.upper(),destination(config,mode,step))
    action=choose('Tindakan',[('run','Lanjutkan + ulangi error' if retry else 'Jalankan / lanjutkan'),
                             ('dry','Periksa konfigurasi offline - tanpa API'),
                             ('check','Cek layanan / dependensi - tanpa klasifikasi')])
    if action:execute_plan(path,config,selected,mode,task,retry,action=='dry',action=='check')


def saved_runs(config):
    folders=set(destination(config,mode,task) for mode,task in OUTPUT_KEYS)
    base=pipeline.ROOT/('experimenting/outputs' if config['profile']=='experimenting' else 'outputs')
    if base.exists():
        folders.update(p.parent.parent for p in base.glob('*/*/run.json'))
    runs=[]
    for folder in sorted(folders):
        for path in sorted(folder.glob('*/run.json')):
            try:
                saved=json.loads(path.read_text(encoding='utf-8'))
                checkpoint=path.with_name('checkpoint.json')
                state=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {}
                runs.append((path,saved,state))
            except (ValueError,OSError) as exc:print('Tidak dapat membaca',path,':',exc)
    return runs


def results_menu(config):
    runs=saved_runs(config)
    if not runs:
        print('\nBelum ada hasil. Mulai dari menu 1: Pilot 10 data.');return
    options=[(i,f'{p.parent.parent.name} / {p.parent.name} - {c.get("state","?")} ({c.get("selected_count","?")} data)')
             for i,(p,s,c) in enumerate(runs)]
    index=choose('Pilih hasil yang sudah tersimpan',options)
    if index is None:return
    path,saved,state=runs[index]
    field('Folder',path.parent)
    field('Status',state.get('state','belum diketahui'))
    field('Tahap OK / error',f'{state.get("ok_stages",0)} / {state.get("error_stages",0)}')
    for name in ('sample_results.csv','step1_rewrites.csv','step2_rewrites.csv','gan_rewrites.csv','metrics.csv'):
        if path.with_name(name).exists():field('CSV',name)
    print('CSV per rewrite ada di rewrites/. Status ERROR perlu diperbaiki sebelum retry.')
    action=choose('Evaluasi dari hasil tersimpan; tidak mengulang model',
                  [('lexical','Evaluasi leksikal - tanpa download model embedding'),
                   ('full','Evaluasi lengkap + SBERT - membutuhkan model embedding')])
    if not action:return
    command=[sys.executable,str(HERE/'evaluate_paper.py'),'--run-dir',str(path.parent.parent),
             '--model',path.parent.name,'--dataset',str(pipeline.rooted(config['dataset']))]
    if action=='lexical':command.append('--lexical-only')
    run_command(command,config)


def save_profile(path,config):
    if path.suffix.lower()!='.json':raise ValueError('Simpan profil sebagai file .json.')
    if config['profile']=='experimenting' and not path.is_relative_to(EXPERIMENT.resolve()):
        raise ValueError('Profil lokal experimenting harus berada di folder experimenting.')
    if config['profile']=='main' and path.is_relative_to(EXPERIMENT.resolve()):
        raise ValueError('Profil lokal main harus terpisah dari experimenting.')
    validate_paths(config)
    from validity_pipeline import configuration
    configuration(config)
    pipeline.atomic_json(path,config)
    print('Pengaturan tersimpan. Perubahan model/endpoint memerlukan folder output baru.')


def settings(path,save_path,config,selected):
    options=[('select','Pilih defender'),('defender','Edit model / endpoint defender'),
             ('attacker','Edit model / endpoint attacker Gemma'),('evaluator','Edit endpoint evaluator PPLX'),('outputs','Ganti folder hasil'),
             ('gan','Bobot GAN dan seed'),('prepare','Siapkan bobot GAN')]
    while True:
        choice=choose('PENGATURAN | '+config['profile'],options)
        if choice is None:return path,config,selected
        candidate=copy.deepcopy(config)
        if choice=='select':
            name=choose('Defender',[*( (n,n+' - '+d.get('model','')) for n,d in config['defenders'].items()),
                                    ('all','Semua model aktif: '+', '.join(pipeline.selected_models(config,'all')))])
            if name is None:continue
            candidate['default_model']=name
        elif choice in {'defender','add'}:
            name=ask('Nama defender baru') if choice=='add' else choose('Pilih defender',[(n,n) for n in config['defenders']])
            if name is None:continue
            pipeline.validate_model_key(name)
            if choice=='add' and name in candidate['defenders']:raise ValueError('Nama sudah ada.')
            current=candidate['defenders'].get(name,dict(transport='direct',url='',model=name,revision='',key_env=''))
            candidate['defenders'][name]=edit_endpoint(current)
            if candidate['defenders'][name]['transport']=='direct' and name=='pplx' and not candidate['defenders'][name].get('template'):
                raise ValueError('Direct PPLX memerlukan template request terverifikasi.')
        elif choice=='attacker':candidate['attacker']=edit_endpoint(candidate['attacker'],True)
        elif choice=='evaluator':candidate['rewrite_evaluator']=edit_endpoint(candidate['rewrite_evaluator'])
        elif choice=='outputs':
            key=choose('Folder hasil',[ (OUTPUT_KEYS[m,t],m+' / '+t+' - '+str(destination(config,m,t)))
                      for m,t in [('pilot','baseline'),('pilot','llm'),('pilot','gan'),('full','llm'),('full','gan')]])
            if key is None:continue
            candidate[key]=ask('Folder baru (relatif terhadap root proyek)',config.get(key,''))
        elif choice=='gan':
            gan=candidate.setdefault('gan',{})
            gan['resource_dir']=ask('Folder bobot SGAN',gan.get('resource_dir','.cache/openattack_gan/AttackAssist.SGAN'))
            gan['seed']=int(ask('Seed',str(gan.get('seed',20261104))))
            if gan['seed']<0:raise ValueError('Seed harus >= 0.')
        elif choice=='prepare':
            print('Dependensi: python -m pip install -r scripts/requirements-gan.txt')
            folder=config.get('gan',{}).get('resource_dir','.cache/openattack_gan/AttackAssist.SGAN')
            command=[sys.executable,str(HERE/'prepare_gan.py'),'--output-dir',str(pipeline.rooted(folder))]
            local=ask('Folder bobot yang sudah ada; Enter untuk download resmi')
            if local:command+=['--from-dir',local]
            run_command(command,config);continue
        elif choice=='history':
            value=choose('Step 1 historis tidak setara dengan rewrite baru terkontrol',
                         [(False,'Buat rewrite baru (default)'),(True,'Pakai bank historis jika tersedia')])
            if value is None:continue
            candidate['reuse_historical_step1']=value
        save_profile(save_path,candidate)
        config=candidate;path=save_path;selected=config.get('default_model',selected)


def help_text(config):
    banner('CARA KERJA')
    print('1. Klasifikasi original. HAM dan spam yang lolos baseline tidak di-attack.')
    print('2. Gemma dan GAN: rewrite -> validasi PPLX -> hanya VALID diklasifikasikan.')
    print('3. Invalid berhenti; valid masih SPAM lanjut Step 2. Maksimal dua step per cabang.')
    print('4. CSV, waktu, dan checkpoint disimpan; hasil OK dilewati saat resume.')
    print('\nMulai: menu 1 -> Gemma + GAN -> Jalankan.')
    print('Lanjutkan error: menu 3 -> ukuran data yang sama -> pekerjaan yang sama.')
    print('Lihat CSV / evaluasi: menu 4. Ganti model / endpoint: menu 5.')
    print('API key: isi OWUI_API_KEY di .env root proyek; tidak ditampilkan di menu.')
    field('Panduan',pipeline.ROOT/('experimenting/README.md' if config['profile']=='experimenting' else 'README.md'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('main','experimenting'),default='main')
    parser.add_argument('--config',type=Path,help='Profil JSON; path relatif terhadap root proyek')
    parser.add_argument('--save-config',type=Path,help='File JSON lokal untuk perubahan menu')
    parser.add_argument('--show',action='store_true',help='Tampilkan profil saja; tanpa API')
    parser.add_argument('--mode',choices=('pilot','full'),default='pilot')
    parser.add_argument('--task',choices=('baseline','llm','gan','all'),help='Jalankan tanpa menu interaktif')
    parser.add_argument('--model',help='Defender: experimenting=pplx, main=clef/d3/all')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--check-server',action='store_true')
    parser.add_argument('--retry-errors',action='store_true')
    args=parser.parse_args(argv)
    if args.dry_run and args.check_server:parser.error('Pilih offline atau cek layanan, bukan keduanya.')
    if not args.task and (args.dry_run or args.check_server or args.retry_errors):parser.error('Gunakan --task bersama opsi eksekusi.')
    local=LOCAL if args.profile=='main' else EXPERIMENT/'adaptive_local.json'
    default=MAIN if args.profile=='main' else EXPERIMENT/'adaptive_config.json'
    path=pipeline.rooted(args.config) if args.config else local if local.exists() else default
    config=load_profile(path)
    if config['profile']!=args.profile:raise ValueError('Profil JSON tidak sesuai CLI yang dibuka.')
    validate_paths(config)
    selected=args.model or config.get('default_model','pplx')
    pipeline.selected_models(config,selected)
    save_path=pipeline.rooted(args.save_config) if args.save_config else local
    if args.show:show_profile(config,path,selected);return 0
    if args.task:return execute_plan(path,config,selected,args.mode,args.task,args.retry_errors,args.dry_run,args.check_server)
    while True:
        show_profile(config,path,selected)
        print('\n  1. Pilot 10 data - mulai di sini')
        print('  2. Benchmark seluruh 3.000 data')
        print('  3. Resume / ulangi tahap yang error')
        print('  4. Hasil CSV dan evaluasi')
        print('  5. Pengaturan model, endpoint, output, dan GAN')
        print('  6. Panduan singkat')
        print('  0. Keluar')
        try:
            choice=input('\nPilih nomor: ').strip()
            if choice=='0':return 0
            if choice in {'1','2'}:workflow(path,config,selected,'pilot' if choice=='1' else 'full')
            elif choice=='3':
                mode=choose('Ukuran run yang ingin dilanjutkan',[('pilot','Pilot 10 data'),('full','Benchmark seluruh dataset')])
                if mode:workflow(path,config,selected,mode,True)
            elif choice=='4':results_menu(config)
            elif choice=='5':path,config,selected=settings(path,save_path,config,selected)
            elif choice=='6':help_text(config)
            else:print('Pilih nomor yang tersedia.')
        except (ValueError,OSError,KeyError) as exc:print('\nTidak dapat melanjutkan:',exc)
        except KeyboardInterrupt:print('\nKembali ke menu. Hasil yang tersimpan tetap tersedia.')
        except EOFError:return 0


if __name__=='__main__':
    try:
        from env_config import load_project_env
        load_project_env()
        sys.exit(main())
    except (ValueError,OSError,KeyError) as exc:
        print(f'Konfigurasi tidak valid: {exc}',file=sys.stderr);sys.exit(2)
    except (EOFError,KeyboardInterrupt):sys.exit(130)
