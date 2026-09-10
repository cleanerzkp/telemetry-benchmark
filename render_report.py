"""Render the two-page report from immutable saved predictions; no model inference."""
from pathlib import Path
import csv,json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from src.metrics import intervals
ROOT=Path(__file__).resolve().parent
NAMES={'rolling':'Średnia ruchoma','iforest':'Isolation Forest 1D','autoencoder':'Autoenkoder DAE'}

def main():
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    m=json.loads((ROOT/'results/metrics.json').read_text())
    f=json.loads((ROOT/'results/freeze.json').read_text())
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    cfg=json.loads((ROOT/'config.json').read_text());d=np.load(ROOT/'results/scores.npz')
    days=np.arange(len(d['labels']))/1440;channel=f['plot_channel'];truth=intervals(d['labels'])
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,1,figsize=(11.6,6.1),sharex=True,constrained_layout=True,
                          gridspec_kw={'height_ratios':[1.25,1,1,1]})
    axes[0].plot(days,d['telemetry'][:,channel],lw=.7,color='#193c63')
    axes[0].set_ylabel(f'Kanał {channel}\nwartość SMD')
    for n,ax in zip(NAMES,axes[1:]):
        ratio=d[n]/f['thresholds'][n];pred=ratio>1
        ax.plot(days,ratio,color='#193c63',lw=.65)
        ax.scatter(days[pred],ratio[pred],s=3,color='#19856b',zorder=3)
        ax.axhline(1,color='#b36b18',lw=1,ls='--')
        ax.set_yscale('symlog',linthresh=.1);ax.set_ylim(bottom=0)
        ax.set_ylabel({'rolling':'Rolling','iforest':'IForest 1D','autoencoder':'DAE'}[n]+'\nscore / próg')
    for ax in axes:
        for a,b in truth: ax.axvspan(a/1440,b/1440,color='#dd5765',alpha=.22)
        ax.grid(axis='y',alpha=.15)
    axes[-1].set_xlabel('Doby od początku ocenianego holdoutu (1 próbka = 1 min)')
    fig.savefig(out/'benchmark.png',dpi=190);plt.close(fig)
    rows=[['Metoda','Precision','Recall','F1','FA / dobę']]
    csvrows=[]
    for n,v in m['methods'].items():
        rows.append([NAMES[n],*[f'{v[k]:.3f}' for k in ['precision','recall','f1']],f"{v['false_alarms_per_day']:.2f}"])
        csvrows.append([n,v['precision'],v['recall'],v['f1'],v['false_alarms_per_day'],v['matched'],v['incidents'],v['alarm_episodes'],v['false_alarms'],v['duplicate_alarms']])
    with (out/'metrics_table.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['method','precision','recall','f1','false_alarms_per_day','matched','incidents','alarm_episodes','false_alarms','duplicates']);w.writerows(csvrows)
    baseline=f['selected_baseline'];b=m['methods'][baseline];ae=m['methods']['autoencoder'];vm=f['validation_metrics']
    relation='wyższe' if ae['f1']>b['f1'] else 'niższe' if ae['f1']<b['f1'] else 'takie samo'
    summary=(f"DAE uzyskał {relation} F1 niż baseline wybrany na walidacji: {ae['f1']:.3f} wobec {b['f1']:.3f}. "
             f"Dopasował {ae['matched']} z {ae['incidents']} incydentów przy {ae['false_alarms_per_day']:.2f} FA/dobę; "
             f"baseline dopasował {b['matched']} z {b['incidents']} przy {b['false_alarms_per_day']:.2f} FA/dobę. "
             'To obserwacja z jednego odłożonego bloku, nie dowód przewagi ogólnej ani gotowości wdrożeniowej.')
    counts='; '.join(f"{NAMES[n]}: {v['matched']}/{v['incidents']} trafień, {v['alarm_episodes']} epizodów, {v['false_alarms']} FA, {v['duplicate_alarms']} duplikatów" for n,v in m['methods'].items())+'.'
    limitation=('Publiczny zbiór nie odwzorowuje heterogenicznej telemetrii radiowej. Brak kontrolowanych scenariuszy degradacyjnych. '
        'Rozpoznanie służy weryfikacji metodyki i narzędzi, nie walidacji rozwiązania. Jedna maszyna, jeden seed i niewielka liczba incydentów '
        'nie pozwalają na wnioski o generalizacji. Ciągłe bloki SMD nie są niezależnymi przebiegami testbedu. '
        'Nie badano few-shot, grafów ani diagnostyki przyczynowej. Wyniki bazowe nie rozstrzygają problemu badawczego planowanego projektu.')
    font=font_manager.findfont('DejaVu Sans');bold=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))
    pdfmetrics.registerFont(TTFont('Body',font));pdfmetrics.registerFont(TTFont('BodyBold',bold))
    body=ParagraphStyle('body',fontName='Body',fontSize=9,leading=12.5,textColor=colors.HexColor('#26364a'),spaceAfter=7)
    small=ParagraphStyle('small',parent=body,fontSize=7.6,leading=10,spaceAfter=5)
    head=ParagraphStyle('head',parent=body,fontName='BodyBold',fontSize=11,leading=15,spaceBefore=7,spaceAfter=6,textColor=colors.HexColor('#193c63'))
    title=ParagraphStyle('title',parent=body,fontName='BodyBold',fontSize=22,leading=27,spaceAfter=9)
    story=[]
    def p(text,style=body):story.append(Paragraph(text,style))
    def h(text):p(text,head)
    p('TB-01  /  ROZPOZNANIE WSTĘPNE ML  /  10.09.2026',small)
    p('Detekcja anomalii<br/>w telemetrii serwerowej',title)
    p('Trzy metody. Jeden zbiór. Protokół ustalony przed wynikami.',head)
    h('Cel i zbiór')
    p('Sprawdzenie lokalnego procesu uczenia, kalibracji i oceny detektorów anomalii jako przygotowanie metodyczne do prac B+R. '
      'Wykorzystano publiczny SMD: wyłącznie machine-1-2, 38 kanałów i punktowe etykiety anomalii [1]. '
      'Poprzedni pilotaż machine-1-1 nie wchodzi do tego porównania.')
    h('Metody bez strojenia')
    p('<b>Rolling:</b> odchylenie od średniej z poprzednich 60 próbek, normalizowane skalą treningową, maksimum po kanałach. '
      '<b>IForest 1D:</b> 38 niezależnych jednozmiennowych lasów, parametry domyślne scikit-learn, seed 42; maksimum score. '
      '<b>DAE:</b> PyTorch, 38-32-8-32-38, ReLU, maskowanie 15%, MSE, Adam, 30 epok, batch 256. '
      'Jedna ustalona konfiguracja; brak optymalizacji hiperparametrów.')
    h('Podział i zamrożenie')
    r=manifest['ranges'];w=cfg['rolling_window']
    p(f"Plik treningowy: {r['fit'][1]:,} próbek do uczenia i {r['calibration'][1]-r['calibration'][0]:,} do kalibracji. "
      f"Plik z etykietami: {r['validation'][1]:,} na walidację i {r['holdout'][1]-r['holdout'][0]:,} na holdout. "
      f"Między blokami pozostawiono 120 próbek; pierwsze {w} próbek każdego bloku pominięto w ocenie jako kontekst. "
      'Nie losowano okien między zbiorami. Skaler dopasowano na treningu; progi to 99. percentyl score w kalibracji.')
    p(f"Wybrany baseline: <b>{NAMES[baseline]}</b>. F1 walidacji: Rolling {vm['rolling']['f1']:.3f}, IForest {vm['iforest']['f1']:.3f}. "
      'Wybór oraz progi zapisano w osobnym commicie przed jednorazowym odczytem holdoutu. Wyniki wszystkich metod raportuje się niezależnie od rankingu testowego.')
    h('Wynik końcowego holdoutu')
    p(f"{m['test_samples']:,} ocenionych próbek, {ae['exposure_days']:.2f} doby, {ae['incidents']} incydentów. Precision, recall i F1 dotyczą incydentów.",small)
    table=Table(rows,colWidths=[180,78,66,62,91],rowHeights=[26]+[25]*3)
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Body'),('FONTSIZE',(0,0),(-1,-1),9),
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#193c63')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#f0f4f8')),('ALIGN',(1,0),(-1,-1),'RIGHT'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),9),
        ('LINEBELOW',(0,0),(-1,0),.5,colors.white)]))
    story.append(table);story.append(Spacer(1,8));p(summary)
    story.append(PageBreak())
    p('TB-01  /  WYNIKI I OGRANICZENIA',small)
    h('Cały przebieg testowy i detekcje')
    story.append(Image(str(out/'benchmark.png'),width=477,height=251))
    p(f'Kanał {channel} (indeks od 0) wybrano przez najwyższą wariancję treningową, przed testem. '
      'Czerwone tło: incydenty. Zielone punkty: rzeczywiste przekroczenia progu. Linia przerywana: zamrożony próg. '
      'Oś score/prog jest symetrycznie logarytmiczna; pokazano pełny holdout.',small)
    h('Jak liczone są alarmy')
    p('Przekroczenia oddalone o nie więcej niż 30 próbek łączono w epizod bez korzystania z etykiet. '
      'Dopasowanie jeden-do-jednego wymaga rzeczywistego przekroczenia wewnątrz incydentu; przerw między alertami nie wypełniano trafieniami. '
      'Duplikaty obniżają precision. FA oznacza epizod bez trafienia w jakikolwiek incydent; FA/dobę = FA / czas całego ocenianego bloku. '
      'Przeliczenie czasu zakłada regularny interwał 1 min według [2]; CSV nie ma timestampów.')
    p(counts,small)
    h('Wnioski i ograniczenia')
    occupancy='; '.join(f"{NAMES[n]} {100*v['raw_alarm_fraction']:.2f}%" for n,v in m['methods'].items())
    p('Udział surowych próbek alarmowych: '+occupancy+'. '
      'Skuteczność trzeba oceniać razem z obciążeniem alarmami. Wynik nie uzasadnia automatycznego wyboru bardziej złożonej metody.')
    p(limitation)
    h('Odtwarzalność i źródła')
    p(f"Protokół: commit {f['protocol_commit'][:12]}. Seed 42, CPU, PyTorch {f['versions']['torch']}; "
      'hashe danych, kodu i modelu zapisano w manifest.json oraz freeze.json. Surowe score, alarmy i dopasowania znajdują się w pakiecie. '
      'Historia Git dokumentuje kolejność lokalną; nie jest niezależnym znacznikiem czasu.',small)
    p('[1] SMD / OmniAnomaly: <link href="https://github.com/NetManAIOps/OmniAnomaly">github.com/NetManAIOps/OmniAnomaly</link>, '
      'commit 7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4.<br/>'
      '[2] Li i in., <link href="https://netman.aiops.org/wp-content/uploads/2024/08/Empirical_Analysis.pdf">An Empirical Analysis of Anomaly Detection Methods for Multivariate Time Series</link>, tabela I (interwał SMD).',small)
    def footer(can,doc):
        can.setStrokeColor(colors.HexColor('#d9e1eb'));can.line(59,39,536,39)
        can.setFont('Body',8);can.setFillColor(colors.HexColor('#62748b'))
        can.drawString(59,26,'TELEMETRY BENCHMARK  |  TB-01  |  RAPORT TECHNICZNY')
        can.drawRightString(536,26,str(doc.page))
    doc=SimpleDocTemplate(str(out/'RAPORT.pdf'),pagesize=(595.28,841.89),rightMargin=59,leftMargin=59,topMargin=36,bottomMargin=51,
                          title='TB-01 - Detekcja anomalii w telemetrii serwerowej',author='Telemetry Benchmark')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    md='''# TB-01: Detekcja anomalii w telemetrii serwerowej\n\n'''+summary+'\n\n'
    md+='| Metoda | Precision | Recall | F1 | FA/dobę |\n|---|---:|---:|---:|---:|\n'
    md+='\n'.join('| '+' | '.join(row)+' |' for row in rows[1:])+'\n\n'+counts+'\n\n![Pełny holdout](benchmark.png)\n\n'+limitation
    md+='\n\nPełna metodyka: ../PROTOCOL.md. Szczegóły i źródła: RAPORT.pdf.\n'
    (out/'RAPORT.md').write_text(md,encoding='utf8')
    print('Rendered',out/'RAPORT.pdf')
if __name__=='__main__':main()
