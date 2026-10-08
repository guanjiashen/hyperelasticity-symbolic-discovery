from pathlib import Path
import io
from pypdf import PdfReader,PdfWriter
from reportlab.pdfgen import canvas
R=Path(__file__).resolve().parents[1]
for name,src in [('yohsuke',R/'figs/yohsuke_cann_predictions.pdf'),('brain',R/'simulation/brain_data/results/NN_output/ffbp_predictions.pdf')]:
 b=io.BytesIO();c=canvas.Canvas(b,pagesize=(576,432))
 c.setFillColorRGB(1,1,1);c.rect(0,75,28,285,fill=1,stroke=0)
 c.setFillColorRGB(0,0,0);c.saveState();c.translate(20,216);c.rotate(90);c.setFont('Helvetica',18)
 c.drawCentredString(0,0,'Nominal stress (kPa)' if name=='brain' else 'P11 (kPa)');c.restoreState()
 if name=='brain':
  c.setFillColorRGB(1,1,1);c.rect(220,0,230,30,fill=1,stroke=0);c.setFillColorRGB(0,0,0);c.setFont('Helvetica',16);c.drawCentredString(320,10,'Stretch / amount of shear')
 c.save();b.seek(0);p=PdfReader(src).pages[0];p.merge_page(PdfReader(b).pages[0]);w=PdfWriter();w.add_page(p)
 with (R/('figs/'+name+'_cann_predictions_kpa.pdf')).open('wb') as f:w.write(f)
