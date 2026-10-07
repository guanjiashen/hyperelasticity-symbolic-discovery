from pathlib import Path
import io
from pypdf import PdfReader,PdfWriter
from reportlab.pdfgen import canvas
R=Path(__file__).resolve().parents[1]
source=R/'figs/vulcanized_cann_predictions.pdf'
dest=R/'figs/vulcanized_cann_predictions_uniform.pdf'
b=io.BytesIO();c=canvas.Canvas(b,pagesize=(576,432))
# Cover the old legend, entirely inside the new legend's footprint.
c.setFillColorRGB(1,1,1);c.rect(50,432-85,203,70,stroke=0,fill=1)
# Match the 0.8 pt axes used in panels (b) and (c).
c.setStrokeColorRGB(0,0,0);c.setLineWidth(.8);c.rect(46.4,51.752,518.8,369.448,stroke=1,fill=0)
# Identical 13 pt, two-column legend; source curves and measurements unchanged.
left,top=52.9,414.7
c.setStrokeColorRGB(.8,.8,.8);c.setLineWidth(.8);c.setFillColorRGB(1,1,1)
c.roundRect(left,top-99.29,281.59,99.29,2,stroke=1,fill=1)
blue=(.1215686,.4666667,.7058824);orange=(1,.4980392,.054902);green=(.172549,.627451,.172549)
entries=[(0,0,blue,'train','UT train'),(0,1,blue,'test','UT test'),(0,2,blue,'line','UT prediction'),(0,3,orange,'train','PS train'),(0,4,orange,'test','PS test'),(1,0,orange,'line','PS prediction'),(1,1,green,'train','ET train'),(1,2,green,'test','ET test'),(1,3,green,'line','ET prediction')]
for col,row,color,kind,label in entries:
 x=left+18.5+col*149.3;y=top-11-row*18.55
 c.setStrokeColorRGB(*color);c.setFillColorRGB(*color)
 if kind=='line':c.setLineWidth(1.5);c.line(x-14,y,x+14,y)
 elif kind=='test':
  c.setLineWidth(1.5);c.line(x-4,y-4,x+4,y+4);c.line(x-4,y+4,x+4,y-4)
 else:
  c.setFillColorRGB(*[.2+.8*v for v in color]);c.setLineWidth(.8);c.circle(x,y,3.6,stroke=1,fill=1)
 c.setFillColorRGB(0,0,0);c.setFont('Helvetica',13);c.drawString(x+23.5,y-4,label)
c.save();b.seek(0)
p=PdfReader(source).pages[0];p.merge_page(PdfReader(b).pages[0]);w=PdfWriter();w.add_page(p)
with dest.open('wb') as f:w.write(f)
print(dest)
