from pathlib import Path
import io
from PIL import Image, ImageDraw
from pypdf import PdfReader
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
R=Path(__file__).resolve().parents[1]
out=R/'figs/si_cann_loss_uniform.pdf'
c=canvas.Canvas(str(out),pagesize=(504,216))
W,H,Y=122,132,48
for j,(name,title) in enumerate([('vulcanized','Treloar'),('yohsuke','Bitoh'),('brain','Brain tissue')]):
 X=39+j*164
 c.setFont('Helvetica-Bold',9);c.drawString(X-23,199,'('+chr(97+j)+')');c.drawCentredString(X+W/2,199,title)
 if j==0:
  im=Image.open(io.BytesIO(PdfReader(R/'figs/vulcanized_cann_loss.pdf').pages[0].images[0].data)).convert('RGB')
  # Original 1600 x 1200 raster: crop to the axes interior, preserving all curve pixels.
  crop=im.crop((166,132,1570,984))
  ImageDraw.Draw(crop).rectangle((1090,10,1390,110),fill='white')
  pixels=crop.load()
  for yy in range(crop.height):
   for xx in range(crop.width):
    rgb=pixels[xx,yy]
    if max(rgb)-min(rgb)<35: pixels[xx,yy]=(255,255,255)
  c.drawImage(ImageReader(crop),X,Y,W,H)
  ticks=[(1,151),(0,281),(-1,410),(-2,540)]
  tickpositions=[(k,Y+H*(615-p)/534) for k,p in ticks]
  c.setStrokeColorRGB(.87,.87,.87);c.setLineWidth(.3)
  for _,pos in tickpositions: c.line(X,pos,X+W,pos)
  first,last=144/1000,944/1000
  xmin,xmax=(first-.103)/(.982-.103),(last-.103)/(.982-.103)
  N=200
 else:
  source=R/'simulation/brain_data/results/NN_output/ffbp_loss_curve.pdf' if name=='brain' else R/('figs/'+name+'_cann_loss.pdf')
  p=pdfplumber.open(source).pages[0]
  left,top,right,bottom=p.lines[-4]['x0'],10.8,565.2,390.04
  def xy(x,y):return X+(x-left)/(right-left)*W,Y+(bottom-y)/(bottom-top)*H
  majors=[l for l in p.lines if l['x1']-l['x0']>400][:3 if name=='brain' else 6]
  c.setStrokeColorRGB(.87,.87,.87);c.setLineWidth(.3)
  for l in majors:
   a,b=xy(l['x0'],l['top']);c.line(a,b,X+W,b)
  for curve in p.curves[:2]:
   c.setStrokeColorRGB(*curve['stroking_color']);c.setLineWidth(.85)
   path=c.beginPath()
   for i,(x,y) in enumerate(curve['pts']):
    a,b=xy(x,y)
    if i==0:path.moveTo(a,b)
    else:path.lineTo(a,b)
   c.drawPath(path)
  powers=list(range(-3,3)) if j==1 else [-3,-2,-1]
  tickpositions=[(k,xy(left,l['top'])[1]) for k,l in zip(powers,majors)]
  curve=p.curves[0]['pts'];xmin=(curve[0][0]-left)/(right-left);xmax=(curve[-1][0]-left)/(right-left);N=400
 c.setStrokeColorRGB(.3,.3,.3);c.setLineWidth(.5);c.rect(X,Y,W,H)
 c.setFillColorRGB(.1,.1,.1)
 for power,pos in tickpositions:
  c.setFont('Helvetica',7);c.drawRightString(X-10,pos-2,'10');c.setFont('Helvetica',5);c.drawString(X-9,pos+1,str(power))
 for v in range(0,N+1,N//4):
  xx=X+W*(xmin+(xmax-xmin)*v/N)
  c.setFont('Helvetica',7);c.drawCentredString(xx,Y-12,str(v));c.line(xx,Y,xx,Y-3)
 c.setFont('Helvetica',8);c.drawCentredString(X+W/2,Y-26,'L-BFGS outer iteration')
 c.saveState();c.translate(X-28,Y+H/2);c.rotate(90);c.drawCentredString(0,0,'Stress loss (kPa\xb2)' if j in (1,2) else 'Stress loss (MPa\xb2)');c.restoreState()
c.setFont('Helvetica',8)
for x,col,label in [(148,(.1216,.4667,.7059),'Training'),(265,(1,.498,.055),'Validation')]:
 c.setStrokeColorRGB(*col);c.setLineWidth(1);c.line(x,9,x+18,9);c.setFillColorRGB(.1,.1,.1);c.drawString(x+23,6,label)
c.save()
print(out)
