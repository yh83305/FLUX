"""Small bottom-right timing footer; original RGB frame remains untouched."""
import math
import cv2
import numpy as np

def draw_frequency(image,timing=None):
    timing=timing or {}
    h,w=image.shape[:2]
    scale=min(.45,max(.25,w/2400.))
    suffix=' [warmup]' if timing.get('warmup') else ''
    lines=[]
    for key,label in [('generation_head_ms','Generation'),('selection_ms','Selection')]:
        ms=timing.get(key)
        value=(f'{1000/ms:.1f} Hz / {ms:.1f} ms' if isinstance(ms,(int,float)) and math.isfinite(ms) and ms>0 else 'N/A')
        lines.append(label+': '+value+suffix)
    # Dedicated footer guarantees no image, trajectory or existing legend is covered.
    margin=8;gap=5
    sizes=[cv2.getTextSize(s,cv2.FONT_HERSHEY_SIMPLEX,scale,1) for s in lines]
    line_height=max(size[0][1]+size[1] for size in sizes)+gap
    footer=2*margin+2*line_height
    footer+=(h+footer)%2
    result=np.zeros((h+footer,w,3),dtype=image.dtype)
    result[:h]=image
    for i,(line,(size,baseline)) in enumerate(zip(lines,sizes)):
        x=max(2,w-margin-size[0]);y=h+margin+i*line_height+size[1]
        cv2.putText(result,line,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(190,200,210),1,cv2.LINE_AA)
    return result
