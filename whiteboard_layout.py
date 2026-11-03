"""Automatic horizontal layout for clean writing boards.

Keep original strokes. For clean horizontal writing, detect rows using the ink
mask and send the complete row directly to the recognizer. This avoids a text
detector cropping a radical, an outer border, a detached F bar or a faint digit.
Photographs/complex layouts return None so the existing detector can handle them.
There are no word replacements, language routing, or answer-specific rules.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _ink_mask(image: Image.Image):
    rgb=np.asarray(image.convert('RGB'))
    gray=np.asarray(image.convert('L'))
    # A channel distance also keeps yellow, cyan and red writing visible.
    white=np.min(rgb,axis=2)>240
    if float(white.mean())<.75:
        return None
    mask=(np.min(rgb,axis=2)<220).astype('uint8')
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if count<=1:return mask,stats[1:]
    keep=np.zeros(count,dtype='uint8')
    keep[1:]=(stats[1:,cv2.CC_STAT_AREA]>=5)
    mask=keep[labels]
    return mask,stats[1:][keep[1:]>0]


def line_regions(image: Image.Image):
    """Return rows in source coordinates, or None for a complex background."""
    ink=_ink_mask(image)
    if ink is None:return None
    mask,stats=ink
    points=np.argwhere(mask)
    if len(points)<8:return []
    y0,x0=points.min(0);y1,x1=points.max(0)+1
    heights=stats[:,cv2.CC_STAT_HEIGHT]
    areas=stats[:,cv2.CC_STAT_AREA]
    major=heights[(areas>=16)&(heights>=8)]
    glyph_height=float(np.percentile(major,80)) if len(major) else float(y1-y0)
    rows=np.flatnonzero(mask.any(1))
    cut=np.flatnonzero(np.diff(rows)>1)
    starts=np.r_[rows[0],rows[cut+1]]
    ends=np.r_[rows[cut]+1,rows[-1]+1]
    # Gaps inside detached Han components or an i/j are smaller than row gaps.
    # The estimate comes from connected-component geometry, not recognized text.
    min_gap=max(12,int(.55*glyph_height))
    bands=[]
    for a,b in zip(starts,ends):
        if bands and a-bands[-1][1]<=min_gap:
            bands[-1][1]=int(b)
        else:bands.append([int(a),int(b)])
    # Three thin horizontal components can be one square glyph rather than rows.
    if len(bands)<=3 and (x1-x0)<=1.8*(y1-y0) and all(b-a<12 for a,b in bands):
        bands=[[int(y0),int(y1)]]
    if len(bands)>12:return None
    regions=[]
    for a,b in bands:
        xs=np.flatnonzero(mask[a:b].any(0))
        if len(xs)==0:continue
        left,right=int(xs[0]),int(xs[-1]+1)
        h=b-a
        pad=max(8,min(32,round(.12*h)))
        region=(max(0,left-pad),max(0,a-pad),min(image.width,right+pad),min(image.height,b+pad))
        # Extremely wide rows only split at a substantial empty vertical stripe.
        # A cut through ink is forbidden; long continuous/cursive rows stay intact.
        if (right-left)/max(1,h)>16:
            columns=np.flatnonzero(mask[a:b,left:right].any(0))+left
            gaps=[(int(l+1),int(r)) for l,r in zip(columns[:-1],columns[1:]) if r-l-1>=.35*h]
            pos=left
            for g0,g1 in gaps:
                if g0-pos<8*h:continue
                cutx=(g0+g1)//2
                regions.append((max(0,pos-pad),region[1],min(image.width,cutx),region[3]))
                pos=cutx
            regions.append((max(0,pos),region[1],region[2],region[3]))
        else:regions.append(region)
    return regions


def monochrome_candidate(image: Image.Image):
    """Increase low-contrast/color visibility without changing stroke geometry."""
    rgb=np.asarray(image.convert('RGB'))
    gray=np.min(rgb,axis=2).astype('float32')
    ink=gray[gray<235]
    if len(ink)<8:return image.convert('RGB')
    black=float(np.percentile(ink,20))
    if black>=245:return image.convert('RGB')
    out=np.clip((gray-black)*255/max(1,255-black),0,255).astype('uint8')
    return Image.fromarray(out,'L').convert('RGB')


def recognize_whiteboard(engine,image: Image.Image,make_alternatives=False):
    """Call an existing RapidOCR object. Caller owns its inference lock.

    The original row is always primary. Optional contrast results are exposed as
    alternatives; a higher confidence never silently overwrites original text.
    Return None when layout is not suitable for the clean-board route.
    """
    regions=line_regions(image)
    if regions is None:return None
    lines=[]
    alternatives=[]
    for region in regions:
        crop=image.crop(region)
        output=engine(crop,use_det=False,use_cls=False,use_rec=True)
        texts=getattr(output,'txts',None)
        scores=getattr(output,'scores',None)
        if texts is None or scores is None:continue
        for text,score in zip(texts,scores):
            text=str(text).strip();score=float(score)
            if not text or not np.isfinite(score):continue
            left,top,right,bottom=region
            lines.append({'text':text,'confidence':round(min(1,max(0,score)),6),
                          'box':[[left,top],[right,top],[right,bottom],[left,bottom]]})
        if make_alternatives:
            other=engine(monochrome_candidate(crop),use_det=False,use_cls=False,use_rec=True)
            texts=getattr(other,'txts',None);scores=getattr(other,'scores',None)
            if texts is not None and scores is not None:
                alternatives.append({'region':region,'kind':'contrast_without_stroke_dilation',
                                     'texts':[str(t).strip() for t in texts],
                                     'scores':[float(s) for s in scores]})
    return {'lines':lines,'regions':regions,'alternatives':alternatives,
            'route':'clean_board_complete_rows_original_strokes'}
