import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class Detection:
    cls_name: str
    cls_id  : int
    conf    : float
    box     : Tuple[float, float, float, float]

    @property
    def cx(self): return (self.box[0] + self.box[2]) / 2
    @property
    def cy(self): return (self.box[1] + self.box[3]) / 2
    @property
    def w(self):  return self.box[2] - self.box[0]
    @property
    def h(self):  return self.box[3] - self.box[1]
    @property
    def area(self): return self.w * self.h


@dataclass
class SemanticGroup:
    group_type   : str
    elements     : List[Detection]
    anchor       : Optional[Detection] = None
    reading_order: int = 0

    @property
    def bbox(self):
        x1 = min(d.box[0] for d in self.elements)
        y1 = min(d.box[1] for d in self.elements)
        x2 = max(d.box[2] for d in self.elements)
        y2 = max(d.box[3] for d in self.elements)
        return x1, y1, x2, y2

    @property
    def cy(self):
        x1, y1, x2, y2 = self.bbox
        return (y1 + y2) / 2


class SALG:
    FLOAT_CLASSES   = {'Table', 'Picture', 'Formula'}
    FLOW_CLASSES    = {'Text', 'List-item'}
    HEADING_CLASSES = {'Section-header', 'Title'}
    ANCHOR_CLASSES  = {'Caption'}
    MARGIN_CLASSES  = {'Page-header', 'Page-footer', 'Footnote'}

    def __init__(self, img_h, img_w, col_gap_ratio=0.06, vert_gap_ratio=0.06,
                 caption_radius=0.20, caption_iou_x=0.30, float_merge_gap=0.03,
                 nms_iou_thresh=0.45, min_conf=0.35):
        self.img_h           = img_h
        self.img_w           = img_w
        self.col_gap         = col_gap_ratio   * img_w
        self.vert_gap        = vert_gap_ratio  * img_h
        self.caption_radius  = caption_radius  * img_h
        self.caption_iou_x   = caption_iou_x
        self.float_merge_gap = float_merge_gap * img_h
        self.nms_iou_thresh  = nms_iou_thresh
        self.min_conf        = min_conf

    def _nms(self, dets):
        if not dets: return dets
        from collections import defaultdict
        by_class = defaultdict(list)
        for d in dets: by_class[d.cls_name].append(d)
        kept = []
        for cls_dets in by_class.values():
            cls_dets = sorted(cls_dets, key=lambda d: -d.conf)
            suppressed = [False] * len(cls_dets)
            for i, d_i in enumerate(cls_dets):
                if suppressed[i]: continue
                kept.append(d_i)
                for j in range(i + 1, len(cls_dets)):
                    if not suppressed[j] and self._iou(d_i, cls_dets[j]) > self.nms_iou_thresh:
                        suppressed[j] = True
        return kept

    @staticmethod
    def _iou(a, b):
        ax1,ay1,ax2,ay2 = a.box; bx1,by1,bx2,by2 = b.box
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0.,ix2-ix1)*max(0.,iy2-iy1)
        if inter == 0: return 0.
        union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/union if union > 0 else 0.

    def _merge_floats(self, dets):
        if not dets: return dets
        merged_flag = [False]*len(dets)
        result = []
        dets = sorted(dets, key=lambda d: d.box[1])
        for i, d_i in enumerate(dets):
            if merged_flag[i]: continue
            group = [d_i]
            for j in range(i+1, len(dets)):
                if merged_flag[j]: continue
                d_j = dets[j]
                if d_j.cls_name != d_i.cls_name: continue
                h_overlap = min(d_i.box[2],d_j.box[2]) - max(d_i.box[0],d_j.box[0])
                if h_overlap < -self.img_w*0.05: continue
                v_gap = d_j.box[1] - max(g.box[3] for g in group)
                if v_gap < self.float_merge_gap:
                    group.append(d_j); merged_flag[j] = True
            x1=min(g.box[0] for g in group); y1=min(g.box[1] for g in group)
            x2=max(g.box[2] for g in group); y2=max(g.box[3] for g in group)
            best = max(group, key=lambda g: g.conf)
            result.append(Detection(best.cls_name, best.cls_id, best.conf, (x1,y1,x2,y2)))
        return result

    def _detect_columns(self, dets):
        if len(dets) < 4: return 1
        bins = 20
        edges = np.linspace(0, self.img_w, bins+1)
        counts, _ = np.histogram([d.cx for d in dets], bins=edges)
        mid_bin = bins//2
        central = counts[mid_bin - bins//6 : mid_bin + bins//6 + 1]
        flanks  = np.concatenate([counts[:mid_bin-bins//6], counts[mid_bin+bins//6+1:]])
        if len(central)==0 or len(flanks)==0: return 1
        valley_depth = np.mean(flanks) - np.min(central)
        return 2 if valley_depth > np.mean(counts)*0.4 else 1

    def _column_of(self, det, n_cols):
        if n_cols == 1: return 0
        return 0 if det.cx < self.img_w/2 else 1

    def _cluster_flow(self, dets):
        if not dets: return []
        sorted_dets = sorted(dets, key=lambda d: d.box[1])
        groups = [[sorted_dets[0]]]
        for det in sorted_dets[1:]:
            prev = groups[-1][-1]
            gap  = det.box[1] - prev.box[3]
            is_heading = det.cls_name in self.HEADING_CLASSES or prev.cls_name in self.HEADING_CLASSES
            if gap < self.vert_gap and not is_heading:
                groups[-1].append(det)
            else:
                groups.append([det])
        return groups

    def _anchor_captions(self, captions, floats):
        anchors = {}
        for i, cap in enumerate(captions):
            best_score, best_f = float('inf'), None
            for f in floats:
                v_dist = abs(cap.cy - f.cy)
                if v_dist >= self.caption_radius: continue
                h_overlap = min(cap.box[2],f.box[2]) - max(cap.box[0],f.box[0])
                h_ratio   = h_overlap / max(cap.w, 1e-3)
                if h_ratio < self.caption_iou_x: continue
                score = v_dist / (h_ratio + 1e-6)
                if score < best_score: best_score, best_f = score, f
            if best_f is not None: anchors[i] = best_f
        return anchors

    def _reading_key(self, g):
        x1,y1,x2,y2 = g.bbox
        cy = (y1+y2)/2
        if g.group_type == 'margin':
            return (-1,0,y1) if y1 < self.img_h*0.1 else (999,0,y1)
        col = 0 if (x2-x1) > self.img_w*0.60 else (0 if x1 < self.img_w/2 else 1)
        strip = int(cy / self.img_h * 20)
        return (strip, col, y1)

    def group(self, detections):
        detections = [d for d in detections if d.conf >= self.min_conf]
        detections = self._nms(detections)
        flow_dets    = [d for d in detections if d.cls_name in self.FLOW_CLASSES]
        heading_dets = [d for d in detections if d.cls_name in self.HEADING_CLASSES]
        float_dets   = [d for d in detections if d.cls_name in self.FLOAT_CLASSES]
        caption_dets = [d for d in detections if d.cls_name in self.ANCHOR_CLASSES]
        margin_dets  = [d for d in detections if d.cls_name in self.MARGIN_CLASSES]
        float_dets   = self._merge_floats(float_dets)
        groups = []
        all_text = flow_dets + heading_dets
        n_cols   = self._detect_columns(all_text)
        for col in range(n_cols):
            col_dets = [d for d in all_text if self._column_of(d, n_cols)==col]
            for cluster in self._cluster_flow(col_dets):
                types = {d.cls_name for d in cluster}
                gtype = 'title' if 'Title' in types else ('section' if 'Section-header' in types else 'flow')
                groups.append(SemanticGroup(group_type=gtype, elements=cluster))
        cap_anchors   = self._anchor_captions(caption_dets, float_dets)
        anchored_caps = set(cap_anchors.keys())
        for f in float_dets:
            elems = [f] + [caption_dets[ci] for ci,anch in cap_anchors.items() if anch is f]
            groups.append(SemanticGroup(group_type='float', elements=elems, anchor=f))
        for i, cap in enumerate(caption_dets):
            if i not in anchored_caps:
                groups.append(SemanticGroup(group_type='isolated_caption', elements=[cap]))
        for det in margin_dets:
            groups.append(SemanticGroup(group_type='margin', elements=[det]))
        groups.sort(key=self._reading_key)
        for i, g in enumerate(groups): g.reading_order = i
        return groups


def groups_to_json(groups, img_path=''):
    # Rename group types to match your preferred labels
    GROUP_REMAP = {
        'float'  : 'title',
        'title'  : 'text',
        'section': 'image',
    }
    return {
        'source_image': img_path,
        'total_groups': len(groups),
        'layout': [
            {
                'reading_order': g.reading_order,
                'type'         : GROUP_REMAP.get(g.group_type, g.group_type),
                'bbox'         : [round(v,2) for v in g.bbox],
                'anchor_class' : g.anchor.cls_name if g.anchor else None,
                'elements': [
                    {'class': e.cls_name, 'confidence': round(e.conf,4), 'bbox': [round(v,2) for v in e.box]}
                    for e in g.elements
                ],
            }
            for g in groups
        ],
    }
