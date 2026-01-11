
import time
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine

class RuntimeGallery:
    def __init__(self, max_samples=10, expiration_seconds=120):
        self.max_samples = max_samples
        self.expiry = expiration_seconds
        
        # Structure:
        # {
        #   "PersonName": {
        #       "body_feats": [],
        #       "face_feats": [],
        #       "ema_body": None,
        #       "ema_face": None,
        #       "last_seen": timestamp,
        #       "confidence": 0.5 (starts neutral)
        #   }
        # }
        self.gallery = {}

    def _cosine_sim(self, v1, v2):
        if v1 is None or v2 is None:
            return 0.0
        # Ensure vectors are normalized
        v1 = v1.reshape(-1)
        v2 = v2.reshape(-1)
        # 1 - cosine_dist = cosine_sim
        return 1.0 - cosine(v1, v2)

    def update(self, name, body_feat=None, face_feat=None, locked=False, similarity=1.0, is_occluded=False):
        """
        Update the gallery for a known identity.
        Rules:
        - Update only if Locked, Similarity > 0.75, Not Occluded.
        - EMA = 0.8 * old + 0.2 * new
        """
        # Pollution prevention
        if is_occluded:
            return
        if similarity < 0.65:
            return  # Never update if weak similarity

        if name not in self.gallery:
            self.gallery[name] = {
                "body_feats": [],
                "face_feats": [],
                "ema_body": None,
                "ema_face": None,
                "last_seen": time.time(),
                "confidence": 0.5
            }

        entry = self.gallery[name]
        entry["last_seen"] = time.time()

        # Confidence update (user rule: +0.1 when matched)
        # Note: 'update' implies a match occurred or a lock is active
        entry["confidence"] = min(2.0, entry["confidence"] + 0.1)

        # Update Features (Strict Condition)
        # Update only if Similarity > 0.75 OR It's a forceful update (e.g. initial Lock)
        if locked or similarity > 0.75:
            
            # Update Body
            if body_feat is not None:
                # Add raw
                if len(entry["body_feats"]) >= self.max_samples:
                    entry["body_feats"].pop(0) # Drop oldest
                entry["body_feats"].append(body_feat)
                
                # Update EMA
                if entry["ema_body"] is None:
                    entry["ema_body"] = body_feat
                else:
                    entry["ema_body"] = 0.8 * entry["ema_body"] + 0.2 * body_feat

            # Update Face
            if face_feat is not None:
                # Add raw
                if len(entry["face_feats"]) >= self.max_samples:
                    entry["face_feats"].pop(0)
                entry["face_feats"].append(face_feat)
                
                # Update EMA
                if entry["ema_face"] is None:
                    entry["ema_face"] = face_feat
                else:
                    entry["ema_face"] = 0.8 * entry["ema_face"] + 0.2 * face_feat

    def compute_score(self, name, probe_feat, is_face=False):
        """
        Compute matching score for a specific identity.
        score = max( max(sim(new, raw)), sim(new, ema) )
        """
        if name not in self.gallery:
            return 0.0
            
        entry = self.gallery[name]
        
        feats_key = "face_feats" if is_face else "body_feats"
        ema_key = "ema_face" if is_face else "ema_body"
        
        raw_feats = entry[feats_key]
        ema_feat = entry[ema_key]
        
        if not raw_feats:
            return 0.0

        # Sim check
        max_raw_sim = 0.0
        for rf in raw_feats:
            s = self._cosine_sim(probe_feat, rf)
            if s > max_raw_sim:
                max_raw_sim = s
        
        ema_sim = 0.0
        if ema_feat is not None:
            ema_sim = self._cosine_sim(probe_feat, ema_feat)
            
        return max(max_raw_sim, ema_sim)

    def get_best_match(self, body_feat=None, face_feat=None):
        """
        Optimized Search: 
        1. Fast checks using EMA first. 
        2. Detailed checks only if necessary.
        """
        best_name = None
        best_score = 0.0
        match_type = "none"

        # Check expiry before searching (Periodic cleanup, not every frame for speed?)
        # For max speed, call cleanup() externally on a schedule, not here.
        # self.cleanup() 

        for name, entry in self.gallery.items():
            # Confidence check
            if entry["confidence"] < 0.2:
                continue

            # ---------------------------------------------------------
            # FAST PATH: Check vs EMA first (Centroid)
            # ---------------------------------------------------------
            # This avoids iterating through raw buffers if the centroid match is either
            # very good (distinct match) or very bad (distinct non-match).
            
            s_body_ema = 0.0
            if body_feat is not None and entry["ema_body"] is not None:
                s_body_ema = self._cosine_sim(body_feat, entry["ema_body"])
                
            s_face_ema = 0.0
            if face_feat is not None and entry["ema_face"] is not None:
                s_face_ema = self._cosine_sim(face_feat, entry["ema_face"])
                
            # Weighted Combo of EMA
            # Strategy: If EMA score is very high (>0.85), trust it immediately.
            # If intermediate (0.6 ~ 0.8), dig deeper into raw buffer.
            
            score_f_ema = s_face_ema * 1.2
            score_b_ema = s_body_ema * 1.0
            current_max_ema = max(score_f_ema, score_b_ema)
            
            final_p_score = 0.0
            p_type = "none"
            
            if current_max_ema > 0.85:
                # Fast Accept
                final_p_score = current_max_ema
                p_type = "runtime_face" if score_f_ema > score_b_ema else "runtime_body"
            elif current_max_ema < 0.4:
                # Fast Reject
                final_p_score = current_max_ema
            else:
                # ---------------------------------------------------------
                # SLOW PATH: Ambiguous EMA, check specific raw poses
                # ---------------------------------------------------------
                s_body_raw = 0.0
                if body_feat is not None:
                     s_body_raw = self.compute_score(name, body_feat, is_face=False)
                
                s_face_raw = 0.0
                if face_feat is not None:
                     s_face_raw = self.compute_score(name, face_feat, is_face=True)
                     
                score_f_raw = s_face_raw * 1.2
                score_b_raw = s_body_raw * 1.0
                
                final_p_score = max(score_f_raw, score_b_raw)
                p_type = "runtime_face" if score_f_raw > score_b_raw else "runtime_body"
            
            
            if final_p_score > best_score:
                best_score = final_p_score
                best_name = name
                match_type = p_type
                
        return best_name, best_score, match_type

    def cleanup(self):
        """
        Remove expired identities.
        - Time > 120s
        - Confidence < 0.2
        """
        to_remove = []
        now = time.time()
        for name, entry in self.gallery.items():
            if (now - entry["last_seen"]) > self.expiry:
                to_remove.append(name)
            elif entry["confidence"] < 0.2:
                to_remove.append(name)
        
        for name in to_remove:
            del self.gallery[name]
            
    def get_confidence(self, name):
        if name in self.gallery:
            return self.gallery[name]["confidence"]
        return 0.0
        
    def decrement_confidence(self, name):
        if name in self.gallery:
             self.gallery[name]["confidence"] -= 0.05
