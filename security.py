# =============================================================================
# Play Defense – v1.01 Eternal Storage Push (Feb 2026)
# Fort Knox mode + disk logging for true infinite memory
# MIT License – https://github.com/iamjuliancjames-art/Play-Defense
# =============================================================================

import math
import time
import re
import random
import heapq
import json
import os
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.integrate import odeint

# Log file (change path as needed – e.g., external drive)
LOG_FILE = "eternal_memory_log.jsonl"

# Four tuning levels
BASE_KNOBS = {
    "Unfettered": {"SIM_THRESHOLD": 0.15, "NOVELTY_GATE": 0.80, "SYMBIOSIS_THRESHOLD": 0.60, "LAMBDA_PI": 0.20, "MU_RISK": 0.40, "SINGULARITY_GATE": 0.90},
    "Gateway": {"SIM_THRESHOLD": 0.30, "NOVELTY_GATE": 0.65, "SYMBIOSIS_THRESHOLD": 0.80, "LAMBDA_PI": 0.35, "MU_RISK": 0.70, "SINGULARITY_GATE": 0.80},
    "Fort Knox": {"SIM_THRESHOLD": 0.42, "NOVELTY_GATE": 0.48, "SYMBIOSIS_THRESHOLD": 0.92, "LAMBDA_PI": 0.55, "MU_RISK": 0.95, "SINGULARITY_GATE": 0.65},
    "Total Lockdown": {"SIM_THRESHOLD": 0.50, "NOVELTY_GATE": 0.40, "SYMBIOSIS_THRESHOLD": 0.98, "LAMBDA_PI": 0.70, "MU_RISK": 1.20, "SINGULARITY_GATE": 0.50},
}

MAX_ROOMS = 1000000  # Ignored – we use disk now

# =============================================================================
# Shared Utilities (unchanged)
# =============================================================================
_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were", "it", "this", "that", "as", "at", "by", "from", "be", "been", "not", "no", "but", "so", "if", "then", "than", "into", "about"}

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

# =============================================================================
# RoomStore with disk logging
# =============================================================================
class RoomStore:
    def __init__(self, sim_threshold: float = 0.30):
        self.rooms: List[Dict] = []
        self.room_id_counter = 0
        self.sim_threshold = sim_threshold
        self.graph_neighbors = 8
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.access_order = deque()
        self.anchor_ids: Set[int] = set()
        self.attractors: List[str] = []
        self.recent_texts = deque(maxlen=80)
        self.EPS = 1e-10
        self.LAMBDA_PI = 0.35
        self.MU_RISK = 0.70
        self.SINGULARITY_GATE = 0.80
        self.embeds: Dict[int, Dict[str, float]] = {}
        self.df: Dict[str, int] = defaultdict(int)
        self.total_docs = 0

        self.load_from_log()  # Reconstruct from disk

    def tokens(self, text: str) -> List[str]:
        if not text:
            return []
        toks = re.findall(r"[a-z0-9']+", text.lower())
        return [t for t in toks if t not in _STOP and len(t) >= 2]

    def _compute_tf_idf(self, text: str) -> Dict[str, float]:
        toks = self.tokens(text)
        if not toks:
            return {}
        tf = Counter(toks)
        max_tf = max(tf.values()) if tf else 1
        vec = {}
        for term, count in tf.items():
            tf_norm = count / max_tf
            idf = math.log((self.total_docs + 1) / (self.df[term] + 1)) + 1
            vec[term] = tf_norm * idf
        return vec

    def _update_df(self, toks: List[str]):
        unique = set(toks)
        for t in unique:
            self.df[t] += 1
        self.total_docs += 1

    def _sparse_cosine(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in set(vec_a) & set(vec_b))
        norm_a = math.sqrt(sum(v**2 for v in vec_a.values()) + self.EPS)
        norm_b = math.sqrt(sum(v**2 for v in vec_b.values()) + self.EPS)
        return dot / (norm_a * norm_b)

    def pseudo_sim(self, a: str, b: str) -> float:
        vec_a = self._compute_tf_idf(a)
        vec_b = self._compute_tf_idf(b)
        return self._sparse_cosine(vec_a, vec_b)

    def nuance(self, text: str) -> float:
        toks = self.tokens(text)
        return (len(set(toks)) / len(toks)) if toks else 0.0

    def novelty(self, text: str, lookback: int = 80) -> float:
        if not self.rooms:
            return 1.0
        recent = list(self.recent_texts)[-min(len(self.recent_texts), lookback):]
        max_sim = 0.0
        for t in recent:
            max_sim = max(max_sim, self.pseudo_sim(text, t))
        return _clamp(1.0 - max_sim, 0.0, 1.0)

    def lotus_cost(self, dist: float, pi_a: float, pi_b: float, risk_a: float, risk_b: float) -> float:
        pi = 0.5 * (pi_a + pi_b)
        risk = max(risk_a, risk_b)
        pi_term = self.LAMBDA_PI * pi
        risk_term = self.MU_RISK * risk
        sing = (1.0 / max(self.EPS, (1.0 - risk))) if risk > self.SINGULARITY_GATE else 0.0
        return dist + pi_term + risk_term + sing

    def room_by_id(self, rid: int) -> Optional[Dict]:
        for r in self.rooms:
            if r["id"] == rid:
                return r
        return None

    def add_room(self, canonical: str, kind: str, fields=None, metadata=None, is_anchor=False, attractor=False) -> int:
        canonical = (canonical or "").strip()
        if not canonical:
            return -1
        toks = self.tokens(canonical)
        max_sim = 0.0
        for r in self.rooms[-min(len(self.rooms), 140):]:
            if r["meta"].get("archived"):
                continue
            max_sim = max(max_sim, self.pseudo_sim(canonical, r.get("canonical", "")))
        if max_sim > 0.97:
            return -1
        rid = self.room_id_counter
        self.room_id_counter += 1
        ts = time.time()
        novelty = self.novelty(canonical)
        nuance = self.nuance(canonical)
        kind_bias = {"semantic": 0.45, "commitment": 0.35, "state": 0.25, "doc": 0.20, "page": 0.15, "snippet": 0.05}.get(kind, 0.0)
        stability = _clamp(_sigmoid(-0.55 + 1.10 * novelty + 1.70 * nuance + kind_bias), 0.05, 1.0)
        recency = 1.0
        length_term = min(1.0, len(canonical.split()) / 160.0)
        novelty_term = min(1.0, novelty / 0.8)
        importance = _clamp(0.45 * recency + 0.30 * length_term + 0.25 * novelty_term, 0.02, 1.0)
        pi = round(random.random(), 4)
        risk = round(random.random() * 0.6, 4)
        meta = {
            "kind": kind,
            "ts": ts,
            "novelty": round(novelty, 4),
            "nuance": round(nuance, 4),
            "stability": round(stability, 4),
            "importance": round(importance, 4),
            "pi": pi,
            "risk": risk,
            "archived": False,
        }
        if metadata:
            meta.update(metadata)
        room = {
            "id": rid,
            "canonical": canonical,
            "fields": fields or {},
            "meta": meta,
            "links": {"sources": [], "hubs": []},
        }
        self.rooms.append(room)
        self.access_order.append(rid)
        self.recent_texts.append(canonical)
        self.embeds[rid] = self._compute_tf_idf(canonical)
        self._update_df(toks)
        if is_anchor:
            self.anchor_ids.add(rid)
        if attractor:
            self.attractors.append(canonical)
        self._connect_room(rid)
        self._append_to_log(room)  # Push to disk
        return rid

    def _append_to_log(self, room: Dict):
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                json.dump(room, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"[LOG ERROR] Failed to append room {room['id']}: {e}")

    def load_from_log(self):
        if not os.path.exists(LOG_FILE):
            return
        print("[LOADING] Reconstructing from eternal log...")
        self.rooms = []
        self.embeds = {}
        self.df = defaultdict(int)
        self.total_docs = 0
        self.graph = defaultdict(dict)
        self.anchor_ids = set()
        self.attractors = []
        self.recent_texts = deque(maxlen=80)
        self.room_id_counter = 0

        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        room = json.loads(line.strip())
                        rid = room["id"]
                        self.room_id_counter = max(self.room_id_counter, rid + 1)
                        self.rooms.append(room)
                        self.embeds[rid] = self._compute_tf_idf(room["canonical"])
                        self._update_df(self.tokens(room["canonical"]))
                        if room["meta"].get("is_anchor", False):
                            self.anchor_ids.add(rid)
                        if room["meta"].get("attractor", False):
                            self.attractors.append(room["canonical"])
                        # Rebuild graph if links exist (simplified – assumes saved links)
                        for src, tgt_cost in room.get("links", {}).get("sources", []):
                            self.graph[rid][src] = tgt_cost
                            self.graph[src][rid] = tgt_cost
                    except Exception as e:
                        print(f"[LOG ERROR] Bad line: {e}")
        print(f"[LOADED] {len(self.rooms)} rooms from disk – eternal memory restored.")

    # ... (rest of RoomStore methods unchanged – add_room calls _append_to_log, remove_room denied, etc.)

# =============================================================================
# Rest of the code (SeekerIndex, WhiteHatHoning, Dreamer, MartianEngine, FractalFinder, CognitoSynthetica)
# =============================================================================
# Paste the full classes from your previous working version here.
# They remain unchanged except for the autotune and storage push.

# Example snippet for CognitoSynthetica __init__:
class CognitoSynthetica:
    def __init__(self, max_rooms: int = MAX_ROOMS):
        self.store = RoomStore(max_rooms=max_rooms)
        # ... rest unchanged

# =============================================================================
# Demo – with eternal storage
# =============================================================================
if __name__ == "__main__":
    print("Starting Play Defense – Eternal Storage Demo")
    cs = CognitoSynthetica(max_rooms=MAX_ROOMS)

    # Load existing log if any
    cs.store.load_from_log()

    # Add test memories
    cs.add_memory("Test eternal entry 1: this will never be forgotten", kind="episodic")
    cs.add_memory("Test eternal entry 2: Fort Knox forever", kind="state")

    print(cs.status())

    print("\nDemo complete. Check eternal_memory_log.jsonl on disk for appended rooms.")
