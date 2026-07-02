"""
Prolog Reasoning Interface
============================
Updated to operate per-person: each call to reason() accepts a
single person dict {id, helmet, vest} and returns violations
tagged with that person's ID.

Strategies (auto-selected):
  1. PySwip  — in-process SWI-Prolog
  2. Subprocess — shells out to `swipl`
  3. Pure-Python fallback — always works, no Prolog required

Public API:
    reasoner = PrologReasoner()
    logs = reasoner.reason_all(persons)   # list[dict] – only violations
"""

import os
import subprocess
import tempfile
from typing import Dict, List

RULES_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "prolog", "rules.pl")
)


# ────────────────────────────────────────────────────────────────
# Rule definitions mirrored in Python (used by fallback + all engines)
# ────────────────────────────────────────────────────────────────

RULES = {
    "no_helmet": {
        "description":  "Helmet is mandatory in work zone",
        "severity":     "high",
        "rule_id":      "R-01",
    },
    "no_vest": {
        "description":  "Safety vest is required in work zone",
        "severity":     "medium",
        "rule_id":      "R-02",
    },
    "no_equipment": {
        "description":  "Person detected without ANY safety equipment — critical hazard",
        "severity":     "critical",
        "rule_id":      "R-03",
    },
}


# ────────────────────────────────────────────────────────────────
# Strategy 1: PySwip
# ────────────────────────────────────────────────────────────────

def _pyswip_reason(person_det: bool, helmet: bool, vest: bool) -> List[str]:
    """Return list of violated rule keys via PySwip."""
    from pyswip import Prolog  # type: ignore

    prolog = Prolog()
    prolog.consult(RULES_FILE.replace("\\", "/"))

    if person_det:
        prolog.assertz("person_detected")
    if helmet:
        prolog.assertz("wearing_helmet")
    if vest:
        prolog.assertz("wearing_vest")

    results = list(prolog.query("violation(V)"))
    return [str(r["V"]) for r in results]


# ────────────────────────────────────────────────────────────────
# Strategy 2: Subprocess (swipl)
# ────────────────────────────────────────────────────────────────

def _subprocess_reason(person_det: bool, helmet: bool, vest: bool) -> List[str]:
    """Return list of violated rule keys via swipl subprocess."""
    facts = []
    if person_det: facts.append(":- assert(person_detected).")
    if helmet:     facts.append(":- assert(wearing_helmet).")
    if vest:       facts.append(":- assert(wearing_vest).")

    script = f"""
:- consult('{RULES_FILE.replace(chr(92), "/")}').
{chr(10).join(facts)}
:- forall(violation(V), (write(V), nl)), halt.
:- halt.
"""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pl", delete=False, encoding="utf-8"
    )
    tmp.write(script)
    tmp.close()

    try:
        result = subprocess.run(
            ["swipl", "-q", "-f", tmp.name],
            capture_output=True, text=True, timeout=10,
        )
        return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    finally:
        os.unlink(tmp.name)


# ────────────────────────────────────────────────────────────────
# Strategy 3: Pure-Python fallback
# ────────────────────────────────────────────────────────────────

def _python_reason(person_det: bool, helmet: bool, vest: bool) -> List[str]:
    """Re-implementation of Prolog rules in pure Python."""
    violations = []
    if not person_det:
        return violations
    if not helmet and not vest:
        violations.append("no_equipment")
    if not helmet:
        violations.append("no_helmet")
    if not vest:
        violations.append("no_vest")
    return violations


# ────────────────────────────────────────────────────────────────
# Main Reasoner Class
# ────────────────────────────────────────────────────────────────

class PrologReasoner:
    """
    Symbolic safety reasoner.  Auto-selects the fastest available backend.

    Usage:
        logs = reasoner.reason_all(persons)
        # persons = [{"id": "A3F2K1", "helmet": False, "vest": True}, ...]
        # returns only entries where a rule is violated, each tagged with person_id
    """

    def __init__(self):
        self.engine = self._select_engine()
        print(f"[PrologReasoner] Engine: {self.engine}")

    # ── engine selection ────────────────────────────────────────

    def _select_engine(self) -> str:
        try:
            from pyswip import Prolog  # type: ignore
            _ = Prolog()
            return "pyswip"
        except Exception:
            pass

        try:
            r = subprocess.run(
                ["swipl", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return "subprocess"
        except Exception:
            pass

        return "python_fallback"

    # ── per-person reasoning ────────────────────────────────────

    def _reason_one(self, person_det: bool, helmet: bool, vest: bool) -> List[str]:
        """Return list of violation keys for one person."""
        try:
            if self.engine == "pyswip":
                return _pyswip_reason(person_det, helmet, vest)
            elif self.engine == "subprocess":
                return _subprocess_reason(person_det, helmet, vest)
        except Exception as e:
            print(f"[PrologReasoner] {self.engine} failed: {e}. Using fallback.")
        return _python_reason(person_det, helmet, vest)

    # ── public API ──────────────────────────────────────────────

    def reason_all(self, persons: List[Dict]) -> List[Dict]:
        """
        Evaluate safety rules for every tracked person.

        Args:
            persons: list of dicts with keys: id, helmet, vest
                     (person detected is implied by being in the list)

        Returns:
            List of violation log dicts (only rule breaches — compliant
            persons produce NO entry, keeping logs clean):
            [
              {
                "person_id": str,
                "violation": str,          # e.g. "no_helmet"
                "rule_id":   str,          # e.g. "R-01"
                "reason":    str,          # human-readable rule text
                "severity":  str,          # critical / high / medium
              },
              ...
            ]
        """
        log_entries = []

        for p in persons:
            pid    = p["id"]
            helmet = p.get("helmet", False)
            vest   = p.get("vest",   False)

            violation_keys = self._reason_one(
                person_det=True, helmet=helmet, vest=vest
            )

            for vkey in violation_keys:
                if vkey not in RULES:
                    continue
                rule = RULES[vkey]
                log_entries.append({
                    "person_id": pid,
                    "violation": vkey,
                    "rule_id":   rule["rule_id"],
                    "reason":    rule["description"],
                    "severity":  rule["severity"],
                })

        return log_entries

    # ── legacy compat (used by old callers) ─────────────────────

    def reason(self, detections: Dict) -> List[Dict]:
        """
        Backward-compatible single-scene reason call.
        Wraps reason_all with a synthetic person dict.
        """
        if not detections.get("person"):
            return [{
                "violation": "none",
                "reason": "No person detected.",
                "severity": "info",
            }]
        persons = [{
            "id":     "UNKNOWN",
            "helmet": detections.get("helmet", False),
            "vest":   detections.get("vest",   False),
        }]
        entries = self.reason_all(persons)
        if not entries:
            return [{
                "violation": "none",
                "reason": "All safety requirements met.",
                "severity": "ok",
            }]
        return [{"violation": e["violation"],
                 "reason": e["reason"],
                 "severity": e["severity"]} for e in entries]

    def get_engine_info(self) -> Dict:
        return {
            "engine":            self.engine,
            "rules_file":        RULES_FILE,
            "rules_file_exists": os.path.exists(RULES_FILE),
        }
