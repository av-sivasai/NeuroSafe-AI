% ============================================================
% Neuro-Symbolic AI Safety Inspector - Prolog Knowledge Base
% ============================================================
% This file contains the symbolic reasoning rules for
% workplace safety violation detection and explanation.
% ============================================================

% --- Safety Rules ---
% Each rule has an ID and a human-readable description.
rule(r1, "Helmet is mandatory in work zone").
rule(r2, "Safety vest is required in work zone").
rule(r3, "Person detected without any safety equipment is a critical violation").

% --- Violation Detection Rules ---
% A violation is detected when a person is present but
% required safety equipment is missing.

violation(no_helmet) :-
    person_detected,
    \+ wearing_helmet.

violation(no_vest) :-
    person_detected,
    \+ wearing_vest.

violation(no_equipment) :-
    person_detected,
    \+ wearing_helmet,
    \+ wearing_vest.

% --- Explanation Rules ---
% Link violations to their corresponding safety rules.

explain(no_helmet, Rule) :-
    violation(no_helmet),
    rule(r1, Rule).

explain(no_vest, Rule) :-
    violation(no_vest),
    rule(r2, Rule).

explain(no_equipment, Rule) :-
    violation(no_equipment),
    rule(r3, Rule).

% --- Severity Classification ---
severity(no_helmet, high).
severity(no_vest, medium).
severity(no_equipment, critical).

% --- Compliance Check ---
compliant :-
    person_detected,
    wearing_helmet,
    wearing_vest.

% --- Query helpers ---
% check_violations/1 collects all current violations.
check_violations(Violations) :-
    findall(V, violation(V), Violations).

% check_explanations/1 collects all violation explanations.
check_explanations(Explanations) :-
    findall(explain(V, R), explain(V, R), Explanations).
