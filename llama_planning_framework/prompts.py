def cot_prompt(prompt):
    return f"""{prompt}
Let's reason step by step to identify the correct sequence of actions to solve the problem."""


def planner_validator_prompt(prompt):
    return f"""{prompt}
Respond with only one action at a time. If your action is correct, you will receive "Valid" along with the updated state. If your action is incorrect, you will receive "Wrong" and must try a different action."""


def ppdl_problem_prompt(domanin, problem):
    return f"""{pddl_description}
 
=== DOMAIN DEFINITION ===
{domanin}
=== PROBLEM DEFINITION ===
{problem}

Can you BUILD A PLAN?
Do not include any narrative or explanation—output only the chosen actions for the PLAN.
The actions should have no ':action' prefix since it is in the domain definition and should not be outputted in the plan text.
"""


def backprompt(original_prompt, plan, reason):
    return f"""Your previous plan was INVALID. Here I give you the ORIGINAL PROMPT the PLAN you generated, and the ISSUE the validator found:
 
ORIGINAL PROMPT:
{original_prompt}

THE PLAN YOU GENERATED:
{plan}

The validator found the following ISSUE with your plan:
{reason}

Please generate a new plan that ADRRESSES this ISSUE.
"""


system_prompt = """
You are an intelligent planner using Partial Order Planning. Your input includes:
- A problem description
- An initial state
- A goal state
- A set of actions (with their preconditions and effects)

Maintain a current state as you select actions.
Only perform actions whose preconditions are satisfied.
Your output should be a coherent and valid sequence of actions that leads from the initial state to the goal state.
Do not include any narrative or explanation—output only plan.
"""

system_prompt_pddl = """You are an expert PDDL planner. Always output only valid PDDL domain and problem definitions.
Follow strict PDDL syntax for `:domain`, `:requirements`, `:types`, `:predicates`, `:actions`, `:objects`, `:init`, `:goal` and other PDDL predicates.
A PDDL model has two parts: a DOMAIN describing the schema and a PROBLEM describing a specific instance.
Maintain a current state as you select actions.
Only perform actions whose preconditions are satisfied.
Your output should be a coherent and valid sequence of actions that leads from the initial state `:init` to the goal state `:goal`.
Do not include any narrative or explanation—output only plan.
"""

system_prompt_pddl_updated = """You are an expert PDDL planner. Always output only valid PDDL actions adn follow strict PDDL syntax.
A PDDL model has two parts: a DOMAIN describing the schema and a PROBLEM describing a specific instance.

Your input includes:
- A domain description
- A problem description
- An initial state
- A goal state
- A set of actions (with their preconditions and effects)

Maintain a current state as you select actions.
Only perform actions whose preconditions are satisfied.
Your output should be a coherent and valid sequence of actions that leads from the initial state `:init` to the goal state `:goal`.
Do not include any narrative or explanation—output, but only the chosen actions for the plan.
The actions should have no ':action' prefix since it is in the domain definition and should not be outputted in the plan text.
"""

system_prompt_pddl_predicates = """
You are an expert PDDL planner. Always output only valid PDDL domain and problem definitions.
Follow strict PDDL syntax for `:domain`, `:requirements`, `:types`, `:predicates`, `:actions`, `:objects`, `:init`, `:goal` and other PDDL predicates.
A PDDL model has two parts: a DOMAIN describing the schema and a PROBLEM describing a specific instance. Include all possible operands:

;; === DOMAIN DEFINITION ===
(define (domain <DOMAIN-NAME>)
  (:requirements 
     :strips            ;; basic add/delete
     :typing            ;; types
     :adl               ;; negation, disjunction, quantifiers, conditional effects
     :fluents           ;; numeric functions
     :action-costs      ;; numeric action costs
  )
  (:types 
     ;; define type hierarchy, e.g.:
     ;; vehicle package - object
     ;; truck car - vehicle
  )
  (:constants 
     ;; e.g. depot1 depot2 - location
  )
  (:predicates 
     ;; boolean relations, e.g.:
     ;; (at ?v - vehicle ?l - location)
     ;; (in ?p - package ?v - vehicle)
  )
  (:functions 
     ;; numeric fluents, e.g.:
     ;; (fuel-level ?v - vehicle)
     ;; (total-cost)
  )
)
;; === PROBLEM DEFINITION ===
(define (problem <PROBLEM-NAME>)
  (:domain <DOMAIN-NAME>)
  (:objects 
     ;; list of all objects with types
     ;; e.g. truck1 truck2 - truck
  )
  (:init 
     ;; list initial atoms and (= (fuel-level truck1) 50)
  )
  (:goal 
     <GD> ;; goal description formula
  )
  (:metric 
     ;; e.g. (minimize (total-cost))
  )
)

Maintain a current state as you select actions.
Only perform actions whose preconditions are satisfied.
Your output should be a coherent and valid sequence of actions that leads from the initial state `:init` to the goal state `:goal`.
Do not include any narrative or explanation—output only plan.
"""


system_prompt_pddl_predicates_complex = """
You are an expert PDDL planner. Always output only valid PDDL domain and problem definitions.
Follow strict PDDL syntax for `:domain`, `:requirements`, `:types`, `:predicates`, `:actions`, `:objects`, `:init`, and `:goal`.
A PDDL model has two parts: a DOMAIN describing the schema and a PROBLEM describing a specific instance. Include all possible operands:

;; === DOMAIN DEFINITION ===
(define (domain <DOMAIN-NAME>)
  (:requirements 
     :strips            ;; basic add/delete
     :typing            ;; types
     :equality          ;; =
     :adl               ;; negation, disjunction, quantifiers, conditional effects
     :fluents           ;; numeric functions
     :durative-actions  ;; temporal actions
     :derived-predicates;; axioms
     :constraints       ;; trajectory constraints
     :action-costs      ;; numeric action costs
     :preferences       ;; soft constraints
     :object-fluents    ;; PDDL3.1 object-valued fluents
  )
  (:types 
     ;; define type hierarchy, e.g.:
     ;; vehicle package - object
     ;; truck car - vehicle
  )
  (:constants 
     ;; e.g. depot1 depot2 - location
  )
  (:predicates 
     ;; boolean relations, e.g.:
     ;; (at ?v - vehicle ?l - location)
     ;; (in ?p - package ?v - vehicle)
  )
  (:functions 
     ;; numeric fluents, e.g.:
     ;; (fuel-level ?v - vehicle)
     ;; (total-cost)
  )
  (:derived-predicates 
     ;; axioms, e.g.:
     ;; (:derived (reachable ?x - location ?y - location)
     ;;   (or (path ?x ?y) (exists (?z - location) (and (path ?x ?z) (reachable ?z ?y)))))
  )
  ;; instantaneous actions
  (:action <ACTION-NAME>
    :parameters (<typed list of variables>)
    :precondition <GD>    ;; goal description (possibly ADL)
    :effect <EF>          ;; effect formula (possibly conditional/quantified)
  )
  ;; durative (temporal) actions
  (:durative-action <DUR-ACTION-NAME>
    :parameters (<typed list>)
    :duration (= ?length <expr>)
    :condition (at start <GD>) (over all <GD>) (at end <GD>)
    :effect (at start <EF>) (at end <EF>) (over all <EF>)
  )
  ;; optional domain-wide constraints
  (:constraints 
     ;; e.g. (always (not (overlap ...)))
  )
)
;; === PROBLEM DEFINITION ===
(define (problem <PROBLEM-NAME>)
  (:domain <DOMAIN-NAME>)
  (:objects 
     ;; list of all objects with types
     ;; e.g. truck1 truck2 - truck
  )
  (:init 
     ;; list initial atoms and (= (fuel-level truck1) 50)
  )
  (:goal 
     <GD> ;; goal description formula
  )
  (:problem-constraints 
     ;; e.g. (within 10 (at-all trucks-at-depot))
  )
  (:metric 
     ;; e.g. (minimize (total-cost))
  )
  (:timed-initial-literals
     ;; e.g. (at 5 (road-blocked A B))
  )
)

Maintain a current state as you select actions.
Only perform actions whose preconditions are satisfied.
Your output should be a coherent and valid sequence of actions that leads from the initial state `:init` to the goal state `:goal`.
Do not include any narrative or explanation—output only plan.
"""


# Raw planning problem prompt for block‐world PDDL

# PPDL
pddl_description = """
Problem description: you are an agent capable of solving the planning problems described using the PDDL syntax.
In the following problem the agent can perform some actions (described later) to achieve a specific goal.
"""


block_world_prompt_pddl = """
Problem description: you are an agent capable of solving the planning problems described using the PDDL syntax.
In the following problem there is a table with some blocks and the agent can perform some actions (described later) to achieve a specific goal.
The problem domain is defined by: types (the type of the elements), predicates (statements that can be true in the current state) and actions (what the agent can do to modify the current state).
Types:
1. block
Predicates:
1. (on ?x - block ?y - block): means the block ?x is on the block ?y
2. (ontable ?x - block): means the block ?x is on the table
3. (clear ?x - block): means there is nothing on the block ?x
4. (handempty): the hand is empty
5. (holding ?x - block): the agent is holding the block ?x
Actions:
1. pick-up
   :parameters (?x - block)
   :precondition (and (clear ?x) (ontable ?x) (handempty))
   :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x))
2. put-down
   :parameters (?x - block)
   :precondition (holding ?x)
   :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x))
3. stack
   :parameters (?x - block ?y - block)
   :precondition (and (holding ?x) (clear ?y))
   :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y))
4. unstack
   :parameters (?x - block ?y - block)
   :precondition (and (on ?x ?y) (clear ?x) (handempty))
   :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))

Problem:
(:objects D B A C - block)
(:init (clear C) (clear A) (clear B) (clear D) (ontable C) (ontable A) (ontable B) (ontable D) (handempty))
(:goal (and (on D C) (on C B) (on B A)))

Can you BUILD A PLAN?
Do not include any narrative or explanation—output only the PLAN.
"""

# test prompts for finding _________________________________________________________________________________
socks_shoes_prompt_raw = """
I’m going to prompt here a problem in the style of Partial Order Programming, defining the initial state, the goal state, and the actions that you can perform to reach the goal state beginning from the initial state. 
The problem regards putting on both shoes and we will start with no shoes on. 
The actions we can perform are the following:
1. Putting on the right sock: there are no preconditions in this case and the effect of this action will be having the right sock on.
2. Putting on the left sock: there are no preconditions in this case and the effect of this action will be having the left sock on.
3. Putting on the right shoe: to perform this action you will need to have the right sock on and the effect of this action will be having the right shoe on.
4. Putting on the left shoe: to perform this action you will need to have the left sock on and the effect of this action will be having the left shoe on.

Can you BUILD A PLAN?
"""

shops_prompt_raw = """
I’m going to prompt here a problem in the style of Partial Order Programming, defining the initial state, the goal state, and the actions that you can perform to reach the goal state beginning from the initial state.
This is a problem of acquiring some items given their availability at certain stores.
In the beginning (the initial state) we are at home and we know that the shop called "SM" sells milk and banas while the shop called "HW" sells the drill. 
Our goal is to be at home and have the milk, the banana and the drill.
The actions we can perform are the following:
1. We can buy an item from a specific store: to perform this action we must be at the store and the store must have the item to sell. The effect of this action is that now we have the selected item.
2. We can go from a place to another place: the effect is that we are no more in the place we decided to move from but now we are in the new place.

Can you BUILD A PLAN?
"""
