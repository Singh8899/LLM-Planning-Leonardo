(define (domain shoe-socks)
  (:requirements :strips :typing)
  (:types person object - object sock - object shoe - object)
  
  (:predicates
    (has-sock ?p - person ?s - sock)
    (has-shoe ?p - person ?sh - shoe)
    (wearing-sock ?p - person ?s - sock)
    (wearing-shoe ?p - person ?sh - shoe)
    (sock-pair ?s1 - sock ?s2 - sock)
  )
  
  (:action pick-up-sock
    :parameters (?p - person ?s - sock)
    :precondition (not (has-sock ?p ?s))
    :effect (has-sock ?p ?s)
  )
  
  (:action pick-up-shoe
    :parameters (?p - person ?sh - shoe)
    :precondition (not (has-shoe ?p ?sh))
    :effect (has-shoe ?p ?sh)
  )
  
  (:action wear-sock
    :parameters (?p - person ?s - sock)
    :precondition (and (has-sock ?p ?s) (not (wearing-sock ?p ?s)))
    :effect (wearing-sock ?p ?s)
  )
  
  (:action wear-shoe
    :parameters (?p - person ?sh - shoe)
    :precondition (and (has-shoe ?p ?sh) (not (wearing-shoe ?p ?sh)))
    :effect (wearing-shoe ?p ?sh)
  )
  
  (:action match-socks
    :parameters (?p - person ?s1 - sock ?s2 - sock)
    :precondition (and (has-sock ?p ?s1) (has-sock ?p ?s2) (sock-pair ?s1 ?s2) (not (wearing-sock ?p ?s1)) (not (wearing-sock ?p ?s2)))
    :effect (and (wearing-sock ?p ?s1) (wearing-sock ?p ?s2))
  )
)