(define (problem test)
(:domain labyrinth)

(:objects
    pos0 pos1 pos2 - gridpos
    card1 card2 card3 card4 card5 card6 card7 card8 card9 - card
)

(:init
    (max-pos pos2)
    (min-pos pos0)
    (next pos1 pos0)
    (next pos2 pos1)
    
    (card-at card1 pos0 pos0)
    (card-at card2 pos0 pos1)
    (card-at card3 pos0 pos2)
    
    (card-at card5 pos1 pos0)
    (card-at card6 pos1 pos1)
    (card-at card4 pos1 pos2)
    
    (card-at card7 pos2 pos0)
    (card-at card8 pos2 pos1)
    (card-at card9 pos2 pos2)
    
    
    (blocked card1 N)
    (blocked card1 E)
    (blocked card1 W)
    
    (blocked card2 S)
    (blocked card2 W)

    (blocked card3 N)
    (blocked card3 S)

    (blocked card4 S)
    (blocked card4 W)

    (blocked card5 N)
    (blocked card5 S)

    (blocked card6 N)
    (blocked card6 E)

    (blocked card8 E)
    (blocked card8 W)

    (blocked card9 E)
    (blocked card9 W)
    
    
    (robot-at card1)

    (= (total-cost) 0)
    (= (move-robot-cost) 2)
    (= (move-card) 1)
)
(:goal
   (and
    (left)
   )
)
(:metric minimize (total-cost))
)

