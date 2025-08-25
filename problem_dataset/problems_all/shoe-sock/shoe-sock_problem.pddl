(define (problem shoe-socks-problem)
  (:domain shoe-socks)
  
  (:objects
    alice bob - person
    sock1 sock2 sock3 sock4 - sock
    shoe1 shoe2 shoe3 shoe4 - shoe
  )
  
  (:init
    (not (has-sock alice sock1))
    (not (has-sock alice sock2))
    (not (has-sock bob sock3))
    (not (has-sock bob sock4))
    (not (has-shoe alice shoe1))
    (not (has-shoe alice shoe2))
    (not (has-shoe bob shoe3))
    (not (has-shoe bob shoe4))
    (sock-pair sock1 sock2)
    (sock-pair sock3 sock4)
  )
  
  (:goal
    (and
      (wearing-sock alice sock1)
      (wearing-sock alice sock2)
      (wearing-sock bob sock3)
      (wearing-sock bob sock4)
      (wearing-shoe alice shoe1)
      (wearing-shoe alice shoe2)
      (wearing-shoe bob shoe3)
      (wearing-shoe bob shoe4)
    )
  )
)