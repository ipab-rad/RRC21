(define 
    (domain dice)
    (:requirements :strips :typing)
    (:types
        location trajectory movable - object
        finger dice - movable
    )

    (:predicates 
        ;`Location` is true if x is a location
        (Location ?x)
        ;`Trajectory` is true if x is a trajectory
        (Trajectory ?x)
        ;`Finger` is true if x is a finger
        (Finger ?x)
        ;`Dice` is true id x is a dice
        (Dice ?x)
        ;`At` is true if ?x is at ?y
        (At ?x ?y )
        ;`ValidMotion` is true if ?trajectory is a valid motion for ?finger
        ;from ?from to ?to
        (ValidMotion ?finger 
                     ?from
                     ?to
                     ?trajectory) 
        ;`ValidPush` is true if ?trajectory is a valid push for ?finger
        ;from ?from_finger to ?to_finger to move ?dice from ?from_dice to ?to_dice
        (ValidPush ?finger
                   ?from_finger
                   ?to_finger
                   ?dice
                   ?from_dice
                   ?to_dice
                   ?trajectory) 
    )
    ;[move]
    (:action move
        :parameters (?finger 
                    ?from
                    ?trajectory
                    ?to) 
        :precondition (and  (Location ?from)
                            (Trajectory ?trajectory)
                            (Location ?to)
                            (Finger ?finger)
                            (ValidMotion ?finger ?from ?to ?trajectory) 
                            (At ?finger ?from))
        :effect (and (At ?finger ?to) 
                    (not (At ?finger ?from))))

    ;[push] 
    (:action push
        :parameters (?finger 
                    ?from_finger 
                    ?to_finger 
                    ?dice
                    ?from_dice
                    ?to_dice
                    ?trajectory)
        :precondition (and 
                        (Finger ?finger)
                        (Dice ?dice)
                        (Location ?from_finger)
                        (Location ?to_finger)
                        (Location ?from_dice)
                        (Location ?to_dice)
                        (Trajectory ?trajectory)
                        (ValidPush ?finger ?from_finger ?to_finger ?dice ?from_dice ?to_dice ?trajectory)
                        (At ?finger ?from_finger)
                        (At ?dice ?from_dice))
        :effect (and (At ?finger ?to_finger)
                    (At ?dice ?to_dice)
                    (not (At ?finger ?from_finger))
                    (not (At ?dice ?from_dice))))
  )