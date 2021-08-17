(define 
    (domain dice-fine-plan)
    (:requirements :strips)
    (:predicates 
        (AtConfig ?loc) ;  true if finger at config ?loc
        (HasDice ?loc)  ;  true if dice is at ?loc
        (ValidMotion ?from ?t ?to) ; true if ?t is a valid motion from ?from to ?to
        (ValidPush ?from ?t ?to) ; true if ?t is a valid push from ?from to ?to
    )
    ;move finger on trajectory
    (:action move
        :parameters (?from ?t ?to)
        :precondition (and (ValidMotion ?from ?t ?to)
                        (AtConfig ?from))
        :effect (and (AtConfig ?to) 
                    (not (AtConfig ?from))))

    ;push dice
    (:action push
        :parameters (?from ?dice ?t ?to)
        :precondition (and (ValidPush ?from ?t ?to)
                        (AtConfig ?from)
                        (HasDice ?dice))
        :effect (and (AtConfig ?dice)
                    (HasDice ?to)
                    (not (HasDice ?dice))
                    (not (AtConfig ?from))))
  )