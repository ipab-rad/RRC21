(define (stream dice-streams)
    (:stream move-stream
        :inputs (?finger ?from ?to)
        :domain (and (Finger ?finger) 
                    (Location ?from) 
                    (Location ?to)))
        :outputs (?t)
        :certified ( and (Trajectory ?t) 
                        (ValidMotion ?finger ?from ?t ?to))
    (:stream push-stream
        :inputs (?finger ?from_finger ?to_finger ?dice ?from_dice ?to_dice)
        :domain (and (Finger ?finger) 
                    (Location ?from_finger)
                    (Location ?to_finger)
                    (Dice ?dice)
                    (Location ?from_dice))
        :outputs (?t, ?to_dice)
        :certified (and (Trajectory ?t)
                        (Location ?to_dice)
                        (ValidPush ?finger 
                                    ?from_finger 
                                    ?to_finger 
                                    ?dice
                                    ?from_dice
                                    ?to_dice
                                    ?trajectory)))
  )